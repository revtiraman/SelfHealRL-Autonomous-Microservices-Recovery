"""PPO training with curriculum learning for SelfHealRL.

Improvements over v1:
  1. RecurrentPPO (LSTM) — memory across steps for partial observability
  2. MaskablePPO — action masking removes illegal actions (~10x smaller effective space)
  3. More training steps — Phase3: 200k→500k, Phase4: 300k→600k
  4. Shaped observation — 104-dim (was 74) with derived features
  5. Prioritized phase advancement — advance only when success_rate > 70%
  6. 8 parallel envs — SubprocVecEnv for 8x more experience per update
"""

from __future__ import annotations

import os
import random
from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

try:
    from sb3_contrib import RecurrentPPO, MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks
    HAS_SB3_CONTRIB = True
except ImportError:
    from stable_baselines3 import PPO as RecurrentPPO
    from stable_baselines3 import PPO as MaskablePPO
    HAS_SB3_CONTRIB = False
    print("WARNING: sb3-contrib not found — falling back to standard PPO (no LSTM/masking)")

from env.selfheal_env import SelfHealEnv
from training.callbacks import CurriculumCallback, MetricsCallback
from training.evaluate import compare_agents, evaluate_agent


# ── Training mode selector ────────────────────────────────────────
# "recurrent"  → RecurrentPPO with LSTM (best for partial observability)
# "masked"     → MaskablePPO (illegal-action masking, no LSTM)
# "standard"   → plain PPO (fallback)
TRAINING_MODE = "recurrent" if HAS_SB3_CONTRIB else "standard"


class MixedDifficultyEnv(gym.Env):
    """Wraps SelfHealEnv and randomly samples difficulty each episode.

    Prevents catastrophic forgetting by replaying easier difficulties
    during harder training phases.
    """

    metadata = {"render_modes": []}

    def __init__(self, difficulties: list[tuple[str, float]], partial_observability: bool = True):
        super().__init__()
        self.difficulties = [d for d, _ in difficulties]
        self.weights = [w for _, w in difficulties]
        self.partial_observability = partial_observability
        sample_env = SelfHealEnv(difficulty=self.difficulties[0], partial_observability=partial_observability)
        self.observation_space = sample_env.observation_space
        self.action_space = sample_env.action_space
        self._env: Optional[SelfHealEnv] = None

    def _new_env(self) -> SelfHealEnv:
        diff = random.choices(self.difficulties, weights=self.weights, k=1)[0]
        return SelfHealEnv(difficulty=diff, partial_observability=self.partial_observability)

    def reset(self, seed=None, options=None):
        self._env = self._new_env()
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        return self._env.step(action)

    def action_masks(self):
        if self._env is not None:
            return self._env.action_masks()
        return np.ones(self.action_space.n, dtype=bool)

    def render(self):
        return self._env.render() if self._env else None

    def close(self):
        if self._env:
            self._env.close()


TRAINING_PHASES = {
    "phase1_easy": {
        "difficulty": "EASY",
        "partial_observability": False,
        "total_timesteps": 50_000,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "n_envs": 8,
        "success_threshold": 0.70,
        "description": "Learn basic recovery on easy scenarios",
    },
    "phase2_medium": {
        "difficulty": [("MEDIUM", 0.7), ("EASY", 0.3)],
        "partial_observability": False,
        "total_timesteps": 100_000,
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "n_envs": 8,
        "success_threshold": 0.70,
        "description": "Handle multi-service failures (70% MEDIUM + 30% EASY replay)",
    },
    "phase3_hard_partial": {
        "difficulty": [("HARD", 0.6), ("MEDIUM", 0.2), ("EASY", 0.2)],
        "partial_observability": True,
        "total_timesteps": 500_000,
        "learning_rate": 5e-5,
        "n_steps": 4096,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.995,
        "n_envs": 8,
        "success_threshold": 0.70,
        "description": "Hard scenarios with partial obs (60% HARD + 20% MED + 20% EASY)",
    },
    "phase4_chaos": {
        "difficulty": [("CHAOS", 0.5), ("HARD", 0.2), ("MEDIUM", 0.2), ("EASY", 0.1)],
        "partial_observability": True,
        "total_timesteps": 600_000,
        "learning_rate": 3e-5,
        "n_steps": 4096,
        "batch_size": 128,
        "n_epochs": 15,
        "gamma": 0.995,
        "n_envs": 8,
        "success_threshold": 0.70,
        "description": "Chaos + all difficulties replay (50/20/20/10 split)",
    },
}


class Trainer:
    """Manages PPO training with curriculum learning."""

    def __init__(self, model_dir: str = "models", log_dir: str = "logs") -> None:
        self.model_dir = model_dir
        self.log_dir = log_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    def _make_env(self, difficulty, partial_observability: bool):
        def _init():
            if isinstance(difficulty, list):
                env = MixedDifficultyEnv(difficulty, partial_observability=partial_observability)
            else:
                env = SelfHealEnv(difficulty=difficulty, partial_observability=partial_observability)
            return Monitor(env)
        return _init

    def _build_vec_env(self, difficulty, partial_observability: bool, n_envs: int):
        fns = [self._make_env(difficulty, partial_observability) for _ in range(n_envs)]
        # SubprocVecEnv for true parallelism; fall back to Dummy if n_envs==1
        if n_envs > 1:
            try:
                return SubprocVecEnv(fns, start_method="fork")
            except Exception:
                return DummyVecEnv(fns)
        return DummyVecEnv(fns)

    def _build_model(self, cfg: dict, env, prev_model_path: Optional[str] = None):
        """Build or load a model based on TRAINING_MODE."""
        if TRAINING_MODE == "recurrent" and HAS_SB3_CONTRIB:
            policy = "MlpLstmPolicy"
            policy_kwargs = dict(
                net_arch=[256, 256],
                lstm_hidden_size=128,
                enable_critic_lstm=True,
                n_lstm_layers=1,
            )
            ModelClass = RecurrentPPO
        elif TRAINING_MODE == "masked" and HAS_SB3_CONTRIB:
            policy = "MlpPolicy"
            policy_kwargs = dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]))
            ModelClass = MaskablePPO
        else:
            from stable_baselines3 import PPO
            policy = "MlpPolicy"
            policy_kwargs = dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]))
            ModelClass = PPO

        if prev_model_path and os.path.exists(prev_model_path):
            print(f"  Loading previous model: {prev_model_path}")
            model = ModelClass.load(prev_model_path, env=env)
            model.learning_rate = cfg["learning_rate"]
        else:
            model = ModelClass(
                policy,
                env,
                learning_rate=cfg["learning_rate"],
                n_steps=cfg["n_steps"],
                batch_size=cfg["batch_size"],
                n_epochs=cfg["n_epochs"],
                gamma=cfg["gamma"],
                policy_kwargs=policy_kwargs,
                verbose=1,
            )
        return model

    def train(
        self,
        phase_name: str,
        prev_model_path: Optional[str] = None,
    ) -> str:
        """Train a single phase. Returns path to saved model."""
        cfg = TRAINING_PHASES[phase_name]
        n_envs = cfg.get("n_envs", 8)

        print(f"\n{'='*60}")
        print(f"Training: {phase_name} — {cfg['description']}")
        print(f"  Difficulty: {cfg['difficulty']}")
        print(f"  Steps: {cfg['total_timesteps']:,} | Envs: {n_envs} | Mode: {TRAINING_MODE}")
        print(f"{'='*60}\n")

        env = self._build_vec_env(cfg["difficulty"], cfg["partial_observability"], n_envs)

        model_path = os.path.join(self.model_dir, f"{phase_name}.zip")
        best_path = os.path.join(self.model_dir, f"{phase_name}_best")

        model = self._build_model(cfg, env, prev_model_path)

        curriculum_cb = CurriculumCallback(
            success_threshold=cfg.get("success_threshold", 0.70),
            verbose=1,
        )
        callbacks = [
            MetricsCallback(save_path=best_path, verbose=1),
            curriculum_cb,
        ]

        model.learn(
            total_timesteps=cfg["total_timesteps"],
            callback=callbacks,
        )

        model.save(model_path)
        print(f"\n  Model saved: {model_path}")
        print(f"  Phase ready flag: {curriculum_cb.ready_for_next}")
        env.close()

        return model_path

    def train_curriculum(self) -> str:
        """Train through all 4 phases sequentially (curriculum learning)."""
        phases = list(TRAINING_PHASES.keys())
        prev_path: Optional[str] = None

        for phase in phases:
            prev_path = self.train(phase, prev_model_path=prev_path)

            # Eval after each phase
            diff_cfg = TRAINING_PHASES[phase]["difficulty"]
            eval_diff = diff_cfg[0][0] if isinstance(diff_cfg, list) else diff_cfg

            if TRAINING_MODE == "recurrent":
                model = RecurrentPPO.load(prev_path)
            elif TRAINING_MODE == "masked":
                model = MaskablePPO.load(prev_path)
            else:
                from stable_baselines3 import PPO
                model = PPO.load(prev_path)

            stats = evaluate_agent(model, num_episodes=20, difficulty=eval_diff)
            print(f"\n  Phase {phase} eval: success={stats['success_rate']:.0%}, "
                  f"reward={stats['mean_reward']:.1f}, grade={stats['mean_grade_score']:.2f}")

        # Save final model
        final_path = os.path.join(self.model_dir, "selfheal_agent_final.zip")
        if TRAINING_MODE == "recurrent":
            model = RecurrentPPO.load(prev_path)
        elif TRAINING_MODE == "masked":
            model = MaskablePPO.load(prev_path)
        else:
            from stable_baselines3 import PPO
            model = PPO.load(prev_path)
        model.save(final_path)
        print(f"\n{'='*60}")
        print(f"Curriculum training complete! Final model: {final_path}")
        print(f"Mode: {TRAINING_MODE}")
        print(f"{'='*60}")
        return final_path

    def evaluate(
        self,
        model_path: str,
        num_episodes: int = 100,
        difficulty: str = "MEDIUM",
    ) -> dict:
        if TRAINING_MODE == "recurrent":
            model = RecurrentPPO.load(model_path)
        elif TRAINING_MODE == "masked":
            model = MaskablePPO.load(model_path)
        else:
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
        return evaluate_agent(model, num_episodes, difficulty)

    def compare_with_baseline(
        self,
        model_path: str,
        num_episodes: int = 100,
        difficulty: str = "MEDIUM",
    ) -> dict:
        if TRAINING_MODE == "recurrent":
            model = RecurrentPPO.load(model_path)
        elif TRAINING_MODE == "masked":
            model = MaskablePPO.load(model_path)
        else:
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
        return compare_agents(model, num_episodes, difficulty)


if __name__ == "__main__":
    trainer = Trainer()
    model_path = trainer.train("phase1_easy")
    stats = trainer.evaluate(model_path, num_episodes=20, difficulty="EASY")
    print(f"\nEval results:")
    print(f"  Success rate: {stats['success_rate']:.0%}")
    print(f"  Mean reward: {stats['mean_reward']:.1f}")
    print(f"  Mean grade: {stats['mean_grade_score']:.2f}")
