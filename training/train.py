"""PPO training with curriculum learning for SelfHealRL."""

from __future__ import annotations

import os
import random
from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env.selfheal_env import SelfHealEnv
from training.callbacks import CurriculumCallback, MetricsCallback
from training.evaluate import compare_agents, evaluate_agent


class MixedDifficultyEnv(gym.Env):
    """Wraps SelfHealEnv and randomly samples difficulty each episode.

    Prevents catastrophic forgetting by replaying easier difficulties
    during harder training phases.
    """

    metadata = {"render_modes": []}

    def __init__(self, difficulties: list[tuple[str, float]], partial_observability: bool = True):
        """
        difficulties: list of (difficulty_name, probability) tuples, must sum to 1.0
        """
        super().__init__()
        self.difficulties = [d for d, _ in difficulties]
        self.weights = [w for _, w in difficulties]
        self.partial_observability = partial_observability
        # Build a representative env to get spaces
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
        "description": "Learn basic recovery on easy scenarios",
    },
    "phase2_medium": {
        # Mix: 70% MEDIUM, 30% EASY replay to prevent forgetting
        "difficulty": [("MEDIUM", 0.7), ("EASY", 0.3)],
        "partial_observability": False,
        "total_timesteps": 100_000,
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "description": "Handle multi-service failures (70% MEDIUM + 30% EASY replay)",
    },
    "phase3_hard_partial": {
        # Mix: 60% HARD, 20% MEDIUM, 20% EASY replay
        "difficulty": [("HARD", 0.6), ("MEDIUM", 0.2), ("EASY", 0.2)],
        "partial_observability": True,
        "total_timesteps": 200_000,
        "learning_rate": 5e-5,
        "n_steps": 4096,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.995,
        "description": "Hard scenarios with partial obs (60% HARD + 20% MED + 20% EASY)",
    },
    "phase4_chaos": {
        # Mix: 50% CHAOS, 20% HARD, 20% MEDIUM, 10% EASY replay
        "difficulty": [("CHAOS", 0.5), ("HARD", 0.2), ("MEDIUM", 0.2), ("EASY", 0.1)],
        "partial_observability": True,
        "total_timesteps": 300_000,
        "learning_rate": 3e-5,
        "n_steps": 4096,
        "batch_size": 128,
        "n_epochs": 15,
        "gamma": 0.995,
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

    def train(
        self,
        phase_name: str,
        prev_model_path: Optional[str] = None,
    ) -> str:
        """Train a single phase. Returns path to saved model."""
        cfg = TRAINING_PHASES[phase_name]
        print(f"\n{'='*60}")
        print(f"Training: {phase_name} — {cfg['description']}")
        print(f"  Difficulty: {cfg['difficulty']}, Steps: {cfg['total_timesteps']:,}")
        print(f"{'='*60}\n")

        env = DummyVecEnv([self._make_env(cfg["difficulty"], cfg["partial_observability"])])

        model_path = os.path.join(self.model_dir, f"{phase_name}.zip")
        best_path = os.path.join(self.model_dir, f"{phase_name}_best")

        if prev_model_path and os.path.exists(prev_model_path):
            print(f"  Loading previous model: {prev_model_path}")
            model = PPO.load(prev_model_path, env=env)
            model.learning_rate = cfg["learning_rate"]
        else:
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=cfg["learning_rate"],
                n_steps=cfg["n_steps"],
                batch_size=cfg["batch_size"],
                n_epochs=cfg["n_epochs"],
                gamma=cfg["gamma"],
                policy_kwargs=dict(
                    net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])
                ),
                verbose=1,
            )

        callbacks = [
            MetricsCallback(save_path=best_path, verbose=1),
            CurriculumCallback(verbose=1),
        ]

        model.learn(
            total_timesteps=cfg["total_timesteps"],
            callback=callbacks,
        )

        model.save(model_path)
        print(f"\n  Model saved: {model_path}")
        env.close()

        return model_path

    def train_curriculum(self) -> str:
        """Train through all 4 phases sequentially (curriculum learning)."""
        phases = list(TRAINING_PHASES.keys())
        prev_path: Optional[str] = None

        for phase in phases:
            prev_path = self.train(phase, prev_model_path=prev_path)

            # Quick eval after each phase (always eval on single difficulty)
            model = PPO.load(prev_path)
            diff_cfg = TRAINING_PHASES[phase]["difficulty"]
            eval_diff = diff_cfg[0][0] if isinstance(diff_cfg, list) else diff_cfg
            stats = evaluate_agent(model, num_episodes=20, difficulty=eval_diff)
            print(f"\n  Phase {phase} eval: success={stats['success_rate']:.0%}, "
                  f"reward={stats['mean_reward']:.1f}, grade={stats['mean_grade_score']:.2f}")

        # Save final model
        final_path = os.path.join(self.model_dir, "selfheal_agent_final.zip")
        model = PPO.load(prev_path)
        model.save(final_path)
        print(f"\n{'='*60}")
        print(f"Curriculum training complete! Final model: {final_path}")
        print(f"{'='*60}")
        return final_path

    def evaluate(
        self,
        model_path: str,
        num_episodes: int = 100,
        difficulty: str = "MEDIUM",
    ) -> dict:
        """Evaluate a trained model."""
        model = PPO.load(model_path)
        return evaluate_agent(model, num_episodes, difficulty)

    def compare_with_baseline(
        self,
        model_path: str,
        num_episodes: int = 100,
        difficulty: str = "MEDIUM",
    ) -> dict:
        """Compare trained model with random baseline."""
        model = PPO.load(model_path)
        return compare_agents(model, num_episodes, difficulty)


if __name__ == "__main__":
    trainer = Trainer()
    # Quick training test — just phase 1
    model_path = trainer.train("phase1_easy")
    stats = trainer.evaluate(model_path, num_episodes=20, difficulty="EASY")
    print(f"\nEval results:")
    print(f"  Success rate: {stats['success_rate']:.0%}")
    print(f"  Mean reward: {stats['mean_reward']:.1f}")
    print(f"  Mean grade: {stats['mean_grade_score']:.2f}")

    comparison = trainer.compare_with_baseline(model_path, num_episodes=20, difficulty="EASY")
    print(f"\nTrained: success={comparison['trained']['success_rate']:.0%}, reward={comparison['trained']['mean_reward']:.1f}")
    print(f"Random:  success={comparison['random']['success_rate']:.0%}, reward={comparison['random']['mean_reward']:.1f}")
