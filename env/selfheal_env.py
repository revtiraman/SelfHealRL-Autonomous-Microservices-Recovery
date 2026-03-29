"""SelfHealEnv — Gymnasium RL environment for autonomous microservice recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import (
    ACTION_BUDGET,
    ACTION_TYPES,
    MAX_STEPS_PER_EPISODE,
    NUM_ACTION_TYPES,
    NUM_ACTIONS,
    NUM_SERVICES,
    OBSERVATION_DIM,
    SERVICE_NAMES,
)
from env.cascade_simulator import CascadeSimulator
from env.failure_engine import FailureEngine, Scenario
from env.observations import ObservationEncoder
from env.service_mesh import ServiceMesh


@dataclass
class StepRecord:
    """Record of a single step in an episode."""
    step: int
    action_type: str
    target_service: str
    action_success: bool
    action_description: str
    reward: float
    system_state: Dict[str, dict]
    down_services: List[str]
    degraded_services: List[str]
    healthy_services: List[str]
    system_health: float


class SelfHealEnv(gym.Env):
    """Gymnasium environment for microservice failure recovery.

    The agent must diagnose and fix cascading failures in a service mesh
    by taking recovery actions (restart, scale_up, rollback, etc.) in
    the correct dependency order.
    """

    metadata = {"render_modes": ["human", "dict"]}

    def __init__(
        self,
        difficulty: str = "MEDIUM",
        partial_observability: bool = True,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.difficulty = difficulty
        self.partial_observability = partial_observability
        self.render_mode = render_mode

        # Observation: flat vector of shape (74,)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBSERVATION_DIM,), dtype=np.float32
        )

        # Action: Discrete(60) = 6 action_types × 10 services
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Components
        self.mesh = ServiceMesh()
        self.failure_engine = FailureEngine()
        self.cascade_sim = CascadeSimulator()
        self.obs_encoder = ObservationEncoder(partial_observability)

        # Episode state
        self.current_step: int = 0
        self.actions_remaining: int = ACTION_BUDGET
        self.episode_history: List[StepRecord] = []
        self.scenario: Optional[Scenario] = None
        self.total_reward: float = 0.0
        self._prev_down: set = set()
        self._prev_degraded: set = set()
        self._rewarded_recovery: set = set()  # prevent reward-farming same service
        self._observed_this_episode: set = set()  # track first vs repeat observes

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Re-seed components
        if seed is not None:
            rng = np.random.default_rng(seed)
            self.mesh.rng = rng
            self.failure_engine.rng = rng

        # Reset all components
        self.mesh.reset()
        self.cascade_sim.reset()
        self.obs_encoder.reset()
        self.current_step = 0
        self.actions_remaining = ACTION_BUDGET
        self.episode_history = []
        self.total_reward = 0.0
        self._rewarded_recovery = set()
        self._observed_this_episode = set()

        # Generate and apply failure scenario
        if self.difficulty == "CHAOS":
            self.scenario = self.failure_engine.generate_chaos()
        else:
            self.scenario = self.failure_engine.generate_scenario(self.difficulty)

        self.failure_engine.apply_scenario(self.mesh, self.scenario)

        # Record root causes in cascade simulator
        for svc_name, _ in self.scenario.root_failures:
            self.cascade_sim.record_root_cause(svc_name)

        # Snapshot initial state
        self._prev_down = set(self.mesh.get_down_services())
        self._prev_degraded = set(self.mesh.get_degraded_services())

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action_type, target_service = self._decode_action(action)

        # ── Apply action ──────────────────────────────────────
        is_observe = action_type == "observe"
        is_noop = action_type == "do_nothing"

        if is_observe:
            self.obs_encoder.mark_observed(target_service)
            action_success = True
            action_desc = f"Observed {target_service}"
        elif is_noop:
            action_success = True
            action_desc = "No action taken"
        else:
            if self.actions_remaining <= 0:
                action_success = False
                action_desc = "No actions remaining — budget exhausted"
            else:
                self.actions_remaining -= 1
                action_success, action_desc = self.mesh.apply_action(
                    action_type, target_service
                )

        # ── Advance simulation ────────────────────────────────
        mesh_events = self.mesh.tick()
        cascade_events = self.cascade_sim.cascade_step(self.mesh, self.current_step)

        # ── Calculate reward ──────────────────────────────────
        reward = self._calculate_reward(
            action_type, target_service, action_success, is_observe, is_noop
        )
        self.total_reward += reward

        # ── Record step ───────────────────────────────────────
        record = StepRecord(
            step=self.current_step,
            action_type=action_type,
            target_service=target_service,
            action_success=action_success,
            action_description=action_desc,
            reward=reward,
            system_state=self.mesh.get_all_statuses(),
            down_services=self.mesh.get_down_services(),
            degraded_services=self.mesh.get_degraded_services(),
            healthy_services=self.mesh.get_healthy_services(),
            system_health=self.mesh.system_health(),
        )
        self.episode_history.append(record)

        # Update previous state
        self._prev_down = set(self.mesh.get_down_services())
        self._prev_degraded = set(self.mesh.get_degraded_services())

        self.current_step += 1

        # ── Check termination ─────────────────────────────────
        terminated = False
        truncated = False

        if self.mesh.is_fully_recovered():
            terminated = True
            reward += 20.0  # full recovery bonus
            self.total_reward += 20.0
        elif self.mesh.all_down():
            terminated = True
            reward -= 15.0
            self.total_reward -= 15.0
        elif self.current_step >= MAX_STEPS_PER_EPISODE:
            truncated = True
            reward -= 10.0
            self.total_reward -= 10.0

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    # ── Observation ───────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        alerts = self.obs_encoder.get_alerts(self.mesh)
        return self.obs_encoder.encode(
            self.mesh, self.current_step, self.actions_remaining, alerts
        )

    # ── Action decoding ───────────────────────────────────────

    def _decode_action(self, action_int: int) -> Tuple[str, str]:
        action_type_idx = action_int // NUM_SERVICES
        service_idx = action_int % NUM_SERVICES
        return ACTION_TYPES[action_type_idx], SERVICE_NAMES[service_idx]

    @staticmethod
    def encode_action(action_type: str, service_name: str) -> int:
        return ACTION_TYPES.index(action_type) * NUM_SERVICES + SERVICE_NAMES.index(service_name)

    # ── Reward ────────────────────────────────────────────────

    def _calculate_reward(
        self,
        action_type: str,
        target_service: str,
        success: bool,
        is_observe: bool,
        is_noop: bool,
    ) -> float:
        reward = 0.0
        svc = self.mesh.services[target_service]
        current_down = set(self.mesh.get_down_services())
        current_degraded = set(self.mesh.get_degraded_services())

        # Time penalty
        reward -= 1.0

        if is_noop:
            down_count = len(self.mesh.get_down_services())
            if down_count > 0:
                reward -= 3.0 * down_count  # urgency: can't do nothing while things are on fire
            return reward

        if is_observe:
            if target_service not in self._observed_this_episode:
                self._observed_this_episode.add(target_service)
                reward += 1.0
            else:
                if self.mesh.get_down_services():
                    reward -= 2.0
            return reward

        # Service recovery: only reward the FIRST recovery per service per episode
        if success and svc.recovering and target_service not in self._rewarded_recovery:
            self._rewarded_recovery.add(target_service)
            reward += 10.0

            # Root cause bonus
            root_services = {s for s, _ in self.scenario.root_failures}
            if target_service in root_services:
                reward += 15.0

            # Early intervention: bonus if service was only degraded (not down yet)
            if target_service in self._prev_degraded and target_service not in self._prev_down:
                reward += 8.0  # preventive action bonus

        # Wasted action: targeted a healthy service
        if svc.is_healthy and not svc.recovering:
            reward -= 3.0

        # Wrong dependency order: tried to fix downstream while upstream still down
        if not is_observe and not is_noop:
            deps_down = [d for d in svc.depends_on if self.mesh.services[d].is_down]
            if deps_down:
                reward -= 5.0

        # Action failed
        if not success and not svc.is_healthy:
            reward -= 2.0

        # New cascades caused this step
        new_down = current_down - self._prev_down
        if new_down:
            reward -= 5.0 * len(new_down)

        return reward

    # ── Info ──────────────────────────────────────────────────

    def _get_info(self) -> dict:
        return {
            "step": self.current_step,
            "system_health": self.mesh.system_health(),
            "down_services": self.mesh.get_down_services(),
            "degraded_services": self.mesh.get_degraded_services(),
            "actions_remaining": self.actions_remaining,
            "total_reward": self.total_reward,
            "scenario": str(self.scenario) if self.scenario else None,
        }

    # ── Render ────────────────────────────────────────────────

    def render(self) -> Optional[dict]:
        if self.render_mode == "dict":
            return self.mesh.get_all_statuses()
        elif self.render_mode == "human":
            health = self.mesh.system_health()
            down = self.mesh.get_down_services()
            degraded = self.mesh.get_degraded_services()
            print(
                f"Step {self.current_step} | Health: {health:.0%} | "
                f"DOWN: {down} | DEGRADED: {degraded} | "
                f"Budget: {self.actions_remaining}"
            )
        return None

    # ── Episode data ──────────────────────────────────────────

    def get_episode_history(self) -> List[StepRecord]:
        return self.episode_history

    def get_episode_summary(self) -> dict:
        """Summary dict suitable for grading."""
        return {
            "steps": self.current_step,
            "total_reward": self.total_reward,
            "final_health": self.mesh.system_health(),
            "fully_recovered": self.mesh.is_fully_recovered(),
            "scenario": self.scenario,
            "history": self.episode_history,
            "root_causes": [s for s, _ in self.scenario.root_failures] if self.scenario else [],
            "cascade_history": self.cascade_sim.cascade_history,
        }


if __name__ == "__main__":
    # Quick validation
    env = SelfHealEnv(difficulty="EASY", partial_observability=False)

    # SB3 compatibility check
    try:
        from stable_baselines3.common.env_checker import check_env
        check_env(env)
        print("SB3 environment check PASSED!")
    except ImportError:
        print("stable-baselines3 not installed — skipping SB3 check")

    # Run a random episode
    obs, info = env.reset(seed=42)
    print(f"\nScenario: {info['scenario']}")
    print(f"Initial health: {info['system_health']:.0%}")

    total_reward = 0.0
    for step in range(MAX_STEPS_PER_EPISODE):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if step < 10 or terminated or truncated:
            at, ts = env._decode_action(action)
            print(
                f"  Step {step}: {at}({ts}) → reward={reward:.1f}, "
                f"health={info['system_health']:.0%}"
            )
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            print(f"Total reward: {total_reward:.1f}")
            print(f"Final health: {info['system_health']:.0%}")
            print(f"Fully recovered: {env.mesh.is_fully_recovered()}")
            break
