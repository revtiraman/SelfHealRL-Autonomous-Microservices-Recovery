"""Custom Stable-Baselines3 callbacks for SelfHealRL training."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MetricsCallback(BaseCallback):
    """Logs episode metrics to tensorboard and tracks best model."""

    def __init__(self, save_path: str = "models/best_model", verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self.episode_rewards: list = []
        self.episode_lengths: list = []

    def _on_step(self) -> bool:
        # Check for completed episodes
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)

                self.logger.record("episode/reward", ep_reward)
                self.logger.record("episode/length", ep_length)

                # Log running averages every 10 episodes
                if len(self.episode_rewards) % 10 == 0:
                    recent = self.episode_rewards[-10:]
                    mean_r = np.mean(recent)
                    self.logger.record("episode/mean_reward_10", mean_r)
                    self.logger.record("episode/total_episodes", len(self.episode_rewards))

                    if mean_r > self.best_mean_reward:
                        self.best_mean_reward = mean_r
                        self.model.save(self.save_path)
                        if self.verbose:
                            print(f"  New best model! Mean reward: {mean_r:.1f}")

        return True


class CurriculumCallback(BaseCallback):
    """Monitors performance and stops training early when success_rate > threshold.

    Prioritized phase advancement: instead of fixed timesteps, the agent
    advances to the next phase as soon as it demonstrates competence.
    """

    def __init__(self, success_threshold: float = 0.70, window: int = 50, verbose: int = 1):
        super().__init__(verbose)
        self.success_threshold = success_threshold
        self.window = window
        self.episode_successes: list = []
        self.ready_for_next = False

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                success = info["episode"]["r"] > 0
                self.episode_successes.append(float(success))

                if len(self.episode_successes) >= self.window:
                    recent_rate = np.mean(self.episode_successes[-self.window:])
                    self.logger.record("curriculum/success_rate", recent_rate)

                    if recent_rate >= self.success_threshold and not self.ready_for_next:
                        self.ready_for_next = True
                        if self.verbose:
                            print(
                                f"\n  *** Phase mastered! "
                                f"Success rate {recent_rate:.1%} >= {self.success_threshold:.0%} "
                                f"— stopping early and advancing ***\n"
                            )
                        return False  # stops model.learn() early

        return True
