"""Observation encoding — converts mesh state to flat numpy arrays for the RL agent."""

from __future__ import annotations

from typing import Dict, List, Set

import numpy as np

from config import (
    MAX_STEPS_PER_EPISODE,
    ACTION_BUDGET,
    ACTION_TYPES,
    FAILURE_TYPES,
    METRIC_NOISE_STD,
    NUM_SERVICES,
    NUM_ACTIONS,
    OBS_GLOBAL,
    OBS_PER_SERVICE,
    OBSERVATION_DIM,
    SERVICE_NAMES,
    STATUS_DEGRADED,
    STATUS_DOWN,
    SERVICES,
)
from env.service_mesh import ServiceMesh


# Failure type → encoded index (0–1 normalized)
_FAILURE_TYPE_IDX = {ft: (i + 1) / len(FAILURE_TYPES) for i, ft in enumerate(FAILURE_TYPES)}


class ObservationEncoder:
    """Encodes the service mesh state into a flat observation vector.

    Per-service features (10 each):
        0: observed flag
        1: cpu (noisy if observed, -1 if not)
        2: memory (noisy if observed, -1 if not)
        3: latency (normalized, noisy if observed, -1 if not)
        4: error_rate (noisy if observed, -1 if not)
        5: inferred_status
        6: alert_active (always visible)
        7: has_unmet_deps  (1.0 if any upstream dependency is DOWN)
        8: steps_since_observed (normalized 0-1; 0 = never observed)
        9: estimated_failure_type (normalized index; 0 = unknown)

    Global features (4):
        0: time_step (normalized 0-1)
        1: actions_remaining (normalized 0-1)
        2: system_health (0-1)
        3: active_alerts_count (normalized)
    """

    def __init__(self, partial_observability: bool = True) -> None:
        self.partial_observability = partial_observability
        self.observed_services: Set[str] = set()
        self._last_observed_step: Dict[str, int] = {}  # service → step when last observed

    def reset(self) -> None:
        self.observed_services.clear()
        self._last_observed_step.clear()

    def mark_observed(self, service_name: str, current_step: int = 0) -> None:
        self.observed_services.add(service_name)
        self._last_observed_step[service_name] = current_step

    def encode(
        self,
        mesh: ServiceMesh,
        current_step: int,
        actions_remaining: int,
        alerts: List[str],
    ) -> np.ndarray:
        obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)

        for i, name in enumerate(SERVICE_NAMES):
            offset = i * OBS_PER_SERVICE
            svc = mesh.services[name]
            has_alert = name in alerts
            can_see = not self.partial_observability or name in self.observed_services

            # Feature 0: observed flag
            obs[offset + 0] = 1.0 if can_see else 0.0

            if can_see:
                obs[offset + 1] = svc.noisy_cpu()
                obs[offset + 2] = svc.noisy_memory()
                obs[offset + 3] = min(1.0, svc.noisy_latency() / 10000.0)
                obs[offset + 4] = svc.noisy_error_rate()
                obs[offset + 5] = float(np.clip(
                    svc.status + np.random.normal(0, 0.02), 0, 1
                ))
            else:
                obs[offset + 1] = -1.0
                obs[offset + 2] = -1.0
                obs[offset + 3] = -1.0
                obs[offset + 4] = -1.0
                obs[offset + 5] = 0.3 if has_alert else -1.0

            # Feature 6: alert (always visible)
            obs[offset + 6] = 1.0 if has_alert else 0.0

            # Feature 7: has_unmet_deps — any upstream dependency is DOWN
            deps_down = any(
                mesh.services[dep].is_down
                for dep in svc.depends_on
                if dep in mesh.services
            )
            obs[offset + 7] = 1.0 if deps_down else 0.0

            # Feature 8: steps_since_observed (normalized; 0 = never seen)
            if name in self._last_observed_step:
                steps_since = current_step - self._last_observed_step[name]
                obs[offset + 8] = min(1.0, steps_since / MAX_STEPS_PER_EPISODE)
            else:
                obs[offset + 8] = 0.0

            # Feature 9: estimated_failure_type (from visible metrics)
            if can_see and svc.failure_type:
                obs[offset + 9] = _FAILURE_TYPE_IDX.get(svc.failure_type, 0.0)
            elif can_see and svc.cpu > 0.85:
                obs[offset + 9] = _FAILURE_TYPE_IDX.get("cpu_spike", 0.0)
            elif can_see and svc.memory > 0.85:
                obs[offset + 9] = _FAILURE_TYPE_IDX.get("memory_leak", 0.0)
            else:
                obs[offset + 9] = 0.0

        # Global features
        global_offset = NUM_SERVICES * OBS_PER_SERVICE
        obs[global_offset + 0] = current_step / MAX_STEPS_PER_EPISODE
        obs[global_offset + 1] = actions_remaining / ACTION_BUDGET
        obs[global_offset + 2] = mesh.system_health()
        obs[global_offset + 3] = min(1.0, len(alerts) / NUM_SERVICES)

        return obs

    def get_action_mask(self, mesh: ServiceMesh) -> np.ndarray:
        """Return boolean mask of valid actions (True = allowed).

        Masked out (False):
          - restart/scale_up/reroute/rollback on a HEALTHY, non-recovering service
          - observe on an already-observed service (when partial_observability)
          - do_nothing on any service when services are DOWN
        """
        mask = np.ones(NUM_ACTIONS, dtype=bool)
        down_services = mesh.get_down_services()

        action_type_indices = {at: i for i, at in enumerate(ACTION_TYPES)}
        fix_actions = {"restart", "scale_up", "reroute", "rollback"}

        for svc_idx, name in enumerate(SERVICE_NAMES):
            svc = mesh.services[name]

            # Mask fix actions on healthy non-recovering services
            if svc.is_healthy and not svc.recovering:
                for at in fix_actions:
                    action_int = action_type_indices[at] * NUM_SERVICES + svc_idx
                    mask[action_int] = False

            # Mask re-observe on already-observed services (partial obs only)
            if self.partial_observability and name in self.observed_services:
                obs_idx = action_type_indices["observe"] * NUM_SERVICES + svc_idx
                mask[obs_idx] = False

        # Mask do_nothing entirely when any service is DOWN
        if down_services:
            dn_idx = action_type_indices["do_nothing"]
            for svc_idx in range(NUM_SERVICES):
                mask[dn_idx * NUM_SERVICES + svc_idx] = False

        # Safety: always allow at least one action
        if not mask.any():
            mask[:] = True

        return mask

    def get_alerts(self, mesh: ServiceMesh) -> List[str]:
        alerts: List[str] = []
        for name, svc in mesh.services.items():
            if svc.error_rate > 0.3 or svc.latency > svc.base_latency * 5:
                alerts.append(name)
            elif svc.is_down:
                alerts.append(name)
        return alerts
