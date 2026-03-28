"""Rule-based heuristic agent for SelfHealRL demos."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from config import ACTION_TYPES, ACTION_SUCCESS_RATES, SERVICE_NAMES, SERVICES


class HeuristicAgent:
    """A rule-based agent that actually fixes services intelligently.

    Priority logic:
      1. Observe a DOWN service if we haven't seen its failure type yet
      2. Fix upstream dependencies before downstream services
      3. Pick the best action for the known failure type
      4. If already recovering, wait (observe something else)
    """

    def __init__(self) -> None:
        self._observed: Dict[str, str] = {}   # service → failure_type
        self._recovering: set = set()          # services currently recovering

    def reset(self) -> None:
        self._observed = {}
        self._recovering = set()

    def act(self, obs_dict: dict) -> Tuple[str, str]:
        """Return (action_type, target_service) given current system state."""
        down = [s for s, d in obs_dict.items() if d["status"] <= 0.1]
        degraded = [s for s, d in obs_dict.items() if 0.1 < d["status"] < 0.9]
        recovering = [s for s, d in obs_dict.items() if d.get("recovering", False)]

        # Update recovering set
        self._recovering = set(recovering)

        # Step 1: observe any DOWN service we haven't diagnosed yet
        undiagnosed_down = [s for s in down if s not in self._observed]
        if undiagnosed_down:
            target = self._pick_upstream_first(undiagnosed_down, obs_dict)
            return "observe", target

        # Step 2: observe any DEGRADED service before it crashes
        undiagnosed_degraded = [s for s in degraded if s not in self._observed]
        if undiagnosed_degraded and not down:
            return "observe", undiagnosed_degraded[0]

        # Step 3: fix DOWN services — upstream first, best action
        fixable_down = [s for s in down if s not in self._recovering]
        if fixable_down:
            target = self._pick_upstream_first(fixable_down, obs_dict)
            action = self._best_action(target)
            return action, target

        # Step 4: prevent degraded from crashing
        fixable_degraded = [s for s in degraded if s not in self._recovering]
        if fixable_degraded:
            target = self._pick_upstream_first(fixable_degraded, obs_dict)
            action = self._best_action(target)
            return action, target

        # Step 5: nothing to do
        return "do_nothing", SERVICE_NAMES[0]

    def record_observation(self, service: str, failure_type: str) -> None:
        """Called after a successful observe action."""
        self._observed[service] = failure_type

    def _pick_upstream_first(self, candidates: List[str], obs_dict: dict) -> str:
        """Pick the service with no down dependencies (i.e., fix root first)."""
        for svc in candidates:
            deps = SERVICES.get(svc, {}).get("depends_on", [])
            if not any(obs_dict.get(d, {}).get("status", 1.0) <= 0.1 for d in deps):
                return svc
        # All have down deps — pick the one closest to root (fewest deps)
        return min(candidates, key=lambda s: len(SERVICES.get(s, {}).get("depends_on", [])))

    def _best_action(self, service: str) -> str:
        """Pick the highest success-rate action for this service's failure type."""
        failure_type = self._observed.get(service, None)
        if failure_type is None:
            return "restart"  # safe default

        best_action = "restart"
        best_rate = 0.0
        for action_type in ACTION_TYPES:
            if action_type in ("observe", "do_nothing"):
                continue
            rate = ACTION_SUCCESS_RATES.get((action_type, failure_type), 0.0)
            if rate > best_rate:
                best_rate = rate
                best_action = action_type
        return best_action

    def action_to_int(self, action_type: str, target_service: str) -> int:
        """Convert (action_type, target_service) to discrete action integer."""
        action_idx = ACTION_TYPES.index(action_type)
        service_idx = SERVICE_NAMES.index(target_service)
        return action_idx * len(SERVICE_NAMES) + service_idx
