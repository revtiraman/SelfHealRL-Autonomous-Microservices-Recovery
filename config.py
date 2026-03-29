"""SelfHealRL — All hyperparameters and configuration."""

from typing import Dict, List

# ─────────────────────────────────────────────
# Service Definitions
# ─────────────────────────────────────────────

SERVICES: Dict[str, dict] = {
    "api-gateway": {
        "base_cpu": 0.30,
        "base_memory": 0.40,
        "base_latency": 50.0,
        "depends_on": ["auth-service"],
        "max_instances": 3,
        "recovery_time": 2,
    },
    "auth-service": {
        "base_cpu": 0.25,
        "base_memory": 0.35,
        "base_latency": 30.0,
        "depends_on": ["user-db", "cache-layer"],
        "max_instances": 3,
        "recovery_time": 2,
    },
    "payment-service": {
        "base_cpu": 0.35,
        "base_memory": 0.45,
        "base_latency": 80.0,
        "depends_on": ["auth-service", "order-db"],
        "max_instances": 2,
        "recovery_time": 3,
    },
    "order-service": {
        "base_cpu": 0.20,
        "base_memory": 0.30,
        "base_latency": 40.0,
        "depends_on": ["order-db", "cache-layer"],
        "max_instances": 2,
        "recovery_time": 2,
    },
    "search-service": {
        "base_cpu": 0.40,
        "base_memory": 0.50,
        "base_latency": 60.0,
        "depends_on": ["restaurant-db", "cache-layer"],
        "max_instances": 2,
        "recovery_time": 2,
    },
    "notification-service": {
        "base_cpu": 0.15,
        "base_memory": 0.20,
        "base_latency": 100.0,
        "depends_on": ["user-db"],
        "max_instances": 1,
        "recovery_time": 1,
    },
    "user-db": {
        "base_cpu": 0.50,
        "base_memory": 0.60,
        "base_latency": 10.0,
        "depends_on": [],
        "max_instances": 1,
        "recovery_time": 4,
    },
    "cache-layer": {
        "base_cpu": 0.20,
        "base_memory": 0.70,
        "base_latency": 5.0,
        "depends_on": [],
        "max_instances": 1,
        "recovery_time": 1,
    },
    "restaurant-db": {
        "base_cpu": 0.45,
        "base_memory": 0.55,
        "base_latency": 15.0,
        "depends_on": [],
        "max_instances": 1,
        "recovery_time": 4,
    },
    "order-db": {
        "base_cpu": 0.40,
        "base_memory": 0.50,
        "base_latency": 12.0,
        "depends_on": [],
        "max_instances": 1,
        "recovery_time": 4,
    },
}

SERVICE_NAMES: List[str] = list(SERVICES.keys())
NUM_SERVICES: int = len(SERVICE_NAMES)

# ─────────────────────────────────────────────
# Service Status Constants
# ─────────────────────────────────────────────

STATUS_HEALTHY = 1.0
STATUS_DEGRADED = 0.5
STATUS_DOWN = 0.0

# ─────────────────────────────────────────────
# Failure Types
# ─────────────────────────────────────────────

FAILURE_TYPES: List[str] = [
    "memory_leak",
    "cpu_spike",
    "connection_timeout",
    "disk_full",
    "bad_deployment",
    "network_partition",
]

# How many steps before each failure type crashes the service
FAILURE_PROGRESSION_STEPS: Dict[str, int] = {
    "memory_leak": 5,
    "cpu_spike": 2,
    "connection_timeout": 3,
    "disk_full": 4,
    "bad_deployment": 3,
    "network_partition": 1,
}

# ─────────────────────────────────────────────
# Action Types
# ─────────────────────────────────────────────

ACTION_TYPES: List[str] = [
    "restart",
    "scale_up",
    "reroute",
    "rollback",
    "observe",
    "do_nothing",
]

NUM_ACTION_TYPES: int = len(ACTION_TYPES)
NUM_ACTIONS: int = NUM_ACTION_TYPES * NUM_SERVICES  # 6 * 10 = 60

# Action success rates: (action_type, failure_type) → probability
ACTION_SUCCESS_RATES: Dict[tuple, float] = {
    # restart
    ("restart", "memory_leak"): 0.60,
    ("restart", "cpu_spike"): 0.75,
    ("restart", "connection_timeout"): 0.80,
    ("restart", "disk_full"): 0.20,
    ("restart", "bad_deployment"): 0.30,
    ("restart", "network_partition"): 0.50,
    # scale_up
    ("scale_up", "memory_leak"): 0.70,
    ("scale_up", "cpu_spike"): 0.85,
    ("scale_up", "connection_timeout"): 0.30,
    ("scale_up", "disk_full"): 0.10,
    ("scale_up", "bad_deployment"): 0.20,
    ("scale_up", "network_partition"): 0.10,
    # reroute
    ("reroute", "memory_leak"): 0.40,
    ("reroute", "cpu_spike"): 0.50,
    ("reroute", "connection_timeout"): 0.70,
    ("reroute", "disk_full"): 0.30,
    ("reroute", "bad_deployment"): 0.40,
    ("reroute", "network_partition"): 0.80,
    # rollback
    ("rollback", "memory_leak"): 0.40,
    ("rollback", "cpu_spike"): 0.30,
    ("rollback", "connection_timeout"): 0.50,
    ("rollback", "disk_full"): 0.20,
    ("rollback", "bad_deployment"): 0.95,
    ("rollback", "network_partition"): 0.30,
    # observe — always succeeds (information gathering)
    ("observe", "memory_leak"): 1.0,
    ("observe", "cpu_spike"): 1.0,
    ("observe", "connection_timeout"): 1.0,
    ("observe", "disk_full"): 1.0,
    ("observe", "bad_deployment"): 1.0,
    ("observe", "network_partition"): 1.0,
    # do_nothing — no effect
    ("do_nothing", "memory_leak"): 0.0,
    ("do_nothing", "cpu_spike"): 0.0,
    ("do_nothing", "connection_timeout"): 0.0,
    ("do_nothing", "disk_full"): 0.0,
    ("do_nothing", "bad_deployment"): 0.0,
    ("do_nothing", "network_partition"): 0.0,
}

# ─────────────────────────────────────────────
# Reward Weights
# ─────────────────────────────────────────────

REWARD_CONFIG: Dict[str, float] = {
    "service_recovered": 10.0,
    "full_recovery": 20.0,
    "root_cause_fixed": 15.0,
    "cascade_caused": -5.0,
    "wasted_action": -3.0,
    "wrong_order": -5.0,
    "time_penalty": -1.0,
    "early_intervention_bonus": 5.0,
    "preventive_action_bonus": 8.0,
    "observe_bonus": 1.0,
}

# ─────────────────────────────────────────────
# Environment Settings
# ─────────────────────────────────────────────

MAX_STEPS_PER_EPISODE: int = 30
ACTION_BUDGET: int = 10  # max non-observe actions per episode

# Observation: 10 values per service + 4 global = 104
# Per-service: observed, cpu, memory, latency, error_rate, status, alert,
#              has_unmet_deps, steps_since_observed (normalized), est_failure_type
OBS_PER_SERVICE: int = 10
OBS_GLOBAL: int = 4
OBSERVATION_DIM: int = OBS_PER_SERVICE * NUM_SERVICES + OBS_GLOBAL  # 104

# Noise
METRIC_NOISE_STD: float = 0.05

# ─────────────────────────────────────────────
# Difficulty Presets
# ─────────────────────────────────────────────

DIFFICULTY_PRESETS: Dict[str, dict] = {
    "EASY": {
        "num_root_failures": 1,
        "max_cascade_depth": 2,
        "failure_speed": 0.5,  # slower progression
    },
    "MEDIUM": {
        "num_root_failures": (1, 2),
        "max_cascade_depth": 4,
        "failure_speed": 1.0,
    },
    "HARD": {
        "num_root_failures": (2, 3),
        "max_cascade_depth": 6,
        "failure_speed": 1.5,
    },
    "CHAOS": {
        "num_root_failures": (2, 4),
        "max_cascade_depth": 10,
        "failure_speed": 2.0,
    },
}
