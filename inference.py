"""SelfHealRL — Baseline inference script (OpenEnv Hackathon).

Connects to the running SelfHealRL environment container via HTTP and runs
an LLM agent against all 3 tasks.

Environment variables:
  API_BASE_URL   — LLM API endpoint  (default: https://router.huggingface.co/v1)
  MODEL_NAME     — Model identifier   (default: gpt-4o-mini)
  HF_TOKEN       — HuggingFace / API key (required, no default)
  ENV_URL        — SelfHealRL env URL (default: http://localhost:8000)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

import requests
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────
# Config from env vars
# ─────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:8000").rstrip("/")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable not set.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ─────────────────────────────────────────────────────────────────
# Task definitions (mirrored from openenv.yaml)
# ─────────────────────────────────────────────────────────────────

TASKS = [
    {"task_id": "task_easy",   "passing_score": 0.7},
    {"task_id": "task_medium", "passing_score": 0.6},
    {"task_id": "task_hard",   "passing_score": 0.5},
]

MAX_STEPS = 30

# ─────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an autonomous SRE (Site Reliability Engineer) agent.
Your job is to diagnose and fix cascading failures in a microservice system.

SERVICE DEPENDENCY GRAPH:
  api-gateway depends on: auth-service
  auth-service depends on: user-db, cache-layer
  payment-service depends on: auth-service, order-db
  order-service depends on: order-db, cache-layer
  search-service depends on: restaurant-db, cache-layer
  notification-service depends on: user-db
  user-db, cache-layer, restaurant-db, order-db: no dependencies (root services)

RULES:
1. Fix upstream services (dependencies) BEFORE downstream services.
2. OBSERVE a service before restarting/fixing it.
3. Best action per failure type:
   - memory_leak -> scale_up
   - cpu_spike -> scale_up or restart
   - bad_deployment -> rollback
   - network_partition -> reroute
   - disk_full -> reroute or restart
   - connection_timeout -> reroute or restart
4. Do NOT act on a healthy service.
5. Do NOT do_nothing when services are down.

AVAILABLE ACTIONS: restart, scale_up, reroute, rollback, observe, do_nothing
AVAILABLE SERVICES: api-gateway, auth-service, payment-service, order-service,
  search-service, notification-service, user-db, cache-layer, restaurant-db, order-db

Respond with ONLY valid JSON:
{"action_type": "<action>", "target_service": "<service>", "reasoning": "<1 sentence>"}
"""

VALID_ACTIONS  = ["restart", "scale_up", "reroute", "rollback", "observe", "do_nothing"]
VALID_SERVICES = [
    "api-gateway", "auth-service", "payment-service", "order-service",
    "search-service", "notification-service", "user-db", "cache-layer",
    "restaurant-db", "order-db",
]

# Dependency graph for heuristic fallback
DEPENDS_ON = {
    "api-gateway":          ["auth-service"],
    "auth-service":         ["user-db", "cache-layer"],
    "payment-service":      ["auth-service", "order-db"],
    "order-service":        ["order-db", "cache-layer"],
    "search-service":       ["restaurant-db", "cache-layer"],
    "notification-service": ["user-db"],
    "user-db": [], "cache-layer": [], "restaurant-db": [], "order-db": [],
}

# ─────────────────────────────────────────────────────────────────
# HTTP helpers — with retry
# ─────────────────────────────────────────────────────────────────

def _http_post(url: str, retries: int = 3, **kwargs) -> dict:
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.post(url, timeout=30, **kwargs)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"POST {url} failed after {retries} attempts: {last_err}")


def env_reset(task_id: str, seed: int = 42) -> dict:
    return _http_post(f"{ENV_URL}/reset/{task_id}", params={"seed": seed})


def env_step(action_type: str, target_service: str) -> dict:
    return _http_post(
        f"{ENV_URL}/step",
        json={"action_type": action_type, "target_service": target_service},
    )


def env_health() -> bool:
    try:
        r = requests.get(f"{ENV_URL}/health", timeout=10)
        return r.status_code == 200
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────────
# Heuristic fallback agent (used when LLM is unavailable)
# ─────────────────────────────────────────────────────────────────

_heuristic_observed: dict = {}
_heuristic_tried: set = set()

def heuristic_action(obs: dict) -> tuple[str, str]:
    """Rule-based agent: observe → fix upstream first → best action."""
    services = {s["name"]: s for s in obs.get("services", [])}
    down     = [n for n, s in services.items() if s.get("status", 1) < 0.1]
    degraded = [n for n, s in services.items() if 0.1 <= s.get("status", 1) < 0.9]

    # Observe any down service not yet observed
    for svc in down:
        if not services[svc].get("observed", False):
            return "observe", svc

    # Find a fixable service: upstream deps are all healthy
    fixable = []
    for svc in down:
        deps_down = [d for d in DEPENDS_ON.get(svc, []) if d in down]
        if not deps_down:
            fixable.append(svc)

    if not fixable:
        fixable = down  # fallback: try even with deps down

    if fixable:
        target = fixable[0]
        ft = services[target].get("failure_type", "") or ""
        if ft == "bad_deployment":
            action = "rollback"
        elif ft == "network_partition":
            action = "reroute"
        elif ft in ("memory_leak",):
            action = "scale_up"
        elif ft == "cpu_spike":
            action = "scale_up"
        else:
            action = "restart"
        return action, target

    # Fix degraded services
    for svc in degraded:
        deps_down = [d for d in DEPENDS_ON.get(svc, []) if d in down]
        if not deps_down:
            return "restart", svc

    return "do_nothing", VALID_SERVICES[0]

# ─────────────────────────────────────────────────────────────────
# Observation formatter
# ─────────────────────────────────────────────────────────────────

def format_observation(obs: dict, step: int) -> str:
    lines = [
        f"STEP {step}/{MAX_STEPS} | "
        f"Actions remaining: {obs.get('actions_remaining', '?')} | "
        f"System health: {obs.get('system_health', 0):.0%}",
        "",
        "SERVICE STATUS:",
    ]
    for svc in obs.get("services", []):
        sv = svc.get("status", -1)
        status_str = "HEALTHY" if sv >= 0.9 else "DEGRADED" if sv >= 0.4 else "DOWN" if sv >= 0 else "UNKNOWN"
        alert_str = " ALERT" if svc.get("alert") else ""
        if svc.get("observed"):
            ft = f" failure={svc['failure_type']}" if svc.get("failure_type") else ""
            detail = f" cpu={svc.get('cpu', 0):.0%} mem={svc.get('memory', 0):.0%}{ft}"
        else:
            detail = " [NOT OBSERVED]"
        lines.append(f"  {svc['name']:25s} {status_str:8s}{alert_str}{detail}")
    down = obs.get("down_services", [])
    if down:
        lines.append(f"\nDOWN: {', '.join(down)}")
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────
# LLM agent (with heuristic fallback)
# ─────────────────────────────────────────────────────────────────

_llm_failures = 0
_LLM_FAILURE_THRESHOLD = 3  # fall back to heuristic after 3 consecutive LLM failures


def llm_action(obs_text: str, obs: dict, history: list) -> tuple[str, str, str]:
    global _llm_failures

    # If LLM has been failing consistently, use heuristic
    if _llm_failures >= _LLM_FAILURE_THRESHOLD:
        action_type, target_service = heuristic_action(obs)
        return action_type, target_service, "heuristic fallback"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-3:]:
        messages.append({"role": "assistant", "content": json.dumps(h)})
    messages.append({"role": "user", "content": obs_text})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, temperature=0.0, max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        parsed = json.loads(match.group() if match else raw)
        action_type    = parsed.get("action_type", "observe")
        target_service = parsed.get("target_service", VALID_SERVICES[0])
        reasoning      = parsed.get("reasoning", "")
        if action_type not in VALID_ACTIONS:
            action_type = "observe"
        if target_service not in VALID_SERVICES:
            target_service = VALID_SERVICES[0]
        _llm_failures = 0  # reset on success
        return action_type, target_service, reasoning
    except Exception as e:
        _llm_failures += 1
        # Fall back to heuristic immediately
        action_type, target_service = heuristic_action(obs)
        return action_type, target_service, f"heuristic fallback (llm error: {e})"

# ─────────────────────────────────────────────────────────────────
# Run one task
# ─────────────────────────────────────────────────────────────────

def run_task(task_id: str, passing_score: float, seed: int = 42) -> dict:
    global _llm_failures
    _llm_failures = 0  # reset per task

    reset_resp = env_reset(task_id, seed=seed)
    obs        = reset_resp.get("observation", {})
    info       = reset_resp.get("info", {})

    print(json.dumps({
        "type":       "START",
        "task_id":    task_id,
        "scenario":   info.get("scenario", ""),
        "difficulty": info.get("difficulty", ""),
        "model":      MODEL_NAME,
        "seed":       seed,
    }))
    sys.stdout.flush()

    action_history = []
    step           = 0
    done           = False
    score          = 0.0
    last_step_info: dict = {}

    while not done and step < MAX_STEPS:
        obs_text = format_observation(obs, step)
        action_type, target_service, reasoning = llm_action(obs_text, obs, action_history)

        step_resp      = env_step(action_type, target_service)
        obs            = step_resp.get("observation", {})
        reward         = step_resp.get("reward", {}).get("total", 0.0)
        done           = step_resp.get("done", False)
        last_step_info = step_resp.get("info", {})
        step          += 1

        action_history.append({
            "action_type": action_type, "target_service": target_service,
            "reasoning": reasoning, "reward": reward,
        })

        print(json.dumps({
            "type":           "STEP",
            "task_id":        task_id,
            "step":           step,
            "action":         f"{action_type}({target_service})",
            "action_type":    action_type,
            "target_service": target_service,
            "reward":         round(float(reward), 4),
            "system_health":  round(float(obs.get("system_health", 0)), 4),
            "down_services":  obs.get("down_services", []),
            "done":           done,
            "reasoning":      str(reasoning),
        }))
        sys.stdout.flush()

    # Extract score from final step info
    if done and last_step_info:
        task_grade = last_step_info.get("task_grade", {})
        if isinstance(task_grade, dict) and "score" in task_grade:
            score = float(task_grade["score"])
        else:
            score = float(last_step_info.get("overall_score", 0.0))

    print(json.dumps({
        "type":            "END",
        "task_id":         task_id,
        "score":           round(score, 4),
        "passed":          score >= passing_score,
        "passing_score":   passing_score,
        "fully_recovered": float(obs.get("system_health", 0)) >= 1.0,
        "final_health":    round(float(obs.get("system_health", 0)), 4),
        "steps_taken":     step,
    }))
    sys.stdout.flush()

    return {"task_id": task_id, "score": score, "passed": score >= passing_score,
            "passing_score": passing_score}

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main() -> int:
    try:
        # Wait up to 120s for env container to be ready
        print(f"Waiting for environment at {ENV_URL} ...", file=sys.stderr)
        for _ in range(24):
            if env_health():
                print("Environment is ready.", file=sys.stderr)
                break
            time.sleep(5)
        else:
            print(f"ERROR: Environment not reachable at {ENV_URL} after 120s", file=sys.stderr)
            # Still continue — don't sys.exit, just report 0 scores
            _emit_zero_results()
            return 0

        results     = {}
        total_start = time.time()

        for task in TASKS:
            task_id       = task["task_id"]
            passing_score = task["passing_score"]
            try:
                grade = run_task(task_id, passing_score, seed=42)
                results[task_id] = grade
            except Exception as e:
                print(json.dumps({
                    "type": "END", "task_id": task_id,
                    "score": 0.0, "passed": False,
                    "passing_score": passing_score, "error": str(e),
                }))
                sys.stdout.flush()
                results[task_id] = {
                    "task_id": task_id, "score": 0.0, "passed": False,
                    "passing_score": passing_score,
                }

        total_elapsed = time.time() - total_start
        overall       = sum(r["score"] for r in results.values()) / max(1, len(results))
        all_passed    = all(r["passed"] for r in results.values())

        print(json.dumps({
            "type":               "SUMMARY",
            "model":              MODEL_NAME,
            "overall_score":      round(overall, 4),
            "all_passed":         all_passed,
            "total_time_seconds": round(total_elapsed, 1),
            "tasks": {
                tid: {"score": r["score"], "passed": r["passed"]}
                for tid, r in results.items()
            },
        }))
        sys.stdout.flush()

    except Exception as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        sys.stdout.flush()

    # Always exit 0 — scores determine pass/fail, not exit code
    return 0


def _emit_zero_results():
    for task in TASKS:
        print(json.dumps({
            "type": "END", "task_id": task["task_id"],
            "score": 0.0, "passed": False,
            "passing_score": task["passing_score"],
            "error": "environment not reachable",
        }))
    print(json.dumps({
        "type": "SUMMARY", "model": MODEL_NAME,
        "overall_score": 0.0, "all_passed": False,
        "total_time_seconds": 0.0,
        "tasks": {t["task_id"]: {"score": 0.0, "passed": False} for t in TASKS},
    }))
    sys.stdout.flush()


if __name__ == "__main__":
    sys.exit(main())
