"""SelfHealRL — Baseline inference script (OpenEnv Hackathon).

Runs an LLM agent against all 3 tasks using the OpenAI API client.
Reads credentials from environment variables:
  API_BASE_URL  — LLM API endpoint (e.g. https://api.openai.com/v1)
  MODEL_NAME    — Model identifier (e.g. gpt-4o-mini)
  HF_TOKEN      — HuggingFace / API key

Usage:
  export API_BASE_URL=https://api.openai.com/v1
  export MODEL_NAME=gpt-4o-mini
  export HF_TOKEN=sk-...
  python inference.py

Output:
  Prints per-task scores and overall baseline score to stdout.
  Runtime: < 20 minutes on 2 vCPU / 8 GB.
"""

from __future__ import annotations

import json
import os
import sys
import time
import re
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI

from config import ACTION_TYPES, MAX_STEPS_PER_EPISODE, SERVICE_NAMES
from core.tasks import TASKS, TaskGrader
from env.failure_engine import FailureEngine
from env.selfheal_env import SelfHealEnv

# ─────────────────────────────────────────────────────────────────
# Config from env vars
# ─────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))

if not HF_TOKEN:
    print("ERROR: HF_TOKEN (or OPENAI_API_KEY) environment variable not set.")
    sys.exit(1)

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ─────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an autonomous SRE (Site Reliability Engineer) agent.
Your job is to diagnose and fix cascading failures in a microservice system.

SERVICE DEPENDENCY GRAPH:
  api-gateway → auth-service → user-db, cache-layer
  payment-service → auth-service, order-db
  order-service → order-db, cache-layer
  search-service → restaurant-db, cache-layer
  notification-service → user-db

RULES:
1. Fix upstream services (dependencies) BEFORE downstream services.
2. OBSERVE a service before restarting/fixing it — you need to know the failure type.
3. Choose the best action for the failure type:
   - memory_leak     → scale_up (70%) or restart (60%)
   - cpu_spike       → scale_up (85%) or restart (75%)
   - bad_deployment  → rollback (95%)
   - network_partition → reroute (80%)
   - disk_full       → restart (20%) — low success, try reroute
   - connection_timeout → reroute (70%) or restart (80%)
4. Do NOT repeat the same action on the same healthy service.
5. Do NOT do_nothing when services are down.

AVAILABLE ACTIONS:
  restart, scale_up, reroute, rollback, observe, do_nothing

AVAILABLE SERVICES:
  api-gateway, auth-service, payment-service, order-service,
  search-service, notification-service, user-db, cache-layer,
  restaurant-db, order-db

Respond with ONLY valid JSON:
{"action_type": "<action>", "target_service": "<service>", "reasoning": "<1 sentence>"}
"""


# ─────────────────────────────────────────────────────────────────
# Observation formatter
# ─────────────────────────────────────────────────────────────────

def format_observation(env: SelfHealEnv, step: int) -> str:
    """Format env state as a text prompt for the LLM."""
    statuses = env.mesh.get_all_statuses()
    alerts = env.obs_encoder.get_alerts(env.mesh)
    observed = env.obs_encoder.observed_services

    lines = [
        f"STEP {step}/{MAX_STEPS_PER_EPISODE} | "
        f"Actions remaining: {env.actions_remaining} | "
        f"System health: {env.mesh.system_health():.0%}",
        "",
        "SERVICE STATUS:",
    ]

    for name in SERVICE_NAMES:
        d = statuses[name]
        status_str = {1.0: "HEALTHY", 0.5: "DEGRADED", 0.0: "DOWN"}.get(
            round(d["status"], 1), f"status={d['status']:.2f}"
        )
        alert_str = " ⚠️ALERT" if name in alerts else ""
        obs_str = ""
        if name in observed or not env.partial_observability:
            svc = env.mesh.services[name]
            ft = f" failure_type={svc.failure_type}" if svc.failure_type else ""
            obs_str = (
                f" cpu={d.get('cpu', 0):.0%} mem={d.get('memory', 0):.0%}"
                f" latency={d.get('latency', 0):.0f}ms err={d.get('error_rate', 0):.0%}"
                f"{ft}"
            )
        else:
            obs_str = " [NOT OBSERVED — use observe action to see metrics]"

        lines.append(f"  {name:25s} {status_str:8s}{alert_str}{obs_str}")

    down = env.mesh.get_down_services()
    degraded = env.mesh.get_degraded_services()
    if down:
        lines.append(f"\nDOWN SERVICES: {', '.join(down)}")
    if degraded:
        lines.append(f"DEGRADED SERVICES: {', '.join(degraded)}")
    if not down and not degraded:
        lines.append("\n✅ All services healthy!")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# LLM agent
# ─────────────────────────────────────────────────────────────────

def llm_action(observation_text: str, history: list) -> tuple[str, str, str]:
    """Call the LLM and return (action_type, target_service, reasoning)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    # Include last 3 steps of history for context
    for h in history[-3:]:
        messages.append({"role": "assistant", "content": json.dumps(h)})

    messages.append({"role": "user", "content": observation_text})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=150,
    )

    raw = response.choices[0].message.content.strip()

    # Parse JSON from response
    try:
        # Extract JSON even if wrapped in markdown
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
        else:
            parsed = json.loads(raw)

        action_type = parsed.get("action_type", "observe")
        target_service = parsed.get("target_service", SERVICE_NAMES[0])
        reasoning = parsed.get("reasoning", "")

        # Validate
        if action_type not in ACTION_TYPES:
            action_type = "observe"
        if target_service not in SERVICE_NAMES:
            target_service = SERVICE_NAMES[0]

        return action_type, target_service, reasoning

    except (json.JSONDecodeError, KeyError):
        # Fallback: parse with regex
        at_match = re.search(r'"action_type"\s*:\s*"(\w+)"', raw)
        ts_match = re.search(r'"target_service"\s*:\s*"([\w-]+)"', raw)
        action_type = at_match.group(1) if at_match else "observe"
        target_service = ts_match.group(1) if ts_match else SERVICE_NAMES[0]
        if action_type not in ACTION_TYPES:
            action_type = "observe"
        if target_service not in SERVICE_NAMES:
            target_service = SERVICE_NAMES[0]
        return action_type, target_service, raw[:80]


# ─────────────────────────────────────────────────────────────────
# Run one task
# ─────────────────────────────────────────────────────────────────

def run_task(task_id: str, seed: int = 42, verbose: bool = True) -> dict:
    """Run a single task with the LLM agent. Returns grade dict."""
    task = TASKS[task_id]
    engine = FailureEngine()

    env = SelfHealEnv(
        difficulty=task.difficulty,
        partial_observability=task.partial_observability,
    )
    env.reset(seed=seed)

    # Apply fixed scenario
    scenario = engine.generate_from_template(task.scenario_template)
    env.mesh.reset()
    engine.apply_scenario(env.mesh, scenario)
    env.cascade_sim.reset()
    for svc, _ in scenario.root_failures:
        env.cascade_sim.record_root_cause(svc)
    env.scenario = scenario
    env._prev_down = set(env.mesh.get_down_services())
    env._prev_degraded = set(env.mesh.get_degraded_services())

    if verbose:
        print(f"\n  Scenario: {scenario.description}")
        print(f"  Root causes: {[s for s, _ in scenario.root_failures]}")

    action_history = []
    done = False
    step = 0

    while not done and step < task.max_steps:
        obs_text = format_observation(env, step)

        try:
            action_type, target_service, reasoning = llm_action(obs_text, action_history)
        except Exception as e:
            if verbose:
                print(f"  ⚠️  LLM error at step {step}: {e} — using observe fallback")
            action_type, target_service, reasoning = "observe", SERVICE_NAMES[0], "fallback"

        action_int = (
            ACTION_TYPES.index(action_type) * len(SERVICE_NAMES)
            + SERVICE_NAMES.index(target_service)
        )

        _, reward, terminated, truncated, _ = env.step(action_int)
        done = terminated or truncated
        step += 1

        action_history.append({
            "action_type": action_type,
            "target_service": target_service,
            "reasoning": reasoning,
            "reward": reward,
        })

        if verbose:
            health = env.mesh.system_health()
            down = env.mesh.get_down_services()
            print(
                f"  Step {step:2d}: {action_type:8s}({target_service:22s}) "
                f"r={reward:+6.1f}  health={health:.0%}  down={down}"
            )

    summary = env.get_episode_summary()
    grade = TaskGrader.grade(task_id, summary)
    return grade


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  SelfHealRL — Baseline Inference")
    print(f"  Model:   {MODEL_NAME}")
    print(f"  API:     {API_BASE_URL}")
    print("=" * 65)

    results = {}
    total_start = time.time()

    for task_id in ["task_easy", "task_medium", "task_hard"]:
        task = TASKS[task_id]
        print(f"\n{'─' * 65}")
        print(f"Task: {task_id} — {task.name} [{task.difficulty}]")
        print(f"{'─' * 65}")

        start = time.time()
        try:
            grade = run_task(task_id, seed=42, verbose=True)
            elapsed = time.time() - start
            results[task_id] = grade
            status = "✅ PASSED" if grade["passed"] else "❌ FAILED"
            print(f"\n  {status} | score={grade['score']:.4f} | "
                  f"pass>={grade['passing_score']} | {elapsed:.1f}s")
            print(f"  Breakdown: {grade['breakdown']}")
        except Exception as e:
            elapsed = time.time() - start
            print(f"\n  ❌ ERROR: {e} ({elapsed:.1f}s)")
            results[task_id] = {
                "task_id": task_id, "score": 0.0, "passed": False,
                "passing_score": task.passing_score,
                "breakdown": {}, "details": {"error": str(e)},
            }

    # Summary
    total_elapsed = time.time() - total_start
    overall = sum(r["score"] for r in results.values()) / len(results)
    all_passed = all(r["passed"] for r in results.values())

    print(f"\n{'=' * 65}")
    print(f"  BASELINE RESULTS  (total time: {total_elapsed:.1f}s)")
    print(f"{'=' * 65}")
    print(f"  {'Task':<15} {'Score':>8}  {'Pass':>6}  {'Status'}")
    print(f"  {'-'*50}")
    for task_id, grade in results.items():
        status = "✅" if grade["passed"] else "❌"
        print(
            f"  {task_id:<15} {grade['score']:>8.4f}  "
            f"{grade['passing_score']:>6.1f}  {status}"
        )
    print(f"  {'-'*50}")
    print(f"  {'OVERALL':<15} {overall:>8.4f}  {'':>6}  "
          f"{'✅ ALL PASSED' if all_passed else '⚠️  SOME FAILED'}")
    print(f"{'=' * 65}")

    # Machine-readable output
    output = {
        "model": MODEL_NAME,
        "tasks": results,
        "overall_score": round(overall, 4),
        "all_passed": all_passed,
        "total_time_seconds": round(total_elapsed, 1),
    }
    print("\nJSON_OUTPUT:", json.dumps(output))

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
