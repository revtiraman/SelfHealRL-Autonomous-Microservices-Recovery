"""SelfHealRL — Gradio Web Interface."""

from __future__ import annotations

import os
import sys

import gradio as gr
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ACTION_TYPES, MAX_STEPS_PER_EPISODE, SERVICE_NAMES
from core.graders import Grader
from core.heuristic_agent import HeuristicAgent
from core.llm_scorer import LLMScorer
from env.selfheal_env import SelfHealEnv
from ui.visualizer import render_action_log, render_mesh

_scorer = LLMScorer(mode="heuristic")


def _load_ppo_model():
    try:
        from sb3_contrib import RecurrentPPO
        for path in ["models/selfheal_agent_final.zip", "models/phase4_chaos.zip", "models/phase3_hard_partial.zip"]:
            if os.path.exists(path):
                return RecurrentPPO.load(path)
    except Exception:
        pass
    return None


def _run_episode(difficulty: str, agent_type: str, seed: int | None = None):
    """Run a full episode with the chosen agent. Returns (env, history)."""
    # PPO uses partial obs only for HARD/CHAOS (matches training phases)
    if agent_type == "Trained PPO Agent":
        partial = difficulty in ("HARD", "CHAOS")
    else:
        partial = False
    env = SelfHealEnv(difficulty=difficulty, partial_observability=partial)
    obs, _ = env.reset(seed=seed)

    heuristic = HeuristicAgent() if agent_type == "Heuristic Agent" else None
    if heuristic:
        heuristic.reset()

    ppo_model = _load_ppo_model() if agent_type == "Trained PPO Agent" else None
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)

    for _ in range(MAX_STEPS_PER_EPISODE):
        if heuristic is not None:
            statuses = env.mesh.get_all_statuses()
            act_type, target = heuristic.act(statuses)
            if act_type == "observe":
                svc_data = env.mesh.services.get(target)
                if svc_data:
                    heuristic.record_observation(target, svc_data.failure_type or "unknown")
            action = heuristic.action_to_int(act_type, target)
        elif ppo_model is not None:
            action, lstm_states = ppo_model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
            action = int(action)
        else:
            action = env.action_space.sample()

        obs, _, term, trunc, _ = env.step(action)
        episode_start = np.array([term or trunc])
        if term or trunc:
            break

    return env


# ═══════════════════════════════════════════════════════════════
# TAB 1: Live Demo
# ═══════════════════════════════════════════════════════════════

def run_agent_demo(difficulty, agent_type):
    env = _run_episode(difficulty, agent_type)
    history = env.episode_history
    statuses = env.mesh.get_all_statuses()

    last = history[-1] if history else None
    action_info = None
    if last:
        action_info = {
            "target": last.target_service,
            "action_type": last.action_type,
            "success": last.action_success,
        }

    mesh_html = render_mesh(
        statuses,
        action_info=action_info,
        step=len(history),
        total_reward=env.total_reward,
        actions_remaining=env.actions_remaining,
    )
    log_html = render_action_log(history)
    summary = env.get_episode_summary()
    grades = Grader.grade_all(summary)

    recovered = env.mesh.is_fully_recovered()
    status_msg = (
        f"{'✅ RECOVERED' if recovered else '❌ NOT RECOVERED'} "
        f"in {len(history)} steps | Reward: {env.total_reward:.1f} | Grade: {grades['overall_score']:.2f}"
    )
    return mesh_html, status_msg, log_html


# ═══════════════════════════════════════════════════════════════
# TAB 2: Agent vs Random
# ═══════════════════════════════════════════════════════════════

def run_comparison(difficulty):
    seed = int(np.random.randint(0, 9999))

    env_h = _run_episode(difficulty, "Heuristic Agent", seed=seed)
    env_r = _run_episode(difficulty, "Random Agent", seed=seed)

    h_html = render_mesh(env_h.mesh.get_all_statuses(), step=len(env_h.episode_history), total_reward=env_h.total_reward)
    r_html = render_mesh(env_r.mesh.get_all_statuses(), step=len(env_r.episode_history), total_reward=env_r.total_reward)

    h_grades = Grader.grade_all(env_h.get_episode_summary())
    r_grades = Grader.grade_all(env_r.get_episode_summary())

    comparison = f"""
| Metric | Heuristic Agent | Random Agent |
|--------|:---------:|:-------:|
| Steps | {len(env_h.episode_history)} | {len(env_r.episode_history)} |
| Reward | {env_h.total_reward:.1f} | {env_r.total_reward:.1f} |
| Recovered | {'✅' if env_h.mesh.is_fully_recovered() else '❌'} | {'✅' if env_r.mesh.is_fully_recovered() else '❌'} |
| Health | {env_h.mesh.system_health():.0%} | {env_r.mesh.system_health():.0%} |
| Grade | {h_grades['overall_score']:.2f} | {r_grades['overall_score']:.2f} |
"""
    return h_html, r_html, comparison


# ═══════════════════════════════════════════════════════════════
# TAB 3: Grading Report
# ═══════════════════════════════════════════════════════════════

def run_grading(difficulty, num_episodes, agent_type):
    num_episodes = int(num_episodes)
    all_grades = []

    for i in range(num_episodes):
        env = _run_episode(difficulty, agent_type, seed=i)
        grades = Grader.grade_all(env.get_episode_summary())
        all_grades.append(grades)

    grader_names = ["recovery", "mttr", "cascade_prevention", "dependency_ordering", "efficiency", "diagnosis"]
    report = f"## Grading Report — {agent_type}\n\n"
    report += f"**{num_episodes} episodes | Difficulty: {difficulty}**\n\n"
    report += "| Grader | Avg Score | Pass Rate |\n|--------|:---------:|:---------:|\n"

    for name in grader_names:
        avg = np.mean([g[name]["score"] for g in all_grades])
        pass_rate = np.mean([g[name]["passed"] for g in all_grades])
        icon = "✅" if pass_rate >= 0.8 else "⚠️" if pass_rate >= 0.5 else "❌"
        report += f"| {icon} {name} | {avg:.2f} | {pass_rate:.0%} |\n"

    report += f"\n**Overall Score: {np.mean([g['overall_score'] for g in all_grades]):.2f}** | "
    report += f"**Overall Pass Rate: {np.mean([g['overall_pass'] for g in all_grades]):.0%}**"
    return report


# ═══════════════════════════════════════════════════════════════
# TAB 4: LLM Analysis
# ═══════════════════════════════════════════════════════════════

def run_llm_analysis(difficulty, agent_type):
    env = _run_episode(difficulty, agent_type, seed=42)
    summary = env.get_episode_summary()
    episode_scores = _scorer.score_episode(summary)
    strategy = _scorer.score_strategy(summary)

    report = f"## LLM Decision Analysis — {agent_type}\n\n"
    report += f"**Scenario:** {summary['scenario']}  \n"
    report += f"**Result:** {'✅ Recovered' if env.mesh.is_fully_recovered() else '❌ Not recovered'} | Reward: {env.total_reward:.1f}\n\n"

    report += "### Critical Decisions\n\n"
    for dec in episode_scores["critical_decisions"]:
        report += f"**Step {dec['step']}: `{dec['action']}`**\n"
        report += f"- Root cause targeting: {dec['root_cause_score']}/10\n"
        report += f"- Dependency awareness: {dec['dependency_score']}/10\n"
        report += f"- Action selection: {dec['action_type_score']}/10\n"
        report += f"- Timing: {dec['timing_score']}/10\n"
        report += f"- **Overall: {dec['overall_score']}/10**\n"
        if "reasoning" in dec:
            report += f"\n> {dec['reasoning']}\n"
        report += "\n"

    report += "### Overall Strategy\n\n"
    report += "| Criterion | Score |\n|-----------|:-----:|\n"
    report += f"| Diagnostic Approach | {strategy['diagnostic_approach']}/10 |\n"
    report += f"| Prioritization | {strategy['prioritization']}/10 |\n"
    report += f"| Patience | {strategy['patience']}/10 |\n"
    report += f"| **Overall Strategy** | **{strategy['overall_strategy']}/10** |\n"
    return report


# ═══════════════════════════════════════════════════════════════
# Build App
# ═══════════════════════════════════════════════════════════════

AGENT_CHOICES = ["Heuristic Agent", "Random Agent", "Trained PPO Agent"]

def build_app() -> gr.Blocks:
    with gr.Blocks(title="SelfHealRL — Autonomous Microservices Recovery") as demo:
        gr.Markdown(
            "# 🏥 SelfHealRL — Autonomous Microservices Recovery\n"
            "> RL agent that diagnoses and fixes cascading failures in microservices"
        )

        with gr.Tabs():
            # Tab 1: Live Demo
            with gr.Tab("🔴 Live Demo"):
                with gr.Row():
                    diff1 = gr.Dropdown(["EASY", "MEDIUM", "HARD", "CHAOS"], value="MEDIUM", label="Difficulty")
                    agent1 = gr.Dropdown(AGENT_CHOICES, value="Heuristic Agent", label="Agent Type")
                    run_btn = gr.Button("▶️ Run Demo", variant="primary")

                mesh_out = gr.HTML()
                status_out = gr.Markdown("Click **Run Demo** to start")
                log_out = gr.HTML()

                run_btn.click(run_agent_demo, [diff1, agent1], [mesh_out, status_out, log_out])

            # Tab 2: Heuristic vs Random
            with gr.Tab("⚔️ Heuristic vs Random"):
                with gr.Row():
                    diff2 = gr.Dropdown(["EASY", "MEDIUM", "HARD", "CHAOS"], value="MEDIUM", label="Difficulty")
                    cmp_btn = gr.Button("⚔️ Run Comparison", variant="primary")

                with gr.Row():
                    h_mesh = gr.HTML(label="Heuristic Agent")
                    r_mesh = gr.HTML(label="Random Agent")
                cmp_md = gr.Markdown()
                cmp_btn.click(run_comparison, [diff2], [h_mesh, r_mesh, cmp_md])

            # Tab 3: Grading Report
            with gr.Tab("📋 Grading Report"):
                with gr.Row():
                    diff3 = gr.Dropdown(["EASY", "MEDIUM", "HARD", "CHAOS"], value="MEDIUM", label="Difficulty")
                    agent3 = gr.Dropdown(AGENT_CHOICES, value="Heuristic Agent", label="Agent Type")
                    n_eps = gr.Slider(5, 50, value=10, step=5, label="Episodes")
                    grade_btn = gr.Button("📊 Evaluate", variant="primary")

                grade_out = gr.Markdown()
                grade_btn.click(run_grading, [diff3, n_eps, agent3], [grade_out])

            # Tab 4: LLM Analysis
            with gr.Tab("🧠 LLM Analysis"):
                with gr.Row():
                    diff4 = gr.Dropdown(["EASY", "MEDIUM", "HARD", "CHAOS"], value="MEDIUM", label="Difficulty")
                    agent4 = gr.Dropdown(AGENT_CHOICES, value="Heuristic Agent", label="Agent Type")
                    llm_btn = gr.Button("🧠 Analyze", variant="primary")

                llm_out = gr.Markdown()
                llm_btn.click(run_llm_analysis, [diff4, agent4], [llm_out])

    return demo


if __name__ == "__main__":
    build_app().launch(share=False, theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"))
