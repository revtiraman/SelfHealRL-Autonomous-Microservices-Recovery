# SelfHealRL — Autonomous Microservices Recovery Agent

An RL agent that learns to **diagnose and fix cascading failures** in a 10-service microservice mesh. Built with Gymnasium, Stable-Baselines3, and Gradio.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-green)
![SB3](https://img.shields.io/badge/Stable--Baselines3-2.2-orange)

## The Problem

When a microservice fails in production, the failure **cascades** through dependent services — a database crash can take down auth, which kills the API gateway, which drops the entire system. SREs must quickly:

1. **Diagnose** — which service is the root cause?
2. **Prioritize** — fix upstream dependencies first
3. **Choose** — restart? rollback? scale up? reroute?
4. **Act fast** — every step costs time and money

SelfHealRL trains an RL agent to do this autonomously.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Service Mesh (10 services)            │
│                                                         │
│  Gateway → Auth → User-DB     Search → Restaurant-DB   │
│              ↓       ↑          ↓                       │
│          Payment → Order-DB   Cache-Layer               │
│              ↓                  ↑                       │
│          Order ─────────────────┘                       │
│              ↓                                          │
│         Notification → User-DB                          │
└─────────────────────────────────────────────────────────┘
         ↕                              ↕
   ┌──────────┐                  ┌─────────────┐
   │ RL Agent │  ← 74-dim obs → │  Reward Fn   │
   │  (PPO)   │                  │ (10 signals) │
   └──────────┘                  └─────────────┘
```

## Key Features

### Environment
- **10-service mesh** modeled after a food delivery platform (gateway, auth, payment, order, search, notification, 4 databases)
- **6 failure types**: memory leak, CPU spike, connection timeout, disk full, bad deployment, network partition
- **Cascade simulation**: failures propagate through the dependency graph
- **Stochastic recovery**: action success depends on failure type (e.g., rollback is 95% effective for bad deployments, but only 30% for network partitions)
- **Partial observability**: agent must `observe` a service before seeing its metrics

### Agent Actions (6 types x 10 services = 60 discrete actions)
| Action | Best For |
|--------|----------|
| `restart` | CPU spikes, connection timeouts |
| `scale_up` | CPU spikes, memory leaks |
| `reroute` | Network partitions |
| `rollback` | Bad deployments (95% success) |
| `observe` | Diagnose before acting |
| `do_nothing` | Wait for recovery |

### Evaluation (3 systems)
1. **6 Programmatic Graders**: recovery, MTTR, cascade prevention, dependency ordering, efficiency, diagnosis accuracy
2. **Multi-objective Reward** (10 components): service recovery (+10), root cause fix (+15), cascade caused (-5), wrong order (-5), etc.
3. **LLM/Heuristic Scorer**: evaluates decision quality on root cause targeting, dependency awareness, timing, and overall strategy

### Training
- **4-phase curriculum learning**: EASY (50k steps) → MEDIUM (100k) → HARD (200k) → CHAOS (300k)
- Progressive difficulty: more simultaneous failures, faster cascade, partial observability
- PPO with tuned hyperparameters per phase

## Results

Evaluated over 20 episodes per cell:

| Difficulty | Heuristic Agent | Random Agent |
|:----------:|:---------------:|:------------:|
| EASY       | **100%** success, grade 0.97 | 45% success, grade 0.60 |
| MEDIUM     | **90%** success, grade 0.87  | 5% success, grade 0.41  |
| HARD       | **65%** success, grade 0.82  | 0% success, grade 0.41  |
| CHAOS      | **75%** success, grade 0.86  | 0% success, grade 0.55  |

The heuristic agent (observe → fix upstream first → best action for failure type) provides a strong baseline. The PPO agent learns similar strategies through curriculum training.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Verify environment
PYTHONPATH=. python run.py test

# Train (single phase)
PYTHONPATH=. python run.py train --phase phase1_easy

# Train (full curriculum ~15 min)
PYTHONPATH=. python run.py train --curriculum

# Evaluate a model
PYTHONPATH=. python run.py eval --model models/phase1_easy.zip

# Launch Gradio demo
PYTHONPATH=. python run.py demo
```

## Gradio Demo (4 tabs)

| Tab | What it does |
|-----|-------------|
| **Live Demo** | Run any agent on any difficulty, watch the animated service mesh |
| **Agent vs Random** | Side-by-side comparison on the same failure scenario |
| **Grading Report** | Batch evaluation across N episodes with 6 grader breakdown |
| **LLM Analysis** | Per-decision scoring with root cause, dependency, timing analysis |

## Project Structure

```
selfheal-rl/
├── config.py                  # Hyperparameters, service definitions, rewards
├── run.py                     # CLI entry point
├── requirements.txt
├── env/
│   ├── selfheal_env.py        # Gymnasium environment (SB3 compatible)
│   ├── service_mesh.py        # 10 services + dependency graph
│   ├── failure_engine.py      # Random + 7 scenario templates
│   ├── cascade_simulator.py   # Cascade propagation + root cause tracking
│   └── observations.py        # 74-dim observation encoder
├── core/
│   ├── reward.py              # Multi-objective reward (10 components)
│   ├── graders.py             # 6 programmatic graders
│   ├── metrics.py             # Episode metrics + JSON export
│   ├── llm_scorer.py          # LLM/heuristic decision scoring
│   └── heuristic_agent.py     # Rule-based baseline agent
├── training/
│   ├── train.py               # PPO + 4-phase curriculum
│   ├── callbacks.py           # Metrics + curriculum callbacks
│   └── evaluate.py            # Evaluation + agent comparison
└── ui/
    ├── app.py                 # Gradio 4-tab dashboard
    ├── visualizer.py          # Animated HTML service mesh
    └── replay.py              # Frame-by-frame episode replay
```

## Advanced Features

- **Partial Observability**: Under HARD/CHAOS, the agent sees `-1` for unobserved service metrics and must spend actions on `observe` before it can diagnose
- **Dynamic Failure Progression**: Failures start as warnings, progress to degraded, then crash — early intervention earns bonus rewards
- **Stochastic Recovery**: Each (action, failure_type) pair has a different success probability, forcing the agent to learn which actions work for which failures
- **Cascade Prevention**: Agent is rewarded for preventing cascades and penalized for causing new ones through wrong-order actions

## Tech Stack

- **Gymnasium** — RL environment interface
- **Stable-Baselines3** — PPO implementation
- **Gradio** — Interactive web demo
- **NumPy** — Observation encoding and metrics

## License

MIT
