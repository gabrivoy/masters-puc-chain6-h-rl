# Simple Chain6 Environment for Hierarchical vs Flat RL Comparison

This repository contains a compact, reproducible setup to compare tabular Q-learning (flat) vs a tabular hierarchical controller on the classic Chain6 environment. The goal is to show how temporal abstraction (sub-goals/options) helps discover a delayed reward path that is hard for flat exploration.

## Quickstart

```bash
# 1) Install deps (Python 3.13 recommended)
uv sync
# or: python -m pip install -e .

# 2) Run the experiment
uv run src/run.py

# 3) Results
# -> results/<YYYYMMDD>_<uuid8>/aggregated.csv
# -> results/<YYYYMMDD>_<uuid8>/plot.png
```

The script runs both agents across multiple random seeds in parallel (Python `multiprocessing`), aggregates a moving average over episodes, and plots mean ± uncertainty band ($\sigma$ or percentiles).

## What’s the Chain6 environment?

A chain of 6 states:

- States: `S1`, `S2`, `S3`, `S4`, `S5`, `S6` (agent starts at `S2`)
- Actions: `LEFT` (deterministic) and `RIGHT` (stochastic)
  - `LEFT`: moves one step to the left (stays in `S1` if already there)
  - `RIGHT`: with probability p moves right; otherwise moves left
- Terminal: reaching `S1` ends the episode

Reward at terminal (extrinsic):

- If the agent visited `S6` sometime during the episode: reward 1.0
- Otherwise: reward 0.01
- All intermediate steps: reward 0.0

This creates a sparse reward: the safe “short path” to `S1` yields 0.01 forever; the “long path” that goes to `S6` first is harder to find, but yields 1.0.

```mermaid
flowchart LR
  classDef terminal fill:#eef7ff,stroke:#2a63bf,stroke-width:2px;
  classDef normal fill:#ffffff,stroke:#999,stroke-width:1px;

  S1((S1)):::terminal
  S2((S2)):::normal
  S3((S3)):::normal
  S4((S4)):::normal
  S5((S5)):::normal
  S6((S6)):::normal

  S1 --- S2 --- S3 --- S4 --- S5 --- S6

  %% LEFT (determinístico)
  S2 -->|LEFT 1.0| S1
  S3 -->|LEFT 1.0| S2
  S4 -->|LEFT 1.0| S3
  S5 -->|LEFT 1.0| S4
  S6 -->|LEFT 1.0| S5
  S1 -->|LEFT 1.0| S1

  %% RIGHT (p=0.5 para direita; senão, esquerda)
  S1 -->|RIGHT 0.5| S1
  S1 -.->|RIGHT 0.5| S2

  S2 -->|RIGHT 0.5| S1
  S2 -.->|RIGHT 0.5| S3

  S3 -->|RIGHT 0.5| S2
  S3 -.->|RIGHT 0.5| S4

  S4 -->|RIGHT 0.5| S3
  S4 -.->|RIGHT 0.5| S5

  S5 -->|RIGHT 0.5| S4
  S5 -.->|RIGHT 0.5| S6

  S6 -->|RIGHT 0.5| S6
  S6 -.->|RIGHT 0.5| S5

  R["Terminal reward at S1:<br/>visited S6 → 1.0<br/>else → 0.01<br/>intermediate steps → 0.0"]
  R --- S1
```

## Agents

### Flat tabular Q-Learning

- Epsilon-greedy exploration with linear annealing.
- Standard TD target with $\gamma$.
- Learns a single state-action value table $Q(s, a)$.

### Hierarchical tabular controller (h-Q-learning style)

- Meta-controller (`Q2`) picks a sub-goal $g \epsilon {3, 4, 5, 6}$.
- Controller (`Q1`) acts to reach the chosen $g$ using an intrinsic reward of 1 when hitting $g$, 0 otherwise.
- When the option terminates (hit $g$ or episode ends), `Q2` updates on the accumulated extrinsic reward and the discounted continuation.
- Separate epsilon schedules for the meta-policy and each controller policy.

Result: the option structure helps the agent purposefully travel rightward to visit `S6` and unlock the large terminal reward.
