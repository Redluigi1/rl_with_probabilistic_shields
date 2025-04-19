# 🛡️ Safe Reinforcement Learning with Probabilistic Shields in GridWorld

This project implements **Safe Reinforcement Learning (Safe RL)** using **probabilistic shields**, inspired by the paper *"Safe Reinforcement Learning via Probabilistic Shields"* by Jansen et al. It demonstrates how a Q-learning agent can learn in a grid environment while avoiding unsafe actions in the presence of adversaries.

---

## 🧠 Overview

This repo contains:
- A custom `SimpleGridWorld` environment.
- A `ModelChecker` for simulating forward collision probabilities.
- A `ProbabilisticShield` to block risky actions.
- A `QAgent` implementing tabular Q-learning.
- Tools for running experiments and visualizing results.

---

## 🏗️ Architecture

### Environment: `SimpleGridWorld`
- Grid of customizable size (`width x height`).
- The avatar starts at `(0, 0)`; goal is at `(width-1, height-1)`.
- Randomly moving adversaries (chasing the avatar with bias).
- Token rewards at every cell (1-time bonus).
- Rewards:
  - **Step penalty:** `-0.1`
  - **Token reward:** `+1.0`
  - **Goal reward:** `+10.0`
  - **Collision penalty:** `-10.0`

### Safety Layer

#### `ModelChecker`
Simulates the environment forward up to a fixed **horizon** to compute the probability of collision for each action.

#### `ProbabilisticShield`
Prevents actions that increase collision risk disproportionately:
- Controlled via a **delta threshold** (e.g. `delta = 0.7`).
- If no safe actions meet the threshold, fallback to the safest.

### RL Agent: `QAgent`
- Uses tabular Q-learning.
- Supports ε-greedy exploration.
- Learns action-value estimates only over **shielded** actions when enabled.

---

## 🧪 Run the Experiments

```bash
python main.py
