---
title: Adaptive Fraud Audit Arena
emoji: 🕵️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - llm-training
---

# 🕵️ Adaptive Fraud Audit Arena (AFAA)

**An RL environment for training language models to reason under conflicting information, limited resources, and adversarial signals.**

🔗 Hugging Face Space: https://huggingface.co/spaces/sharad0x/openenv-afaa-gym
📝 Blog / Deep Dive: [Deep Dive Writeup](https://github.com/sharad0x/Sovereign-SRE-Gym/blob/main/blog.md)

---

# 📌 Motivation

Most RL environments evaluate whether an agent can reach a correct answer.

AFAA focuses on a different question:

> **How does an agent behave when the signals themselves are unreliable, incomplete, or misleading?**

The goal is to move beyond correctness toward **decision-making under uncertainty**.

---

# 🧭 Environment Overview

The agent acts as an auditor investigating departments connected through a hidden fraud graph.

Available actions include:

* interviewing a CFO
* interacting with a whistleblower
* querying a database
* applying negotiation strategies
* submitting a final audit

The episode ends when the agent submits a decision or runs out of budget.

---

# ⚙️ Core Mechanics

## 1. Multi-Source Signals

* CFO and Whistleblower provide potentially conflicting information
* signals evolve over time

## 2. Imperfect Tools

* database outputs may be noisy or misleading
* the agent must interpret—not blindly trust

## 3. Dynamic Environment

* underlying fraud structure may change
* previously valid reasoning can become outdated

## 4. Belief-Based State

The agent maintains:

* `global_beliefs`
* entropy (uncertainty)
* conflict score

This enables tracking reasoning evolution over time.

---

# 🧪 Reward Design

The environment uses a rubric-based reward system:

* Correctness
* Progress
* Efficiency
* Consistency
* Anti-Hacking
* Exploration

The reward reflects both:

> outcome and reasoning behavior

---

# ⚠️ Understanding Training Signals

Training in AFAA produces signals that differ from standard RL environments.

## Negative Rewards

Early-stage agents receive strong penalties due to:

* inconsistent reasoning
* inefficient exploration

## High Variance

Rewards fluctuate due to:

* stochastic signals
* conflicting sources
* dynamic structure

## Delayed Improvement

Agents must first stabilize reasoning before improving outcomes.

---

# 📈 Training Signal (Preview)

Below is a representative reward curve from early training:

![Reward Curve](./assets/reward_curve.png)

*Reward trend across training steps. Improvement is gradual due to strict reasoning constraints.*

---

# 🧠 Why This Environment Matters

AFAA is designed for scenarios involving:

* unreliable information
* sequential decision-making
* evolving systems

Examples:

* financial audits
* investigations
* decision support systems

---

# 🔗 Links

* Hugging Face Space: https://huggingface.co/spaces/sharad0x/openenv-afaa-gym
* Blog: https://github.com/sharad0x/Sovereign-SRE-Gym/blob/main/blog.md
* Repository: https://github.com/sharad0x/Sovereign-SRE-Gym

---

# 🧠 Note

This environment explores how RL systems behave when **reasoning quality—not just final correctness—is treated as a core objective.**
