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
📝 Blog / Deep Dive: *[ADD LINK]*

---

# 📌 Motivation

Most RL environments evaluate whether an agent can reach a correct answer.

AFAA focuses on a different question:

> **How does an agent behave when the signals themselves are unreliable, incomplete, or misleading?**

This includes situations where:

* multiple sources disagree
* tools provide noisy or corrupted outputs
* the environment changes during interaction

The goal is to move beyond correctness toward **decision-making under uncertainty**.

---

# 🧭 Environment Overview

The agent plays the role of an auditor investigating departments connected through a hidden fraud graph.

At each step, it can:

* interview a CFO
* interact with a whistleblower
* query a database
* apply negotiation strategies
* submit a final audit

The episode ends when the agent submits a decision or runs out of budget.

---

# ⚙️ Core Mechanics

## 1. Multi-Source Signals

Two primary entities provide information:

* **CFO** → may cooperate or strategically mislead
* **Whistleblower** → may be accurate or noisy

Claims are tracked and can conflict over time.

---

## 2. Imperfect Tools

The database provides structured signals, but:

* results may be noisy or misleading
* anomalies are exposed, not hidden

The agent must interpret—not blindly trust—the outputs.

---

## 3. Dynamic Environment

The fraud graph is not always fixed:

* connections may change
* previously valid reasoning paths may become outdated

Mutation events are controlled and observable.

---

## 4. Belief-Based State

The agent maintains a belief distribution over departments.

State includes:

* `global_beliefs`
* entropy (uncertainty)
* conflict score

This allows tracking **reasoning evolution**, not just outcomes.

---

# 🧠 What Is Evaluated

## 1. Decision Accuracy

Can the agent identify the correct root cause?

## 2. Reasoning Stability

Does the agent maintain consistent beliefs over time?

## 3. Robustness to Deception

Can the agent handle misleading or conflicting signals?

---

# 🧪 Reward Design

AFAA uses a rubric-based reward system:

* **Correctness**
* **Progress**
* **Efficiency**
* **Consistency**
* **Anti-Hacking**
* **Exploration**

The reward reflects both:

> what the agent does and how it reasons

---

# ⚠️ Understanding Training Signals (Preliminary)

Training in AFAA produces signals that may appear unusual compared to standard RL environments.

---

## Negative Rewards

Early-stage agents often produce strongly negative rewards due to:

* inconsistent reasoning
* inefficient exploration
* over-reliance on single signals

➡️ The reward function is intentionally strict.

---

## High Variance

Rewards fluctuate because:

* signals are stochastic
* sources may conflict
* environment structure may change

➡️ Variance reflects reasoning difficulty, not instability.

---

## Delayed Improvement

Agents must first learn to:

* reduce contradictions
* stabilize beliefs
* avoid misleading signals

➡️ Improvements appear gradually, not immediately.

---

## Structured Output Constraints

LLM agents may produce:

* parsing errors
* invalid actions

➡️ This reflects real-world integration constraints.

---

# 📈 Training Results (To Be Added)

## 📊 Sample Training Signals (Preview)

![Reward Curve](./assets/reward.png)

![Entropy Reduction](./assets/entropy.png)

![Baseline vs Trained](./assets/comparison.png)

---

# 🧠 Why This Environment Matters

AFAA is designed for scenarios where:

* information is unreliable
* decisions are sequential
* systems change over time

Examples:

* financial audits
* incident investigations
* intelligence analysis

---

# 🔗 Links

* Hugging Face Space: https://huggingface.co/spaces/sharad0x/openenv-afaa-gym
* Blog: [Deep Dive Writeup](https://github.com/sharad0x/Sovereign-SRE-Gym/blob/main/blog.md)
* Repository: https://github.com/sharad0x/Sovereign-SRE-Gym

---

# 🧠 Note

This environment is designed to explore how RL systems behave when **reasoning quality—not just final correctness—is treated as a core objective.**
