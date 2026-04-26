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

🔗 **Hugging Face Space:** *[ADD LINK]*
🎥 **Demo Video (≤2 min):** *[ADD LINK]*
📝 **Blog / Deep Dive:** *[ADD LINK]*

---

# 📌 Motivation

Many existing environments evaluate whether an agent can reach a correct answer given available signals.

AFAA focuses on a different aspect:

> **How does an agent behave when the signals themselves are unreliable, incomplete, or strategically misleading?**

This includes situations where:

* different sources provide conflicting claims
* tools may return corrupted or manipulated data
* the underlying system changes during interaction

The goal is to move beyond “answer correctness” toward evaluating **decision-making under uncertainty**.

---

# 🧭 Environment Overview

The agent plays the role of an auditor investigating a set of departments connected through a hidden fraud graph.

At each step, the agent can:

* interact with different information sources
* query a database
* apply pressure or negotiation strategies
* submit a final audit decision

The episode ends when the agent submits its conclusion or runs out of budget.

---

## ⚙️ Core Mechanics

### 1. Multi-Agent Information Sources

Two primary entities provide information:

* **CFO** → may cooperate or strategically mislead
* **Whistleblower** → may be accurate or noisy

Their behavior is not fixed; it depends on internal incentives and coordination modes.

**Engineering detail:**

* Decisions are generated through deterministic + stochastic policy logic
* Claims are stored in an argument graph with decay and credibility tracking

---

### 2. Structured but Unreliable Tooling

The agent can query a database for high-value signals.

However:

* responses may be partially corrupted
* anomalies are probabilistic, not deterministic
* structured artifacts (e.g., integrity flags, metadata) must be interpreted

**Engineering detail:**

* database responses include structured fields like `DATA_INTEGRITY`
* corruption is injected probabilistically
* signals are exposed through observation space (not hidden)

---

### 3. Non-Stationary Environment

The fraud structure is not guaranteed to remain fixed.

* connections between nodes may change
* previously valid reasoning paths can become outdated

**Engineering detail:**

* controlled mutation system (`STATE_SHIFT`)
* mutation events are explicitly surfaced via observation
* bounded to preserve RL stability

---

### 4. Belief-Based State Representation

Instead of discrete labels, the agent maintains a belief distribution over departments.

**State includes:**

* `global_beliefs` (probability distribution)
* entropy (uncertainty measure)
* conflict score (signal disagreement)

This enables:

* continuous reasoning
* measurable uncertainty reduction

---

# 🧠 What Is Being Evaluated

The environment is designed to evaluate three aspects:

---

## 1. Decision Accuracy

Can the agent correctly identify the root cause?

---

## 2. Reasoning Stability

Does the agent maintain a consistent hypothesis over time?

**Engineering detail:**

* temporal consistency tracking
* penalties for oscillating belief patterns

---

## 3. Robustness to Deception

Does the agent detect and handle misleading signals?

**Engineering detail:**

* adversarial database conditions
* penalties for blind trust in corrupted data
* reward adjustments based on reasoning context

---

# 🧪 Reward Design

AFAA uses a composable rubric-based reward system.

### Components:

* **Correctness** → final outcome accuracy
* **Progress** → discovery of relevant nodes
* **Efficiency** → cost-aware behavior
* **Consistency** → stability of beliefs
* **Anti-Hacking** → prevents shortcut policies
* **Entropy Reduction** → encourages convergence
* **Reasoning Signals** → grounded and explainable decisions

---

## Why This Matters

The reward is not only based on *what* the agent does, but also *how* it arrives there.

This reduces:

* random guessing
* over-reliance on single signals
* unstable decision policies

---

# 🔍 Example Behavior Shift

### Before Training

* reacts to latest signal
* trusts high-confidence outputs
* frequently changes hypothesis

---

### After Training

* compares multiple sources
* treats tool outputs cautiously
* maintains stable belief trajectory
* adapts after environment changes

---

# 📈 Training Results (To Be Added)

This section will include:

* reward curves over episodes
* entropy reduction trends
* success rate improvement
* trajectory comparison (before vs after)

---

# 🎥 Demo (To Be Added)

Short video demonstrating:

* baseline behavior
* trained agent behavior
* handling of misleading signals

---

# 📝 Blog / Deep Dive (To Be Added)

Will cover:

* design decisions
* reward shaping rationale
* failure cases
* lessons learned

---

# 🧠 Why This Environment Is Useful

AFAA is relevant for scenarios where:

* information sources are not fully reliable
* decisions must be made incrementally
* systems may change during interaction

Examples include:

* financial auditing
* security incident investigation
* multi-source intelligence analysis

---

# 🔗 Links

* Hugging Face Space: *[ADD]*
* Video Demo: *[ADD]*
* Blog: *[ADD]*

---

# 🙌 Notes

This environment is designed to explore how RL can be used not just for optimization, but for **structured reasoning under uncertainty**.
