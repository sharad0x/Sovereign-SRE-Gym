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
  - grpo
---

# 🕵️ Adaptive Fraud Audit Arena (AFAA)
**A Sovereign Auditor Gym for training LLMs to detect deception in non-stationary financial environments.**

Welcome to the Adaptive Fraud Audit Arena (AFAA). This production-grade Reinforcement Learning environment, built on **OpenEnv 0.2.3**, challenges LLM agents to act as forensic auditors uncovering root-cause fraud in a complex, shifting corporate graph. 

This project targets **Theme #1 (Multi-Agent & Theory of Mind)** and **Theme #3.1 (Professional World Modeling)** by forcing agents to navigate conflicting NPC incentives, verify claims against ground-truth data, and manage a strict audit budget.

---

## 📖 The Story: When Language Models Enter the Real World

> *“The CFO looks calm. The Whistleblower sounds confident. Both are wrong.”*

Modern LLMs are powerful—but dangerously trusting.

They excel at pattern recognition, but in the real world:
- people lie,
- incentives conflict,
- and truth is rarely stated directly.

---

### 🧠 The Problem We’re Tackling

Most benchmarks train models to **predict text**.

But real-world intelligence requires:
- **distrust under uncertainty**
- **strategic questioning**
- **adaptive reasoning when the world changes mid-task**

AFAA is built around a simple but brutal question:

> **Can an LLM learn to act like an auditor instead of a chatbot?**

---

### 🎭 The Scenario

You are the Lead Auditor.

- Millions are missing.
- The CFO is protecting their reputation.
- The Whistleblower may be right—or dangerously misinformed.
- The fraud chain is not static—it can mutate under pressure.

Every action has a cost.

Every statement may be misleading.

Every delay reduces your chance of uncovering the truth.

---

### ⚠️ Why Standard LLMs Fail Here

A naive LLM will:
- trust high-confidence statements blindly  
- follow the first plausible narrative  
- ignore contradictions  
- fail to re-evaluate after new evidence  

In AFAA, this behavior leads to:
- ❌ wrong accusations  
- ❌ wasted budget  
- ❌ failure to identify the root cause  

---

### 🔁 What Changes Through Training

The agent must evolve from:

> “Which answer sounds right?”

to:

> “Which agent should I trust—and why?”

It learns to:
- detect contradictions between agents  
- strategically choose when to verify vs. probe  
- adapt beliefs after **STATE_SHIFT events**  
- balance cost vs. certainty under strict constraints  

---

### 🧠 The Core Challenge

This is not a QA task.

This is:
> **belief management under adversarial uncertainty**

And the only way to solve it is through **interaction, not memorization**.

---

## 🚀 Environment Innovation: Non-Stationary Deception

AFAA goes beyond standard static RL benchmarks by introducing **Dynamic Topology Shifts**. It specifically tests an agent's ability to maintain a "Theory of Mind" in an adversarial setting.

### The Novel Mechanics:
1. **Dynamic Topology Shifts:** The fraud chain can mutate mid-audit. If the agent pressures the wrong node, the fraudsters might panic and change their cover story, emitting a `mutation_flag`. The agent must learn to discard old beliefs and adapt instantly.
2. **Multi-Agent Incentive Modeling:** The agent receives natural language signals from NPCs. It must learn the difference between an evasive CFO protecting their bonus and a misinformed Whistleblower acting on rumors. 
3. **Resource-Constrained Investigation:**
   * **Action Space:** `QUERY_DATABASE` (High cost, high truth), `INTERVIEW_CFO` (Low cost, high deception risk), `PRESSURE_CFO` (High risk, high reward), `OFFER_LENIENCY`, and `SUBMIT_AUDIT`.
   * **Observation Space:** The agent tracks its `budget_remaining`, `belief_entropy`, `conflict_score`, and a normalized `state_vector` to gauge how close it is to the truth.

---

## 🧠 Why This Environment Matters

AFAA is not just a simulation.

It models real-world decision-making problems:
- financial fraud investigation  
- cybersecurity incident response  
- intelligence analysis under conflicting reports  

Where:
- truth is hidden  
- signals are noisy  
- and wrong decisions are costly  

The goal is not just to find answers—

> it is to learn how to **reason under deception**.

---

## 🧪 Agent Behavior: Before vs After Training

### 🟥 Before Training (Naive Policy)
- Follows first high-confidence signal  
- Rarely uses database efficiently  
- Ignores contradictions between agents  
- Fails when fraud chain mutates  

👉 Behavior: **Reactive, gullible, inefficient**

---

### 🟩 After Training (Learned Policy)
- Cross-validates conflicting claims  
- Uses database only when necessary  
- Detects deception patterns  
- Adapts within 1–2 steps after mutation  

👉 Behavior: **Strategic, skeptical, adaptive**

---

## ⚙️ Training Pipeline & Reward Setup

Our training pipeline is built for efficiency and strict behavioral shaping. We utilize **Group Relative Policy Optimization (GRPO)** with 4-bit Unsloth quantization, connecting a Colab T4 trainer to our live Hugging Face Space via OpenEnv's WebSocket scaling. The package management is handled cleanly via `uv` for reproducible builds.

### The Composable Rubric System
RL is only as good as its reward signal. We use a multi-tiered rubric to objectively score subjective interactions:
* **The "Silver Bullet" (Correctness - 5.0x Weight):** Massive positive reward for `SUBMIT_AUDIT` with the correct root-cause department. Heavy penalties for false accusations.
* **Anti-Hacking Penalty:** Penalizes repetitive action loops and "guessing" (e.g., submitting an audit when the agent's internal belief confidence is mathematically < 0.3).
* **Efficiency Tax:** A standard time penalty forces the agent to solve the audit using the fewest possible budget steps.
* **Consistency Reward:** Rewards stable reasoning and penalizes wild, erratic shifts in the agent's global belief distribution late in the episode.

---

### Why the Reward Design Matters

The reward system is designed to prevent shortcut learning.

An agent that:
- blindly trusts NPCs  
- overuses database queries  
- or randomly submits audits  

will consistently receive lower rewards.

Only agents that:
- manage uncertainty  
- resolve contradictions  
- and act efficiently  

can achieve high scores.

This ensures that training reflects **real capability improvement**, not reward exploitation.

---

## 📈 Evidence of Training (In Progress)

Training is currently underway.

This section will include:

- 📊 Reward curves across episodes  
- 📉 Entropy reduction trends (belief stabilization)  
- 🔁 Before vs after behavior comparisons  
- 🎯 Success rate improvements  

These metrics will demonstrate that the agent is not only improving rewards—but also learning **strategic, deception-aware reasoning**.