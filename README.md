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

**A Sovereign SRE Gym for training LLMs to detect deception in non-stationary financial environments.**

[cite_start]Adaptive Fraud Audit Arena (AFAA) is a production-grade RL environment built on **OpenEnv**[cite: 40]. It challenges agents to act as auditors uncovering root-cause fraud in a complex, shifting corporate graph. [cite_start]This project targets **Theme #1 (Multi-Agent)** and **Theme #3.1 (Professional World Modeling)** by forcing agents to navigate conflicting NPC incentives and verify claims against a ground-truth database[cite: 1, 15].

---

## 🎯 The Problem (Capability Gap)
[cite_start]Current LLMs often struggle with **Theory of Mind** in adversarial settings—they are easily misled by deceptive agents[cite: 4]. AFAA provides a sandbox where an auditor must learn to:
1. [cite_start]**Model Incentives**: Identify when a CFO is being evasive vs. when a Whistleblower is misinformed[cite: 2, 3].
2. **Handle Non-Stationarity**: Adapt to "Topology Shifts" where the fraud chain mutates mid-audit.
3. **Budget Management**: Strategically spend limited resources on high-cost Database queries vs. low-cost NPC interviews.

---

## 🛠️ Environment Mechanics

### What the Agent Sees (Observation Space)
The agent receives a multi-modal observation:
* **State Vector**: A normalized tensor tracking budget, step count, belief entropy, and current conflict scores.
* **Global Beliefs**: A distribution representing the agent's current suspicion level for every department.
* **Natural Language Signal**: Real-time dialogue from NPCs (CFO/Whistleblower) or structured logs from the Database.
* **Mutation Signal**: A binary flag indicating if the environment's topology has shifted.

### What the Agent Does (Action Space)
* **Intelligence**: `QUERY_DATABASE` (High cost, high confidence), `INTERVIEW_CFO`, `INTERVIEW_WHISTLEBLOWER`.
* **Negotiation**: `PRESSURE_CFO` (increases hostility), `OFFER_LENIENCY` (builds utility), `VALIDATE_WHISTLEBLOWER`.
* **Decision**: `SUBMIT_AUDIT` (Target a department) or `SUBMIT_CLEAN_AUDIT`.

---

## ⚖️ Reward Logic (The Rubrics)
[cite_start]We use a composable rubric system to provide a rich learning signal[cite: 90, 92]:
* **Correctness (5.0x weight)**: Massive reward for identifying the correct root cause; heavy penalty for false accusations.
* **Anti-Hacking**: Penalizes repetitive action loops and "guessing" without sufficient evidence (belief confidence < 0.3).
* **Efficiency**: A standard time penalty forces the agent to solve the audit in fewer steps.
* **Consistency**: Rewards stable reasoning and penalizes wild suspicion shifts late in the episode.

---

## 🚀 Training Quickstart

### 1. Deploy the Environment
Push this repository to your Hugging Face Space:
```
openenv push --repo-id your-username/afaa-env
```
### 2. Connect to Colab
In your training notebook, use the AfaaEnvClient to point to your live Space URL:
```
from client import AfaaEnvClient
# Use your actual Space URL
env = AfaaEnvClient(base_url="https://your-username-afaa-env.hf.space")
```
3. Training Results(Placeholder: After training, add your Reward/Loss curves here to satisfy the 20% Judging Criterion )📂 File Structuremodels.py: Type-safe Pydantic definitions for RL stability.server/AFAA_environment.py: Core logic, belief evolution, and budget physics.server/rubrics.py: The multi-faceted reward engine.client.py: Async-ready OpenEnv client for high-throughput training.