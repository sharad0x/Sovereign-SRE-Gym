# 🕵️ Adaptive Fraud Audit Arena: Reasoning Under Unreliable Information

Most environments evaluate whether an agent can reach a correct answer.

This environment explores a different question:

> **How does an agent behave when the information itself is unreliable?**

---

## 🧭 The Setup

The agent acts as an auditor investigating fraud across departments.

At each step, it can:

* interact with multiple sources
* query a database
* submit a final decision

The objective is to identify the root cause within a limited budget.

---

## ⚠️ Conflicting Signals

Different sources may disagree:

* CFO may point to one department
* Whistleblower may point to another

This forces the agent to evaluate consistency over time rather than relying on a single signal.

---

## 🛠️ Imperfect Tools

The database provides structured outputs, but:

* some results may be noisy
* some may be misleading

The agent must interpret outputs rather than treating them as ground truth.

---

## 🧠 Explicit Reasoning State

The environment exposes reasoning through:

* belief distribution
* entropy (uncertainty)
* conflict score

This allows tracking how the agent’s understanding evolves.

---

## 🔁 Dynamic Environment

The underlying structure can change during an episode.

* relationships may shift
* prior conclusions may become outdated

The agent must adapt its reasoning accordingly.

---

## ⚙️ What the Agent Must Learn

The agent must:

* compare conflicting signals
* manage uncertainty
* maintain consistent beliefs
* decide when to commit

This goes beyond simple prediction.

---

## 📊 Training Dynamics

Training was performed under two setups:

- Main Run:
  - Steps: 120  
  - Batch Size: 8  
  - Gradient Accumulation: 1  

- Short-Run Diagnostic:
  - Steps: 8  
  - Batch Size: 2  

Training Notebooks:

* Main Training: https://drive.google.com/file/d/1apF3WIndnPr5d4QzrU3kIfQIutibqX5J/view?usp=drive_link  
* Short-run Diagnostic: https://drive.google.com/file/d/1n6-xjJ4C9kUrDwM8NHh5W-l2OWvojmH3/view?usp=sharing  

---

### Short-Run Behavior (8 Steps)

![8-Step Reward Curve](./assets/8steps_reward_curve.png)

Observations:

- Extremely high variance across steps  
- Rapid reward swings driven by stochastic transitions  
- No observable stabilization within such a short horizon  

This run is included as a **sanity check** to show immediate interaction dynamics.  
It is not representative of training performance.

---

### Reward Curve (120 Steps)

![Reward Curve](./assets/120steps_reward_curve.png)

Observations:

- High variance due to stochastic environment and multi-source signals  
- Frequent negative spikes caused by penalties (invalid reasoning / inefficiency)  
- Smoothed trend shows partial stabilization, though variance remains high  

---

### Loss Curve

![Loss Curve](./assets/loss_curve.png)

Observations:

- Loss is highly noisy with frequent sign changes  
- No clear downward trend over 120 steps  
- Large variance across updates despite small batch size  
- Occasional spikes indicate unstable gradient updates  

---

### Interpretation

- Training is not yet stable; the policy is still exploring and updating aggressively  
- Reward signals remain high-variance due to stochastic transitions and delayed feedback  
- The short-run curve highlights immediate instability, while the longer run shows partial stabilization  

- Loss behavior is not a reliable convergence indicator in this setup  
- Reward trends provide a more interpretable signal of learning progress

---

## 🔍 Key Observation

The agent does not immediately learn correct answers.

Instead, it first learns to:

* reduce contradictions
* stabilize belief updates
* avoid misleading signals

---

## 🧠 Why This Matters

Many real-world systems operate under:

* incomplete information
* conflicting sources
* changing conditions

AFAA provides a controlled environment to study such behavior.

---

## 🏁 Summary

> **Reliable decisions require reasoning—not just prediction.**

---

## 🔗 Links

* Hugging Face Space: https://huggingface.co/spaces/sharad0x/openenv-afaa-gym
* README: https://github.com/sharad0x/Sovereign-SRE-Gym/blob/main/README.md
* Repository: https://github.com/sharad0x/Sovereign-SRE-Gym
