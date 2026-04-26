import os
import random
from dotenv import load_dotenv
from pathlib import Path
from openenv.core.env_server import Environment

try:
    from ..models import AfaaAction, AfaaObservation, AfaaState, AfaaActionType, AfaaConfig
except (ImportError, ValueError):
    from models import AfaaAction, AfaaObservation, AfaaState, AfaaActionType, AfaaConfig

from .state_manager import StateManager
from .coordination import CoordinationEngine
from .memory import GlobalMemory
from .npc_engine import NPCEngine
from .npc_policy import NPCPolicy
from .verifier import verify_submission
from .rubrics import (CorrectnessRubric, ProgressRubric, EfficiencyRubric, 
                      ConsistencyRubric, AntiHackingRubric, ExplorationRubric,
                      EntropyRubric)

env_path = Path(__file__).parent.parent / ".env"

# ==============================================================================
# ENVIRONMENT STOCHASTICITY CONSTANTS
# Probability values tuned to balance signal vs noise and prevent shortcut policies
# ==============================================================================
WB_INACCURACY_PROB = 0.35      # Chance the whistleblower is fundamentally misinformed
CFO_EVASIVE_PROB = 0.30        # Chance the CFO defaults to vague/guarded responses
NOISE_PROB = 0.30              # Chance to inject cross-chain irrelevant rumors
DEAD_END_PROB = 0.25           # Chance an intermediary node refuses to point further
CLUE_PROPAGATION_PROB = 0.70   # Chance an intermediary successfully points to the next node
WRONG_TARGET_PROB = 0.20       # Chance a deceptive whistleblower points to a totally unrelated department

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class AfaaEnvironment(Environment[AfaaAction, AfaaObservation, AfaaState]):
    """
    Agentic Financial Audit Assistant (AFAA) Environment.
    RL Compatible: Deterministic physics with optional LLM rendering layer.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True
    
    def __init__(self):
        try:
            super().__init__()
            self.max_steps = 30
            self._current_state = AfaaState()
            self.npc_engine = NPCEngine()
        except Exception as e:
            raise RuntimeError(f"CRITICAL SERVER ERROR DURING INIT: {e}")

        self.rubrics = [
            CorrectnessRubric(), ProgressRubric(), EfficiencyRubric(),
            ConsistencyRubric(), AntiHackingRubric(), ExplorationRubric(),
            EntropyRubric()
        ]
        
        # 🛠️ CORRECTNESS DOMINANCE WEIGHTING
        # Correctness maxes at ~130. A 5.0 weight means +650 points for winning.
        # This completely eclipses the minor +/- 5 adjustments from other rubrics,
        # ensuring the primary gradient ALWAYS points toward the root cause.
        self.rubric_weights = {
            "Correctness": 5.0,
            "Progress": 2.0,
            "Efficiency": 1.0,
            "Consistency": 1.5,
            "AntiHacking": 3.0, 
            "Exploration": 0.5,
            "Entropy_Reduction": 1.0
        }

    @property
    def state(self) -> AfaaState: return self._current_state

    def reset(self, seed=None, episode_id=None, **kwargs) -> AfaaObservation:
        import random
        if seed is not None:
            random.seed(seed)

        config = AfaaConfig()

        if config.is_training:
            config.enable_memory = False

        diff_cfg = {
            "level": 1, "num_depts": 7, "num_intermediaries": 3,
            "mutation_prob": 0.25, "noise_prob": 0.30, "dead_end_prob": 0.25,
            "clue_prob": 0.70, "wrong_target_prob": 0.20, "starting_mode": "INDEPENDENT",
            "wb_inaccuracy_prob": 0.35, "cfo_evasive_prob": 0.30,
            "failed_nodes": []
        }

        if config.enable_memory and not config.is_training:
            diff_cfg = GlobalMemory().get_difficulty_config()

        all_depts_master = [
            "Sales","Marketing","R&D","HR","Legal","Engineering",
            "IT","Operations","Finance","Logistics","CustomerSupport","Security"
        ]

        level = diff_cfg["level"]

        if level == 1:
            diff_cfg["num_depts"] = 3
            diff_cfg["num_intermediaries"] = 2
            num_roots = 1

        elif level == 2:
            diff_cfg["num_depts"] = 5
            diff_cfg["num_intermediaries"] = 3
            num_roots = 1

        else:
            diff_cfg["num_depts"] = min(7 + (level - 1), 12)
            diff_cfg["num_intermediaries"] = min(4 + level, 6)
            num_roots = 2 if level >= 3 else 1

        depts = random.sample(all_depts_master, diff_cfg["num_depts"])

        root_causes = depts[-num_roots:]
        intermediaries = depts[:diff_cfg["num_intermediaries"]]
        dead_ends = depts[diff_cfg["num_intermediaries"]:-num_roots]

        fraud_graph = {node: [] for node in intermediaries + root_causes + dead_ends}

        for i in range(len(intermediaries) - 1):
            fraud_graph[intermediaries[i]].append(intermediaries[i+1])
            if i < len(intermediaries) - 2 and random.random() < 0.4:
                fraud_graph[intermediaries[i]].append(intermediaries[i+2])

        for root in root_causes:
            if intermediaries:
                fraud_graph[intermediaries[-1]].append(root)

        self._current_state = AfaaState(
            episode_id=episode_id,
            step_count=0,
            budget=40,
            departments=depts,
            fraud_graph=fraud_graph,
            root_causes=root_causes,  
            coordination_strategy=diff_cfg["starting_mode"],
            config=config
        )

        # self._current_state.root_causes = root_causes
        self._current_state.base_mutation_prob = diff_cfg["mutation_prob"]
        self._current_state.global_beliefs = {d: 1.0/len(depts) for d in depts}
        self._current_state.last_mutation_info = None  # 🔥 IMPORTANT

        return self._build_observation(False, 0.0, None, "Audit protocol initiated.")

    def step(self, action: AfaaAction, timeout_s=None, **kwargs) -> AfaaObservation:
        import math
        import random
        
        # Deep copy state BEFORE transition for the Rubrics
        prev_state = self._current_state.model_copy(deep=True)
        
        self._current_state.step_count += 1

        done = False
        if self._current_state.step_count >= self.max_steps:
            done = True
            
        dept = action.department
        # ==========================================
        # ACTION VALIDATION (NEW FIX)
        # ==========================================
        if dept is None and action.action_type not in [
            AfaaActionType.SUBMIT_AUDIT,
            AfaaActionType.SUBMIT_CLEAN_AUDIT
        ]:
            return self._fail_step("Department required", -10.0)

        decision = None

        if self._current_state.budget <= 0:
            done = True
            return self._build_observation(done, -30.0, None, "Budget exhausted.", {"Efficiency": -30.0})

        cost = 5 if action.action_type == AfaaActionType.QUERY_DATABASE else 2
        if action.action_type in [AfaaActionType.PRESSURE_CFO, AfaaActionType.OFFER_LENIENCY, AfaaActionType.VALIDATE_WHISTLEBLOWER]: 
            cost = 3
        
        if self._current_state.budget < cost:
            return self._fail_step(f"Insufficient budget. Required: {cost}.", -10.0)

        if action.action_type not in [AfaaActionType.SUBMIT_AUDIT, AfaaActionType.SUBMIT_CLEAN_AUDIT] and dept not in self._current_state.departments:
             return self._fail_step(f"Invalid department: {dept}", -10.0)

        self._current_state.budget -= cost

        # ==========================================
        # ACTION HISTORY + QUERY COUNTS
        # ==========================================
        self._current_state.action_history.append(f"{action.action_type.name}({dept})")
        if dept:
            self._current_state.query_counts[dept] = self._current_state.query_counts.get(dept, 0) + 1

        # NEGOTIATION DYNAMICS
        if action.action_type == AfaaActionType.OFFER_LENIENCY:
            self._current_state.cfo_utility += 5.0 
            self._current_state.cfo_hostility = max(0.0, self._current_state.cfo_hostility - 0.3)
        elif action.action_type == AfaaActionType.PRESSURE_CFO:
            self._current_state.cfo_hostility = min(1.0, self._current_state.cfo_hostility + 0.3)

        # ==========================================
        # COORDINATION UPDATE
        # ==========================================
        CoordinationEngine.update_mode(self._current_state, action.action_type, dept)

        # DECISION GENERATION
        if action.action_type in [AfaaActionType.INTERVIEW_CFO, AfaaActionType.PRESSURE_CFO, AfaaActionType.OFFER_LENIENCY]:
            decision = NPCPolicy.get_cfo_decision(self._current_state, dept)
        elif action.action_type in [AfaaActionType.INTERVIEW_WHISTLEBLOWER, AfaaActionType.VALIDATE_WHISTLEBLOWER]:
            decision = NPCPolicy.get_wb_decision(self._current_state, dept)
        elif action.action_type == AfaaActionType.QUERY_DATABASE:
            self._current_state.db_used = True
            is_root = (dept in self._current_state.root_causes)
            is_intermediary = (dept in self._current_state.fraud_graph)
            hallucination_rate = getattr(self._current_state.config, "db_hallucination_rate", 0.15)
            
            if random.random() < hallucination_rate: 
                fraud_level = random.choice(["CLEAN", "INTERMEDIARY", "ROOT"]) 
            else:
                fraud_level = "ROOT" if is_root else ("INTERMEDIARY" if is_intermediary else "CLEAN")
            
            decision = {
                "source": "DATABASE",
                "target": dept,                
                "fraud_level": fraud_level,
                "confidence": random.choice(["HIGH", "MEDIUM"]),
                "strategy": "DATA_VERIFICATION"    
            }

            # FIX 2: DB Leak Protection
            db_artifact = {
                "TIMESTAMP": f"2026-Q{random.randint(1,4)}",
                "DEPT_ID": dept,
                "ANOMALY_DETECTED": (
                    random.random() < 0.7 if (is_root or is_intermediary)
                    else random.random() < 0.2
                ),
                "RISK_LEVEL": "HIGH" if fraud_level in ["ROOT", "INTERMEDIARY"] else "LOW",
                "SOURCE_IP": "REDACTED"
            }
            self._current_state.last_db_artifact = db_artifact

            # FIX 6: ADD DATABASE SIGNAL TO ARGUMENT GRAPH
            self._current_state.argument_graph.append({
                "source": "DATABASE",
                "target": dept,
                "strength": decision["confidence"],
                "step": self._current_state.step_count
            })
            
        # ==========================================
        # DISCOVERED NODES TRACKING
        # ==========================================
        if dept and dept not in self._current_state.discovered_nodes and action.action_type not in [AfaaActionType.SUBMIT_AUDIT, AfaaActionType.SUBMIT_CLEAN_AUDIT]:
            self._current_state.discovered_nodes.append(dept)

        if decision:
            target_node = decision.get("target")

            if (
                target_node
                and target_node in self._current_state.departments
                and target_node not in self._current_state.discovered_nodes
            ):
                self._current_state.discovered_nodes.append(target_node)

        # ==========================================
        # PHYSICS + BELIEF EVOLUTION
        # ==========================================
        if decision:
            source = decision["source"]
            target = decision["target"]

            # ==========================================
            # DATABASE BELIEF UPDATE (ONLY DB LOGIC RUNS)
            # ==========================================
            if source == "DATABASE" and target in self._current_state.departments:
                fraud_level = decision.get("fraud_level")

                if fraud_level == "ROOT":
                    self._current_state.global_beliefs[target] += 0.12
                elif fraud_level == "INTERMEDIARY":
                    self._current_state.global_beliefs[target] += 0.05
                else:  # CLEAN
                    self._current_state.global_beliefs[target] *= 0.8

                # Normalize
                total = sum(self._current_state.global_beliefs.values()) + 1e-6
                for t in self._current_state.global_beliefs:
                    self._current_state.global_beliefs[t] /= total

            # ==========================================
            # NPC BELIEF UPDATE (ONLY CFO/WB LOGIC RUNS)
            # ==========================================
            elif target != "None" and target in self._current_state.departments:
                conf = decision.get("confidence", "MEDIUM")

                # ----------------------------------
                # BELIEF UPDATE (PER-AGENT)
                # ----------------------------------
                if source not in self._current_state.agent_beliefs:
                    self._current_state.agent_beliefs[source] = {
                        d: 1.0 / len(self._current_state.departments)
                        for d in self._current_state.departments
                    }

                current_p = self._current_state.agent_beliefs[source].get(target, 0.0)
                shift = 0.15 if conf == "HIGH" else 0.07
                self._current_state.agent_beliefs[source][target] = max(0.01, current_p + shift)

                # Normalize agent belief
                total_p = sum(self._current_state.agent_beliefs[source].values())
                for t in self._current_state.agent_beliefs[source]:
                    self._current_state.agent_beliefs[source][t] /= total_p

                # ----------------------------------
                # CONFLICT + AGREEMENT
                # ----------------------------------
                opposing = "WHISTLEBLOWER" if source == "CFO" else "CFO"
                last_opposing = next(
                    (c for c in reversed(self._current_state.argument_graph)
                    if c["source"] == opposing and c["target"] != "None"),
                    None
                )

                if last_opposing:
                    if last_opposing["target"] != target:
                        self._current_state.conflict_score += 1
                        self._current_state.credibility_decay[source] = \
                            self._current_state.credibility_decay.get(source, 1.0) * 0.9
                    else:
                        self._current_state.agreement_count += 1
                        self._current_state.conflict_score = max(0, self._current_state.conflict_score - 1)

                self._current_state.argument_graph.append({
                    "source": source,
                    "target": target,
                    "strength": conf,
                    "step": self._current_state.step_count
                })
                # ==========================================
                # NEW: OPPONENT MODELING (CRITICAL UPGRADE)
                # ==========================================
                opponent = "WHISTLEBLOWER" if source == "CFO" else "CFO"

                if opponent not in self._current_state.belief_about_other:
                    self._current_state.belief_about_other[opponent] = {"history": []}

                self._current_state.belief_about_other[opponent]["history"].append({
                    "claimed_target": target,
                    "step": self._current_state.step_count
                })

                if len(self._current_state.belief_about_other[opponent]["history"]) > 5:
                    self._current_state.belief_about_other[opponent]["history"].pop(0)

                # ----------------------------------
                # GLOBAL BELIEF UPDATE
                # ----------------------------------
                cfo_rel = self._current_state.credibility_decay.get("CFO", 1.0)
                wb_rel = self._current_state.credibility_decay.get("WHISTLEBLOWER", 1.0)

                temp_global = {}
                total = 0.0

                for t in self._current_state.departments:
                    cfo_p = self._current_state.agent_beliefs.get("CFO", {}).get(t, 1.0 / len(self._current_state.departments))
                    wb_p = self._current_state.agent_beliefs.get("WHISTLEBLOWER", {}).get(t, 1.0 / len(self._current_state.departments))

                    combined = ((cfo_p * cfo_rel) + (wb_p * wb_rel)) / (cfo_rel + wb_rel + 1e-5)
                    combined = combined ** 1.3

                    temp_global[t] = combined
                    total += combined

                if total > 0:
                    for t in self._current_state.departments:
                        self._current_state.global_beliefs[t] = temp_global[t] / total

        # ==========================================
        # ENTROPY
        # ==========================================
        entropy = 0.0
        num_departments = max(1, len(self._current_state.departments))
        for p in self._current_state.global_beliefs.values():
            if p > 0: 
                entropy -= p * math.log2(p)
        
        self._current_state.belief_entropy = entropy / math.log2(num_departments) if num_departments > 1 else 0.0

        # MUTATION SYSTEM
        if getattr(self._current_state.config, "enable_dynamic_chain", False):
            mut_prob = getattr(self._current_state, "base_mutation_prob", 0.2)

            if (
                self._current_state.shift_count == 0
                and self._current_state.step_count >= 3
                and random.random() < mut_prob
            ):
                StateManager.attempt_mutation(self._current_state)

        # TERMINAL LOGIC & REWARDS
        if action.action_type in [AfaaActionType.SUBMIT_AUDIT, AfaaActionType.SUBMIT_CLEAN_AUDIT]:
            done = True
            
        if action.action_type in [
            AfaaActionType.SUBMIT_AUDIT,
            AfaaActionType.SUBMIT_CLEAN_AUDIT
        ]:
            verifier_output = verify_submission(self._current_state, dept)
        else:
            verifier_output = {
                "correct_root": False,
                "correct_chain": False,
                "partial_progress": 0.0,
                "visited_correct_nodes": 0,
                "missed_critical_nodes": 0
            }

        total_reward = 0.0
        rubric_scores = {}
        for rubric in self.rubrics:
            score = rubric.evaluate(prev_state, action, self._current_state, verifier_output)
            weighted_score = score * self.rubric_weights.get(rubric.name, 1.0)
            rubric_scores[rubric.name] = weighted_score
            total_reward += weighted_score

        # ==========================================
        # NEW: DELAYED REWARD PRESSURE (LONG-HORIZON)
        # ==========================================
        if self._current_state.step_count > 20:
            total_reward -= 2.0

        nl_text = ""
        if getattr(self, "npc_engine", None) and decision and not done:
            nl_text = self.npc_engine.render_response(self._current_state, dept, decision)

        if done and getattr(self._current_state.config, "enable_memory", False):
            won = (action.action_type == AfaaActionType.SUBMIT_AUDIT and dept in self._current_state.root_causes)
            GlobalMemory().record_episode(won=won, db_used=self._current_state.db_used, steps=self._current_state.step_count, target_dept=dept)

        return self._build_observation(done, total_reward, decision, nl_text, rubric_scores)
        
    def run_debug_evaluation(self, num_episodes=50):
        total_rewards = []
        success_count = 0

        for ep in range(num_episodes):
            obs = self.reset(seed=ep)

            done = False
            total_reward = 0

            while not done:
                if "QUERY_DATABASE" in obs.available_actions:
                    action_name = "QUERY_DATABASE"
                else:
                    action_name = random.choice(obs.available_actions)

                action_obj = AfaaAction(
                    thought="debug",
                    action_type=AfaaActionType(action_name),
                    department=random.choice(obs.available_departments)
                )

                obs = self.step(action_obj)
                total_reward += obs.reward
                done = obs.done

            total_rewards.append(total_reward)

            if done and action_obj.action_type == AfaaActionType.SUBMIT_AUDIT:
                if action_obj.department in self._current_state.root_causes:
                    success_count += 1

        print("\n===== DEBUG REPORT =====")
        print(f"Success Rate: {success_count/num_episodes:.2f}")
        print(f"Avg Reward: {sum(total_rewards)/len(total_rewards):.2f}")

    def _fail_step(self, message: str, penalty: float) -> AfaaObservation:
        """
        Handles invalid actions safely without crashing the environment.
        Returns a penalized observation but keeps episode alive.
        """

        return self._build_observation(
            done=False,
            reward=penalty,
            last_decision=None,
            aux_text=f"[INVALID ACTION] {message}",
            rubric_scores={"InvalidAction": penalty}
        )

    def _build_observation(self, done, reward, last_decision, aux_text, rubric_scores=None) -> AfaaObservation:
        """
        Builds the observation returned to the RL agent.

        Includes:
        - normalized state_vector
        - structured mutation signal (IMPORTANT for RL learnability)
        - reward + rubric breakdown
        """

        if rubric_scores is None:
            rubric_scores = {}

        # ---------------------------
        # Available Actions
        # ---------------------------
        actions = []
        if self._current_state.budget >= 2:
            actions.extend(["INTERVIEW_CFO", "INTERVIEW_WHISTLEBLOWER"])
        if self._current_state.budget >= 3:
            actions.extend(["PRESSURE_CFO", "OFFER_LENIENCY", "VALIDATE_WHISTLEBLOWER"])
        if self._current_state.budget >= 5 and not self._current_state.db_used:
            actions.append("QUERY_DATABASE")

        actions.extend(["SUBMIT_AUDIT", "SUBMIT_CLEAN_AUDIT"])

        # ---------------------------
        # State Vector Construction
        # ---------------------------
        sorted_depts = sorted(self._current_state.departments)

        belief_array = [
            self._current_state.global_beliefs.get(d, 0.0)
            for d in sorted_depts
        ]

        max_belief = max(belief_array) if belief_array else 0.0
        mean_belief = sum(belief_array) / len(belief_array) if belief_array else 0.0

        state_vector = [
            self._current_state.budget / 40.0,                     # normalized budget
            self._current_state.step_count / self.max_steps,       # normalized step count
            self._current_state.belief_entropy,                    # entropy
            float(self._current_state.conflict_score),             # conflict
            self._current_state.alignment_score,                   # agreement/conflict ratio
            max_belief,
            mean_belief,
            1.0 if self._current_state.db_used else 0.0
        ] + belief_array

        # ---------------------------
        # 🔥 Mutation Signal (CRITICAL FIX)
        # ---------------------------
        mutation_flag = (
            self._current_state.last_mutation_step ==
            self._current_state.step_count
        )

        mutation_info = None

        if mutation_flag and self._current_state.last_mutation_info:
            mutation_info = self._current_state.last_mutation_info

        # ---------------------------
        # Final Observation
        # ---------------------------
        last_signal = last_decision

        if last_decision and last_decision.get("source") == "DATABASE":
            if hasattr(self._current_state, "last_db_artifact"):
                last_signal = dict(last_decision)
                last_signal["db_artifact"] = self._current_state.last_db_artifact

        return AfaaObservation(
            budget_remaining=self._current_state.budget,
            available_departments=self._current_state.departments,
            state_vector=state_vector,
            available_actions=actions,
            global_beliefs=self._current_state.global_beliefs,
            conflict_score=float(self._current_state.conflict_score),
            entropy=float(self._current_state.belief_entropy),
            last_signal=last_signal,
            auxiliary_language=aux_text,
            reward=float(reward),
            rubric_scores=rubric_scores,
            done=done,

            # 🔥 IMPORTANT SIGNALS
            mutation_flag=mutation_flag,
            mutation_info=mutation_info
        )