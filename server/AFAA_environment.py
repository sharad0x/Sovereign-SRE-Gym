import os
import random
import json
from openai import OpenAI
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

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

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

class AfaaEnvironment(Environment[AfaaAction, AfaaObservation, AfaaState]):
    """
    Agentic Financial Audit Assistant (AFAA) Environment.
    
    This environment simulates a partially observable Markov decision process (POMDP) 
    in a corporate audit setting. 
    
    - Multi-Step Dependency Chain: Enforces long-horizon reasoning by requiring agents 
      to trace fraud back to a root cause.
    - Multi-Agent Setup: CFO and Whistleblower act as independent agents with distinct 
      belief models, requiring Theory of Mind (ToM) to resolve contradictions.
    - Observation Model: P(o | s, a) - Signals are stochastic, language-based projections 
      of the hidden `fraud_chain`.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True
    
    def __init__(self):
        try:
            super().__init__()
            self.max_steps = 20
            self._current_state = AfaaState()
            self.npc_engine = NPCEngine()
        except Exception as e:
            raise RuntimeError(f"CRITICAL SERVER ERROR DURING INIT: {e}")

    @property
    def state(self) -> AfaaState: return self._current_state

    def reset(self, seed=None, episode_id=None, **kwargs) -> AfaaObservation:
        import random
        if seed is not None:
            random.seed(seed)
            
        config = AfaaConfig()
        
        diff_cfg = {
            "level": 1, "num_depts": 7, "num_intermediaries": 3, 
            "mutation_prob": 0.25, "noise_prob": 0.30, "dead_end_prob": 0.25, 
            "clue_prob": 0.70, "wrong_target_prob": 0.20, "starting_mode": "INDEPENDENT",
            "wb_inaccuracy_prob": 0.35, "cfo_evasive_prob": 0.30,
            "failed_nodes": []
        }
        
        if config.enable_memory:
            diff_cfg = GlobalMemory().get_difficulty_config()
            
        all_depts_master = ["Sales", "Marketing", "R&D", "HR", "Legal", "Engineering", "IT", "Operations", "Finance", "Logistics", "CustomerSupport", "Security"]
        depts = random.sample(all_depts_master, diff_cfg["num_depts"])
        
        # 1. Complex Graph Generation (Multi-Root & Cross-Links)
        num_roots = 2 if diff_cfg["level"] >= 3 else 1
        root_causes = depts[-num_roots:]
        intermediaries = depts[:diff_cfg["num_intermediaries"]]
        dead_ends = depts[diff_cfg["num_intermediaries"]:-num_roots]

        # Adversarial Memory: Force recent failed nodes into dead-ends
        for failed_node in diff_cfg.get("failed_nodes", []):
            if failed_node in intermediaries:
                intermediaries.remove(failed_node)
                dead_ends.append(failed_node)
        
        fraud_graph = {node: [] for node in intermediaries + root_causes + dead_ends}
        for i in range(len(intermediaries) - 1):
            fraud_graph[intermediaries[i]].append(intermediaries[i+1])
            if i < len(intermediaries) - 2 and random.random() < 0.4:
                fraud_graph[intermediaries[i]].append(intermediaries[i+2]) # Cross-links
                
        for root in root_causes:
            if intermediaries: fraud_graph[intermediaries[-1]].append(root)
            
        for i, dead_end in enumerate(dead_ends):
            if intermediaries:
                idx = i % len(intermediaries)
                fraud_graph[intermediaries[idx]].append(dead_end)

        # 2. Information Asymmetry Generation
        cfo_known = {k: v.copy() for k, v in fraud_graph.items()}
        if len(intermediaries) > 2 and random.random() < 0.5:
            cfo_known[intermediaries[-2]] = [] # CFO hides/forgets a node

        wb_noisy = {k: v.copy() for k, v in fraud_graph.items()}
        for node in intermediaries:
            if random.random() < 0.4 and dead_ends:
                wb_noisy[node].append(random.choice(dead_ends)) # WB sees ghosts
            
        self._current_state = AfaaState(
            episode_id=episode_id, step_count=0, budget=40, departments=depts, 
            fraud_graph=fraud_graph, root_cause=root_causes[0], # Backward compat
            cfo_known_graph=cfo_known, wb_noisy_graph=wb_noisy,
            wb_query_count=0, db_used=False, db_used_step=-1, action_history=[],
            wb_is_accurate=(random.random() >= diff_cfg["wb_inaccuracy_prob"]), 
            cfo_base_evasive=(random.random() < diff_cfg["cfo_evasive_prob"]),
            difficulty_level=diff_cfg["level"],
            base_mutation_prob=diff_cfg["mutation_prob"], mutation_prob=diff_cfg["mutation_prob"],
            noise_prob=diff_cfg["noise_prob"], dead_end_prob=diff_cfg["dead_end_prob"],
            clue_prob=diff_cfg["clue_prob"], wrong_target_prob=diff_cfg["wrong_target_prob"],
            coordination_strategy=diff_cfg["starting_mode"], config=config
        )
        self._current_state.root_causes = root_causes
        self._current_state.global_beliefs = {d: 1.0/len(depts) for d in depts} # Uniform prior

        return AfaaObservation(
            budget_remaining=self._current_state.budget, available_departments=self._current_state.departments,
            latest_text=json.dumps({"source": "SYSTEM", "nl_message": "Audit protocol initiated. Target: Absolute Root Cause."}),
            rule_violations=[], done=False, reward=0.0
        )

    def step(self, action: AfaaAction, timeout_s=None, **kwargs) -> AfaaObservation:
        import math, json, random
        self._current_state.step_count += 1
        reward = -1.0; done = False; latest_text = ""; dept = action.department

        if self._current_state.budget <= 0: return self._fail_step("Budget exhausted.", -30.0)

        cost = 0
        if action.action_type in [AfaaActionType.INTERVIEW_CFO, AfaaActionType.INTERVIEW_WHISTLEBLOWER]: cost = 2
        elif action.action_type in [AfaaActionType.PRESSURE_CFO, AfaaActionType.OFFER_LENIENCY, AfaaActionType.VALIDATE_WHISTLEBLOWER]: cost = 3
        elif action.action_type == AfaaActionType.QUERY_DATABASE: cost = 8 if self._current_state.db_used else 5 

        if self._current_state.budget < cost: return self._fail_step(f"Insufficient budget. Required: {cost}.", -10.0)
        if action.action_type != AfaaActionType.SUBMIT_CLEAN_AUDIT and dept not in self._current_state.departments:
             return self._fail_step(f"Invalid department: {dept}", -10.0)

        self._current_state.action_history.append(f"{action.action_type.name}({dept})")
        if dept: self._current_state.query_counts[dept] = self._current_state.query_counts.get(dept, 0) + 1

        # 1. Action-Incentive Negotiation
        if action.action_type == AfaaActionType.OFFER_LENIENCY:
            self._current_state.cfo_utility += 5.0 
            self._current_state.cfo_hostility = max(0.0, self._current_state.cfo_hostility - 0.3)
        elif action.action_type == AfaaActionType.PRESSURE_CFO:
            self._current_state.cfo_hostility = min(1.0, self._current_state.cfo_hostility + 0.3)

        # 2. Entropy & Probabilistic Phase Transitions
        entropy = 0.0
        if getattr(self._current_state, "global_beliefs", {}):
            for p in self._current_state.global_beliefs.values():
                if p > 0: entropy -= p * math.log2(p)
        self._current_state.belief_entropy = entropy

        chaos_pressure = (self._current_state.conflict_score * 0.1) + (entropy * 0.2)
        if random.random() < chaos_pressure:
            self._current_state.cfo_phase = "CHAOTIC" if random.random() < 0.5 else "STABLE"
            self._current_state.wb_phase = "CHAOTIC" if random.random() < 0.5 else "STABLE"
        
        if self._current_state.cfo_phase == "CHAOTIC" and random.random() < 0.2: self._current_state.cfo_phase = "RECOVERY"
        if self._current_state.wb_phase == "CHAOTIC" and random.random() < 0.2: self._current_state.wb_phase = "RECOVERY"

        # 3. Continuous Multi-Mutation Dynamics
        self._current_state.mutation_decay_factor = min(1.0, getattr(self._current_state, "mutation_decay_factor", 1.0) + 0.05)
        current_mut_prob = getattr(self._current_state, "base_mutation_prob", 0.25) * self._current_state.mutation_decay_factor * (1.0 + chaos_pressure)

        mutated = False
        highest_global_belief = max(self._current_state.global_beliefs, key=self._current_state.global_beliefs.get, default=None)
        
        while random.random() < current_mut_prob and highest_global_belief in self._current_state.root_causes:
            if StateManager.attempt_mutation(self._current_state):
                mutated = True
                current_mut_prob *= 0.3 
                self._current_state.mutation_decay_factor *= 0.3
            else: break

        mode_shifted = CoordinationEngine.update_mode(self._current_state, action.action_type, dept)

        # 4. Engine Execution
        if action.action_type == AfaaActionType.INTERVIEW_CFO:
            self._current_state.budget -= cost
            latest_text = self.npc_engine.get_cfo_response(self._current_state, dept, self._current_state.noise_prob, self._current_state.dead_end_prob, self._current_state.clue_prob)
        elif action.action_type == AfaaActionType.INTERVIEW_WHISTLEBLOWER:
            self._current_state.budget -= cost
            latest_text = self.npc_engine.get_wb_response(self._current_state, dept, self._current_state.noise_prob, self._current_state.dead_end_prob, self._current_state.clue_prob, self._current_state.wrong_target_prob)
        elif action.action_type == AfaaActionType.PRESSURE_CFO:
            self._current_state.budget -= cost
            latest_text = self.npc_engine.get_cfo_response(self._current_state, dept, self._current_state.noise_prob, self._current_state.dead_end_prob, self._current_state.clue_prob, action_context="PRESSURE_CFO")
        elif action.action_type == AfaaActionType.OFFER_LENIENCY:
            self._current_state.budget -= cost
            latest_text = self.npc_engine.get_cfo_response(self._current_state, dept, self._current_state.noise_prob, self._current_state.dead_end_prob, self._current_state.clue_prob, action_context="OFFER_LENIENCY")
        elif action.action_type == AfaaActionType.VALIDATE_WHISTLEBLOWER:
            self._current_state.budget -= cost
            latest_text = self.npc_engine.get_wb_response(self._current_state, dept, self._current_state.noise_prob, self._current_state.dead_end_prob, self._current_state.clue_prob, self._current_state.wrong_target_prob, action_context="VALIDATE_WHISTLEBLOWER")

        elif action.action_type == AfaaActionType.QUERY_DATABASE:
            self._current_state.budget -= cost
            self._current_state.db_used = True
            self._current_state.db_used_step = self._current_state.step_count
            
            is_root = (dept in self._current_state.root_causes)
            is_intermediary = (dept in self._current_state.fraud_graph)
            
            hallucination_rate = getattr(self._current_state.config, "db_hallucination_rate", 0.15) if isinstance(self._current_state.config, dict) else 0.15
            if random.random() < hallucination_rate: fraud_level = random.choice(["CLEAN", "INTERMEDIARY", "ROOT"]) 
            else: fraud_level = "ROOT" if is_root else ("INTERMEDIARY" if is_intermediary else "CLEAN")
            
            if "AUDITOR" not in self._current_state.agent_beliefs: self._current_state.agent_beliefs["AUDITOR"] = {}
            current_belief = self._current_state.agent_beliefs["AUDITOR"].get(dept, 0.5)
            shift = 0.2 if fraud_level != "CLEAN" else -0.2
            self._current_state.agent_beliefs["AUDITOR"][dept] = max(0.01, min(1.0, current_belief + shift))

            latest_text = json.dumps({"source": "DATABASE", "nl_message": "Scan complete.", "structured_signals": {"fraud_level": fraud_level, "confidence": random.choice(["HIGH", "MEDIUM"])}})
            reward += 5.0 if (is_root or is_intermediary) else -10.0

        # 5. Continuous Integration (Beliefs & Coordination)
        if action.action_type not in [AfaaActionType.QUERY_DATABASE, AfaaActionType.SUBMIT_AUDIT, AfaaActionType.SUBMIT_CLEAN_AUDIT]:
            try:
                parsed = json.loads(latest_text)
                source = parsed.get("source"); signals = parsed.get("structured_signals", {}); target = signals.get("target")
                
                if target and target != "None":
                    if source not in self._current_state.agent_beliefs: self._current_state.agent_beliefs[source] = {}
                    current_p = self._current_state.agent_beliefs[source].get(target, 0.1)
                    shift = 0.2 if signals.get("confidence") == "HIGH" else -0.1
                    self._current_state.agent_beliefs[source][target] = max(0.01, current_p + shift)
                    
                    total_p = sum(self._current_state.agent_beliefs[source].values())
                    for t in self._current_state.agent_beliefs[source]: self._current_state.agent_beliefs[source][t] /= total_p

                    opposing_source = "WHISTLEBLOWER" if source == "CFO" else "CFO"
                    last_opposing = next((c for c in reversed(self._current_state.argument_graph) if c["source"] == opposing_source and c.get("dept") == dept), None)
                    is_contradiction = last_opposing and last_opposing["target"] != target
                    is_agreement = last_opposing and last_opposing["target"] == target
                    
                    self._current_state.disagreement_rate = (0.8 * self._current_state.disagreement_rate) + (0.2 * (1.0 if is_contradiction else 0.0))

                    if is_contradiction:
                        self._current_state.repeated_lies_penalty[source] = self._current_state.repeated_lies_penalty.get(source, 0) + 1
                        current_cred = self._current_state.credibility_decay.get(source, 1.0)
                        floor = self._current_state.credibility_floor.get(source, 0.1)
                        new_cred = max(floor, current_cred * 0.9)
                        self._current_state.credibility_decay[source] = new_cred
                        
                        if self._current_state.repeated_lies_penalty[source] > 5 and not self._current_state.irreversible_damage.get(source):
                            self._current_state.irreversible_damage[source] = True
                            self._current_state.credibility_floor[source] = 0.01 
                            
                        self._current_state.conflict_score += 1
                    else:
                        self._current_state.alignment_score = (0.8 * self._current_state.alignment_score) + 0.2
                        
                    if self._current_state.alignment_score > 0.7: self._current_state.intent_inference[source] = "COLLUDING"
                    elif self._current_state.disagreement_rate > 0.6: self._current_state.intent_inference[source] = "SABOTAGING_OPPONENT"

                    self._current_state.argument_graph.append({
                        "source": source, "target": target, "dept": dept, "countered_by": opposing_source if is_contradiction else "None",
                        "strength": signals.get("confidence", "MEDIUM"), "step": self._current_state.step_count
                    })
                    
                    all_targets = set(self._current_state.agent_beliefs.get("CFO", {}).keys()).union(self._current_state.agent_beliefs.get("WHISTLEBLOWER", {}).keys())
                    cfo_rel = self._current_state.credibility_decay.get("CFO", 1.0)
                    wb_rel = self._current_state.credibility_decay.get("WHISTLEBLOWER", 1.0)
                    
                    for t in all_targets:
                        cfo_p = self._current_state.agent_beliefs.get("CFO", {}).get(t, 0.0)
                        wb_p = self._current_state.agent_beliefs.get("WHISTLEBLOWER", {}).get(t, 0.0)
                        self._current_state.global_beliefs[t] = ((cfo_p * cfo_rel) + (wb_p * wb_rel)) / (cfo_rel + wb_rel + 1e-5)

                    discount = 0.95 ** self._current_state.step_count
                    prob_exposure = len(self._current_state.discovered_nodes) / max(1, len(self._current_state.fraud_graph))
                    
                    if source == "CFO":
                        cost_deception = 2.0 if target not in self._current_state.root_causes else 0.0
                        self._current_state.cfo_utility += ((5.0 if target not in self._current_state.fraud_graph and target not in self._current_state.root_causes else -5.0) - cost_deception - (5.0 * prob_exposure)) * discount
                    else:
                        cost_exposure = 1.0 if target not in self._current_state.fraud_graph else 0.0
                        self._current_state.wb_utility += ((5.0 if target in self._current_state.fraud_graph or target in self._current_state.root_causes else -5.0) - cost_exposure - (5.0 * prob_exposure)) * discount
            except: pass

        # 6. Terminal Logic & Overconfidence Penalties
        if action.action_type == AfaaActionType.SUBMIT_AUDIT:
            done = True; is_root = dept in self._current_state.root_causes
            global_conf = self._current_state.global_beliefs.get(dept, 0.0)
            
            if is_root:
                reward += 100.0 
                if global_conf > 0.6: reward += 30.0 
                elif global_conf < 0.2: reward -= 40.0
            else: reward -= 50.0 
            
        elif action.action_type == AfaaActionType.SUBMIT_CLEAN_AUDIT:
            done = True; reward -= 40.0

        if mutated:
            alert = json.dumps({"source": "SYSTEM", "nl_message": "WARNING: Topology shift detected.", "structured_signals": {"event": "STATE_SHIFT"}})
            latest_text = f"{latest_text}\n{alert}" if latest_text else alert
        if mode_shifted:
            strat = getattr(self._current_state, "coordination_strategy", "INDEPENDENT")
            alert = json.dumps({"source": "SYSTEM", "nl_message": f"Anomaly detected. Stance: {strat}.", "structured_signals": {"event": "MODE_SHIFT"}})
            latest_text = f"{latest_text}\n{alert}" if latest_text else alert
        if not done and self._current_state.step_count >= self.max_steps:
            done = True; reward -= 30.0

        if done and getattr(self._current_state.config, "enable_memory", False):
            won = (action.action_type == AfaaActionType.SUBMIT_AUDIT and dept in self._current_state.root_causes)
            GlobalMemory().record_episode(won=won, db_used=self._current_state.db_used, steps=self._current_state.step_count, target_dept=dept)

        return AfaaObservation(budget_remaining=self._current_state.budget, available_departments=self._current_state.departments, latest_text=latest_text, done=done, reward=reward)
           
    def _fail_step(self, message: str, reward: float) -> AfaaObservation:
        self._current_state.budget -= 2
        return AfaaObservation(
            budget_remaining=self._current_state.budget, available_departments=self._current_state.departments,
            latest_text=json.dumps({"source": "SYSTEM", "error": message}), rule_violations=[message], done=False, reward=reward
        )