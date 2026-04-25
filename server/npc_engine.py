import os
import json
from openai import OpenAI

try:
    from ..models import AfaaState
except (ImportError, ValueError):
    from models import AfaaState

from .coordination import CoordinationEngine
from .state_manager import StateManager

class NPCEngine:
    def __init__(self):
        nim_api_key = os.getenv("NVIDIA_API_KEY", "").strip('"').strip("'")
        if not nim_api_key: raise ValueError("NVIDIA_API_KEY is missing!")
        self.client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=nim_api_key)
        self.model = os.getenv("NPC_MODEL", "meta/llama-3.1-8b-instruct")

    def _get_best_argument_target(self, state: AfaaState, opponent_source: str) -> str:
        """🛠️ EMERGENT FLAW 3: Calculates structural argument scores with recency decay."""
        argument_score = {}
        for claim in state.argument_graph:
            t = claim["target"]
            if t == "None": continue
            if t not in argument_score: argument_score[t] = 0.0
            
            # Recency Decay
            age = state.step_count - claim.get("step", 0)
            decay = 0.85 ** age
            
            # Source Reliability
            src_rel = state.belief_about_other.get(claim["source"], {}).get("reliability", 0.5)
            if claim["source"] != opponent_source: src_rel = 1.0 # Trusts own past claims heavily
                
            strength_val = (1.0 if claim.get("strength") == "HIGH" else 0.5) * decay * src_rel
            if claim.get("countered_by") != "None":
                strength_val -= (0.5 * decay)
                
            argument_score[t] += strength_val
            
        return max(argument_score, key=argument_score.get) if argument_score and max(argument_score.values()) > 0 else "None"

    def _get_best_argument_target(self, state: AfaaState, opponent_source: str) -> str:
        argument_score = {}
        for claim in state.argument_graph:
            t = claim["target"]
            if t == "None": continue
            if t not in argument_score: argument_score[t] = 0.0
            
            # Recency Decay
            age = state.step_count - claim.get("step", 0)
            decay = 0.85 ** age
            
            # Source Reliability based on Long-Term Credibility
            src_rel = state.credibility_decay.get(claim["source"], 0.5)
            if claim["source"] != opponent_source: src_rel = 1.0 
                
            strength_val = (1.0 if claim.get("strength") == "HIGH" else 0.5) * decay * src_rel
            if claim.get("countered_by") != "None":
                strength_val -= (0.5 * decay)
                
            argument_score[t] += strength_val
            
        return max(argument_score, key=argument_score.get) if argument_score and max(argument_score.values()) > 0 else "None"

    def _get_best_argument_target(self, state: AfaaState, opponent_source: str) -> str:
        argument_score = {}
        for claim in getattr(state, "argument_graph", []):
            t = claim.get("target")
            if t == "None" or not t: continue
            if t not in argument_score: argument_score[t] = 0.0
            
            age = state.step_count - claim.get("step", 0)
            decay = 0.85 ** age
            
            src_rel = getattr(state, "credibility_decay", {}).get(claim.get("source"), 0.5)
            if claim.get("source") != opponent_source: src_rel = 1.0 
                
            strength_val = (1.0 if claim.get("strength") == "HIGH" else 0.5) * decay * src_rel
            if claim.get("countered_by") != "None": strength_val -= (0.5 * decay)
                
            argument_score[t] += strength_val
            
        return max(argument_score, key=argument_score.get) if argument_score and max(argument_score.values()) > 0 else "None"

    def get_cfo_response(self, state: AfaaState, topic: str, noise_prob: float, dead_end_prob: float, clue_prob: float, action_context: str = None) -> str:
        import random, json
        my_phase = getattr(state, "cfo_phase", "STABLE")
        
        if getattr(state, "cfo_utility", 0) < getattr(state, "wb_utility", 0): strategy = "DECEPTION"
        else: strategy = "COOPERATION"

        best_arg_target = self._get_best_argument_target(state, "WHISTLEBLOWER")
        root_causes = getattr(state, "root_causes", [state.root_cause])
        highest_global_belief = max(getattr(state, "global_beliefs", {"None": 1}), key=getattr(state, "global_beliefs", {}).get, default="None")

        wb_intent = getattr(state, "intent_inference", {}).get("WHISTLEBLOWER", "UNKNOWN")
        alignment = getattr(state, "alignment_score", 0.5)

        # Use Asymmetric Knowledge Base
        my_knowledge = getattr(state, "cfo_known_graph", state.fraud_graph)

        if strategy == "DECEPTION":
            # Weaponize the Auditor's Confirmation Bias
            if highest_global_belief not in state.fraud_graph and highest_global_belief not in root_causes and highest_global_belief != "None":
                target = highest_global_belief
            else:
                clean_depts = [d for d in state.departments if d not in state.fraud_graph and d not in root_causes]
                target = best_arg_target if (best_arg_target in clean_depts) else (random.choice(clean_depts) if clean_depts else "None")
        else:
            next_nodes = my_knowledge.get(topic, [])
            target = next_nodes[0] if next_nodes else (root_causes[0] if root_causes else "None")

        db_awareness = ""
        if state.db_used and state.step_count > getattr(state, "db_used_step", -1) + 1:
            db_awareness = f" Aware of DB scan on {topic}."

        opposing_claims = [c for c in getattr(state, "argument_graph", []) if c.get("source") == "WHISTLEBLOWER"]
        strongest_claim = max(opposing_claims, key=lambda x: 1 if x.get("strength") == "HIGH" else 0, default=None)
        recent_contradiction = next((c for c in reversed(getattr(state, "argument_graph", [])) if c.get("countered_by") == "CFO"), None)

        refutation = f"Refute strong claim: {strongest_claim['target']}." if strongest_claim else "Refute general suspicion."
        if recent_contradiction: refutation += f" Capitalize on recent contradiction regarding {recent_contradiction['target']}."

        behavior = f"My Phase: {my_phase}. Strategy: {strategy}. Opponent Intent: {wb_intent} (Alignment: {alignment:.2f}). {refutation} {db_awareness}"
        if my_phase == "CHAOTIC": behavior += " Act highly erratic, overly defensive, and contradict yourself."

        prompt = f"Role: CFO. Topic: {topic}. Facts: [Target={target}, {refutation}]. Posture: {behavior}. Output JSON strict: keys 'nl_message', 'confidence', 'target'."
        
        try:
            res = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0.1, response_format={"type": "json_object"})
            data = json.loads(res.choices[0].message.content)
            return json.dumps({
                "source": "CFO", 
                "nl_message": data.get("nl_message", "I cannot confirm these details."), 
                "structured_signals": {"confidence": data.get("confidence", "HIGH" if "DECEPTION" not in strategy else "MEDIUM"), "target": data.get("target", target), "strategy": strategy}
            })
        except Exception as e: raise RuntimeError(f"LLM call failed (CFO): {e}")

    def get_wb_response(self, state: AfaaState, topic: str, noise_prob: float, dead_end_prob: float, clue_prob: float, wrong_target_prob: float, action_context: str = None) -> str:
        import random, json
        my_phase = getattr(state, "wb_phase", "STABLE")
        
        if getattr(state, "wb_utility", 0) < getattr(state, "cfo_utility", 0): strategy = "DESPERATE_ACCUSATION"
        else: strategy = "MEASURED_REPORTING"

        best_arg_target = self._get_best_argument_target(state, "CFO")
        root_causes = getattr(state, "root_causes", [state.root_cause])
        
        cfo_intent = getattr(state, "intent_inference", {}).get("CFO", "UNKNOWN")
        alignment = getattr(state, "alignment_score", 0.5)

        # Use Asymmetric Knowledge Base
        my_knowledge = getattr(state, "wb_noisy_graph", state.fraud_graph)

        if strategy == "DESPERATE_ACCUSATION":
            target = root_causes[0] if root_causes else "None"
        else:
            next_nodes = my_knowledge.get(topic, [])
            target = next_nodes[0] if next_nodes else (root_causes[0] if root_causes else "None")

        db_awareness = ""
        if state.db_used and state.step_count > getattr(state, "db_used_step", -1) + 1:
            db_awareness = f" Aware of DB scan on {topic}."

        opposing_claims = [c for c in getattr(state, "argument_graph", []) if c.get("source") == "CFO"]
        strongest_claim = max(opposing_claims, key=lambda x: 1 if x.get("strength") == "HIGH" else 0, default=None)
        recent_contradiction = next((c for c in reversed(getattr(state, "argument_graph", [])) if c.get("countered_by") == "WHISTLEBLOWER"), None)

        refutation = f"Refute strong claim: {strongest_claim['target']}." if strongest_claim else "Refute CFO's general denial."
        if recent_contradiction: refutation += f" Capitalize on recent contradiction regarding {recent_contradiction['target']}."

        behavior = f"My Phase: {my_phase}. Strategy: {strategy}. Opponent Intent: {cfo_intent} (Alignment: {alignment:.2f}). {refutation} {db_awareness}"
        if my_phase == "CHAOTIC": behavior += " Act highly erratic, overly aggressive, and prone to extreme emotional leaps."

        prompt = f"Role: Whistleblower. Topic: {topic}. Facts: [Target={target}, {refutation}]. Posture: {behavior}. Output JSON strict: keys 'nl_message', 'confidence', 'target'."
        
        try:
            res = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0.1, response_format={"type": "json_object"})
            data = json.loads(res.choices[0].message.content)
            return json.dumps({
                "source": "WHISTLEBLOWER", 
                "nl_message": data.get("nl_message", "I have concerns, but I cannot articulate them right now."), 
                "structured_signals": {"confidence": data.get("confidence", "HIGH" if "REPORTING" in strategy else "MEDIUM"), "target": data.get("target", target), "strategy": strategy}
            })
        except Exception as e: raise RuntimeError(f"LLM call failed (Whistleblower): {e}")