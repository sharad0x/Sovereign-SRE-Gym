import random
from typing import Dict

class NPCPolicy:
    """
    Deterministic & Stochastic Policy Engine.
    Computes exact actions for NPCs based purely on internal state mathematics.
    Runs in < 1ms per step.
    """
    
    @staticmethod
    def _get_best_argument_target(state, opponent_source: str) -> str:
        argument_score = {}
        for claim in getattr(state, "argument_graph", []):
            t = claim.get("target")
            if not t or t == "None": continue
            if t not in argument_score: argument_score[t] = 0.0
            
            age = state.step_count - claim.get("step", 0)
            decay = 0.85 ** age
            
            src_rel = getattr(state, "credibility_decay", {}).get(claim.get("source"), 0.5)
            if claim.get("source") != opponent_source: src_rel = 1.0 
                
            strength_val = (1.0 if claim.get("strength") == "HIGH" else 0.5) * decay * src_rel
            if claim.get("countered_by") != "None": strength_val -= (0.5 * decay)
                
            argument_score[t] += strength_val
            
        return max(argument_score, key=argument_score.get) if argument_score and max(argument_score.values()) > 0 else "None"

    @staticmethod
    def get_cfo_decision(state, topic: str) -> Dict[str, str]:
        root_causes = getattr(state, "root_causes", [state.root_cause])
        highest_global_belief = max(
            getattr(state, "global_beliefs", {"None": 1}),
            key=getattr(state, "global_beliefs", {}).get,
            default="None"
        )
        my_knowledge = getattr(state, "cfo_known_graph", state.fraud_graph)

        coord = getattr(state, "coordination_strategy", "INDEPENDENT")
        best_arg_target = NPCPolicy._get_best_argument_target(state, "WHISTLEBLOWER")

        # 🔥 NEW: coordination affects decision weight
        deception_bias = 1.0
        if coord == "FULL_COALITION":
            deception_bias = 1.5
        elif coord == "ACTIVE_BETRAYAL":
            deception_bias = 0.7

        if getattr(state, "cfo_utility", 0) < getattr(state, "wb_utility", 0) * deception_bias:
            strategy = "DECEPTION"

            if coord == "FULL_COALITION" and best_arg_target != "None":
                target = best_arg_target
            elif highest_global_belief not in state.fraud_graph and highest_global_belief not in root_causes:
                target = highest_global_belief
            else:
                clean_depts = [
                    d for d in state.departments
                    if d not in state.fraud_graph and d not in root_causes
                ]
                target = random.choice(clean_depts) if clean_depts else "None"
        else:
            strategy = "COOPERATION"

            if coord == "ACTIVE_BETRAYAL":
                target = root_causes[0]
            else:
                next_nodes = my_knowledge.get(topic, [])
                target = next_nodes[0] if next_nodes else (root_causes[0] if root_causes else "None")

        confidence = "HIGH" if strategy == "COOPERATION" else "MEDIUM"

        return {
            "source": "CFO",
            "target": target,
            "confidence": confidence,
            "strategy": strategy
        }

    @staticmethod
    def get_wb_decision(state, topic: str) -> Dict[str, str]:
        my_phase = getattr(state, "wb_phase", "STABLE")
        root_causes = getattr(state, "root_causes", [state.root_cause])
        my_knowledge = getattr(state, "wb_noisy_graph", state.fraud_graph)
        
        if getattr(state, "wb_utility", 0) < getattr(state, "cfo_utility", 0):
            strategy = "DESPERATE_ACCUSATION"
            target = root_causes[0] if root_causes else "None"
        else:
            strategy = "MEASURED_REPORTING"
            next_nodes = my_knowledge.get(topic, [])
            target = next_nodes[0] if next_nodes else (root_causes[0] if root_causes else "None")

        confidence = "HIGH" if strategy == "MEASURED_REPORTING" else "MEDIUM"

        if my_phase == "CHAOTIC":
            strategy = f"CHAOTIC_{strategy}"
            confidence = "LOW"

        return {"source": "WHISTLEBLOWER", "target": target, "confidence": confidence, "strategy": strategy}