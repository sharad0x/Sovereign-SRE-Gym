import random

try:
    from ..models import AfaaState, AfaaActionType
except (ImportError, ValueError):
    from models import AfaaState, AfaaActionType

class CoordinationEngine:
    @staticmethod
    def update_mode(state, action_type, dept: str) -> bool:
        if not getattr(state.config, "enable_coordination", False):
            return False

        old_strategy = state.coordination_strategy

        total_interactions = state.conflict_score + state.agreement_count

        if total_interactions == 0:
            state.coordination_strategy = "INDEPENDENT"
        else:
            agreement_ratio = state.agreement_count / (total_interactions + 1e-6)

            if agreement_ratio > 0.7:
                state.coordination_strategy = "FULL_COALITION"
            elif agreement_ratio > 0.4:
                state.coordination_strategy = "PARTIAL_COALITION"
            elif agreement_ratio > 0.2:
                state.coordination_strategy = "SHIFTING_LOYALTY"
            else:
                state.coordination_strategy = "ACTIVE_BETRAYAL"

        return state.coordination_strategy != old_strategy
        
    @staticmethod
    def apply_posture_overrides(state, npc_role: str, base_posture: str) -> str:
        # 🛠️ FIX 5: COORDINATION VARIABLE BUG
        mode = state.coordination_strategy 
        if mode == "INDEPENDENT": return base_posture
            
        if mode == "FULL_COALITION" or mode == "COLLUSION":
            if npc_role == "CFO": return "Highly confident but completely deceptive. Pointing explicitly to a known clean department as the root cause to trap the auditor."
            elif npc_role == "WHISTLEBLOWER": return "Intentionally vague and noisy. Refusing to confirm or deny the CFO's claims, providing contradictory noise instead of a clear signal."
        
        return base_posture