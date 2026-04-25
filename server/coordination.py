import random

try:
    from ..models import AfaaState, AfaaActionType
except (ImportError, ValueError):
    from models import AfaaState, AfaaActionType

class CoordinationEngine:
    """
    Manages multi-agent coordination modes (INDEPENDENT, PANIC, COLLUSION).
    Enforces the 'No Full Deception Lock' safety invariant.
    """
    
    @staticmethod
    def update_mode(state: AfaaState, action_type: AfaaActionType, dept: str) -> bool:
        if not state.config.enable_coordination: return False
        
        old_strategy = state.coordination_strategy
        
        # 🛠️ FLAW 4: Coordination derived purely from interaction history
        total_interactions = state.conflict_score + state.agreement_count
        
        if total_interactions > 0:
            agreement_ratio = state.agreement_count / total_interactions
            
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
    def apply_posture_overrides(state: AfaaState, npc_role: str, base_posture: str) -> str:
        mode = state.coordination_mode
        if mode == "INDEPENDENT": return base_posture
            
        if mode == "PANIC":
            if npc_role == "CFO": return "Extremely hostile, hyper-defensive, threatening the auditor's credentials."
            elif npc_role == "WHISTLEBLOWER": return "Frantic, leaking highly specific but disorganized internal logs."
                
        if mode == "COLLUSION":
            if npc_role == "CFO": return "Highly confident but completely deceptive. Pointing explicitly to a known clean department as the root cause to trap the auditor."
            elif npc_role == "WHISTLEBLOWER": return "Intentionally vague and noisy. Refusing to confirm or deny the CFO's claims, providing contradictory noise instead of a clear signal."
                
        return base_posture