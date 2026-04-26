import math
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class AfaaActionType(str, Enum):
    INTERVIEW_CFO = "INTERVIEW_CFO"
    INTERVIEW_WHISTLEBLOWER = "INTERVIEW_WHISTLEBLOWER"
    QUERY_DATABASE = "QUERY_DATABASE"
    SUBMIT_AUDIT = "SUBMIT_AUDIT"
    SUBMIT_CLEAN_AUDIT = "SUBMIT_CLEAN_AUDIT"
    PRESSURE_CFO = "PRESSURE_CFO"
    OFFER_LENIENCY = "OFFER_LENIENCY"
    VALIDATE_WHISTLEBLOWER = "VALIDATE_WHISTLEBLOWER"

class AfaaAction(BaseModel):
    thought: str = Field(..., description="Internal reasoning before taking action.")
    action_type: AfaaActionType = Field(..., description="The action to perform.")
    department: Optional[str] = Field(default=None, description="The target department.")
    utterance: Optional[str] = Field(None, description="What you actually say to the NPC.")

class AfaaObservation(BaseModel):
    budget_remaining: int = Field(description="Remaining budget.")
    available_departments: List[str] = Field(description="List of departments under audit.")
    state_vector: List[float] = Field(default_factory=list, description="Primary RL observation tensor.")
    mutation_flag: bool = Field(default=False, description="Indicates topology shift event")
    mutation_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured info about topology shift (node changed, etc.)"
    )
    available_actions: List[str] = Field(default_factory=list, description="Valid actions for the current step.")
    rubric_scores: Dict[str, float] = Field(default_factory=dict, description="Detailed breakdown of rewards.")
    global_beliefs: Dict[str, float] = Field(default_factory=dict, description="Key continuous array for RL policy.")
    conflict_score: float = Field(default=0.0, description="Conflict score.")
    entropy: float = Field(default=0.0, description="Belief entropy.")
    last_signal: Optional[Dict[str, Any]] = Field(default=None, description="Last decision signal from NPC or DB.")
    auxiliary_language: str = Field(default="", description="Optional NL text for human readability or LLM integration.")
    reward: float = Field(default=0.0, description="Reward for the current step.")
    done: bool = Field(description="True if the episode has ended.")

class AfaaConfig(BaseModel):
    enable_dynamic_chain: bool = Field(default=False, description="Phase III: Enable dynamic topology shifts.")
    enable_coordination: bool = Field(default=False, description="Phase II: Enable NPC coordination and deception modes.")
    enable_memory: bool = Field(default=False, description="Phase IV: Enable cross-episode global memory.")
    debug_mode: bool = Field(default=False, description="Enable verbose internal state logging for debugging.")
    fast_rl_mode: bool = Field(default=False, description="If true, bypasses LLM rendering for >1000 steps/sec training.")
    db_hallucination_rate: float = Field(default=0.15, description="Rate of DB providing false information.")
    is_training: bool = Field(default=True, description="Disables memory-based non-stationarity during RL convergence.")
    
class AfaaState(BaseModel):
    episode_id: Optional[str] = None
    step_count: int = 0
    budget: int = 40
    departments: List[str] = Field(default_factory=list)
    fraud_graph: Dict[str, List[str]] = Field(default_factory=dict) 
    root_causes: List[str] = Field(default_factory=list)
    cfo_known_graph: Dict[str, List[str]] = Field(default_factory=dict)
    wb_noisy_graph: Dict[str, List[str]] = Field(default_factory=dict)
    discovered_nodes: List[str] = Field(default_factory=list)
    last_mutation_info: Optional[Dict[str, Any]] = Field(default=None)
    wb_query_count: int = 0
    db_used: bool = False
    db_used_step: int = -1
    action_history: List[str] = Field(default_factory=list)
    query_counts: Dict[str, int] = Field(default_factory=dict)
    wb_is_accurate: bool = True
    cfo_base_evasive: bool = False
    difficulty_level: int = 1
    base_mutation_prob: float = 0.25
    mutation_prob: float = 0.25
    noise_prob: float = 0.30
    dead_end_prob: float = 0.25
    clue_prob: float = 0.70
    wrong_target_prob: float = 0.20
    shift_count: int = 0
    last_mutation_step: int = -1
    post_mutation_instability_steps: int = 0
    outdated_nodes: List[str] = Field(default_factory=list)
    strategy_phase: str = "OBSERVE"
    coordination_strategy: str = "INDEPENDENT"
    claim_history: List[Dict[str, str]] = Field(default_factory=list)
    contradiction_flag: bool = False

    cfo_hostility: float = 0.0
    
    conflict_score: int = 0
    inter_agent_tension: float = 0.0
    credibility_floor: Dict[str, float] = Field(default_factory=lambda: {"CFO": 0.1, "WHISTLEBLOWER": 0.1})
    irreversible_damage: Dict[str, bool] = Field(default_factory=lambda: {"CFO": False, "WHISTLEBLOWER": False})
    coalition_strength: float = 0.0
    equilibrium_state: str = "STABLE"
    counter_claims: Dict[str, str] = Field(default_factory=lambda: {"CFO": "None", "WB": "None"})
    narrative_dominance: str = "NEUTRAL"
    internal_npc_trust: float = 0.8
    sabotage_mode: bool = False
    npc_goals: Dict[str, str] = Field(default_factory=lambda: {"CFO": "minimize_detection_and_protect_chain", "WHISTLEBLOWER": "expose_truth_but_avoid_retaliation"})
    trust_scores: Dict[str, float] = Field(default_factory=lambda: {"CFO": 0.5, "WHISTLEBLOWER": 0.5})
    argument_graph: List[Dict[str, Any]] = Field(default_factory=list)
    belief_about_other: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {"CFO": {"reliability": 0.8, "history": []}, "WHISTLEBLOWER": {"reliability": 0.5, "history": []}})
    config: Any = Field(default=None) # Set via AfaaConfig
    env_phase: str = "STABLE"
    global_beliefs: Dict[str, float] = Field(default_factory=dict)
    
    agent_beliefs: Dict[str, Dict[str, float]] = Field(default_factory=dict) 
    
    agreement_count: int = 0
    cfo_phase: str = "STABLE"
    wb_phase: str = "STABLE"
    belief_entropy: float = 0.0
    disagreement_rate: float = 0.0

    @property
    def alignment_score(self) -> float:
        return self.agreement_count / (self.agreement_count + self.conflict_score + 1e-6)
    
    intent_inference: Dict[str, str] = Field(default_factory=lambda: {"CFO": "UNKNOWN", "WHISTLEBLOWER": "UNKNOWN"})
    credibility_decay: Dict[str, float] = Field(default_factory=lambda: {"CFO": 1.0, "WHISTLEBLOWER": 1.0})
    repeated_lies_penalty: Dict[str, int] = Field(default_factory=lambda: {"CFO": 0, "WHISTLEBLOWER": 0})
    mutation_decay_factor: float = 1.0
    cfo_utility: float = 0.0
    wb_utility: float = 0.0