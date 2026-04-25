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

class AfaaObservation(BaseModel):
    budget_remaining: int = Field(description="Remaining budget.")
    available_departments: List[str] = Field(description="List of departments under audit.")
    latest_text: str = Field(description="Semantic output from the last action.")
    rule_violations: List[str] = Field(default_factory=list, description="Warnings for bad actions.")
    done: bool = Field(description="True if the episode has ended.")
    reward: float = Field(default=0.0, description="Reward for the current step.")

class AfaaConfig(BaseModel):
    enable_dynamic_chain: bool = Field(default=False, description="Phase III: Enable dynamic topology shifts.")
    enable_coordination: bool = Field(default=False, description="Phase II: Enable NPC coordination and deception modes.")
    enable_memory: bool = Field(default=False, description="Phase IV: Enable cross-episode global memory.")
    debug_mode: bool = Field(default=False, description="Enable verbose internal state logging for debugging.")

class AfaaState(BaseModel):
    episode_id: Optional[str] = None
    step_count: int = 0
    budget: int = 40
    departments: List[str] = Field(default_factory=list)
    
    discovered_nodes: List[str] = Field(default_factory=list, description="Tracks valid nodes explored for sub-goal rewards.")
    
    wb_query_count: int = 0
    db_used: bool = False
    db_used_step: int = Field(default=-1, description="The step at which the database was queried (for delayed awareness).")
    action_history: List[str] = Field(default_factory=list)
    query_counts: Dict[str, int] = Field(default_factory=dict, description="Explicit tracking of queries per department.")
    
    wb_is_accurate: bool = True
    cfo_base_evasive: bool = False

    # 🛠️ FLAW 7: Complex Graph Topology (Multiple Roots)
    fraud_graph: Dict[str, List[str]] = Field(default_factory=dict, description="DAG of fraud with cross-links.") 
    root_causes: List[str] = Field(default_factory=list, description="Multiple possible terminal nodes.")
    
    # 🛠️ FLAW 4: Belief Normalization State
    agent_beliefs: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Normalized P(target | evidence).")

    # 🛠️ FLAW 1: Phase Transitions & Unbounded Emergence
    env_phase: str = Field(default="STABLE", description="STABLE, CHAOTIC, or RECOVERY")
    
    # 🛠️ FLAW 2: Global Belief Model
    global_beliefs: Dict[str, float] = Field(default_factory=dict, description="Aggregated belief across all sources weighted by reliability.")
    
    # 🛠️ FLAW 4: Interaction-driven Coordination 
    agreement_count: int = Field(default=0, description="Tracks how many times agents have mutually agreed.")

    # 🛠️ FLAW 1 & 6: Probabilistic & Per-Agent Instability
    cfo_phase: str = Field(default="STABLE", description="STABLE, CHAOTIC, or RECOVERY")
    wb_phase: str = Field(default="STABLE", description="STABLE, CHAOTIC, or RECOVERY")
    belief_entropy: float = Field(default=0.0, description="Information theory measure of auditor confusion.")
    disagreement_rate: float = Field(default=0.0, description="Rolling average of NPC contradictions.")

    # 🛠️ FLAW 3: True Coordination & Intent Inference
    alignment_score: float = Field(default=0.5, description="Continuous measure of how often NPCs target the same nodes.")
    intent_inference: Dict[str, str] = Field(default_factory=lambda: {"CFO": "UNKNOWN", "WHISTLEBLOWER": "UNKNOWN"})

    # 🛠️ FLAW 4: Long-Term Argument Memory
    credibility_decay: Dict[str, float] = Field(default_factory=lambda: {"CFO": 1.0, "WHISTLEBLOWER": 1.0})
    repeated_lies_penalty: Dict[str, int] = Field(default_factory=lambda: {"CFO": 0, "WHISTLEBLOWER": 0})

    # 🛠️ FLAW 7: Continuous Mutation
    base_mutation_prob: float = Field(default=0.25)
    mutation_decay_factor: float = Field(default=1.0, description="Drops after each mutation to prevent immediate re-mutation, but recovers.")

    difficulty_level: int = Field(default=1, description="Current curriculum difficulty level (1-5).")
    mutation_prob: float = Field(default=0.25, description="Dynamic probability of topology mutation.")
    noise_prob: float = Field(default=0.30, description="Dynamic probability of cross-chain noise.")
    dead_end_prob: float = Field(default=0.25, description="Dynamic probability of dead-end deflections.")
    clue_prob: float = Field(default=0.70, description="Dynamic probability of intermediary propagating clue.")
    wrong_target_prob: float = Field(default=0.20, description="Dynamic probability of WB pointing to wrong target.")
    

    # 🛠️ STRATEGIC GAME UPGRADE 
    last_speaker: str = Field(default="NONE", description="Tracks the last NPC to speak for debate loops.")
    db_truth: Dict[str, str] = Field(default_factory=dict, description="Stores deterministic DB results for NPC awareness.")
    cfo_hostility: float = Field(default=0.0, description="Persistent hostility level of the CFO (0.0 to 1.0).")
    cfo_utility: float = Field(default=0.0, description="CFO's strategic score: success in avoiding detection.")
    wb_utility: float = Field(default=0.0, description="WB's strategic score: success in exposing truth.")

    shift_count: int = Field(default=0, description="Tracks how many times the chain has mutated.")
    last_mutation_step: int = Field(default=-1, description="Records the step_count of the last mutation.")
    post_mutation_instability_steps: int = Field(default=0, description="Window of instability following a state shift.")
    outdated_nodes: List[str] = Field(default_factory=list, description="Nodes removed from the chain during a mutation.")
    
    # 🛠️ ADVANCED COORDINATION & STRATEGY
    strategy_phase: str = Field(default="OBSERVE", description="OBSERVE (steps<5), DECEIVE (high conflict), COMMIT (else).")
    coordination_strategy: str = Field(default="INDEPENDENT", description="COALITION, BETRAYAL, or INDEPENDENT.")

    claim_history: List[Dict[str, str]] = Field(default_factory=list, description="Shared memory of claims: {'source': str, 'dept': str, 'target': str}")
    contradiction_flag: bool = Field(default=False, description="True if a direct contradiction was detected between CFO and WB.")
    conflict_score: int = Field(default=0, description="Escalates as NPCs contradict each other.")
    inter_agent_tension: float = Field(default=0.0, description="Measures the hostility between the CFO and Whistleblower.")
    
    # 🛠️ FLAW 6: Information Asymmetry
    cfo_known_graph: Dict[str, List[str]] = Field(default_factory=dict, description="CFO knows exact paths, but maybe not all of them.")
    wb_noisy_graph: Dict[str, List[str]] = Field(default_factory=dict, description="WB knows paths but with inherent hallucinations/noise.")

    # 🛠️ FLAW 4: Irreversible Trust Damage
    credibility_floor: Dict[str, float] = Field(default_factory=lambda: {"CFO": 0.1, "WHISTLEBLOWER": 0.1})
    irreversible_damage: Dict[str, bool] = Field(default_factory=lambda: {"CFO": False, "WHISTLEBLOWER": False})

    # 🛠️ FLAW 3 & 7: Coordination & Equilibrium
    coalition_strength: float = Field(default=0.0, description="Continuous strength of NPC alignment.")
    equilibrium_state: str = Field(default="STABLE", description="STABLE, OSCILLATING, or COLLAPSED")
    
    counter_claims: Dict[str, str] = Field(default_factory=lambda: {"CFO": "None", "WB": "None"}, description="Stores the specific target being argued against.")
    narrative_dominance: str = Field(default="NEUTRAL", description="Tracks which NPC's narrative is currently most influential.")
    internal_npc_trust: float = Field(default=0.8, description="Trust between NPCs. Low trust leads to internal sabotage.")
    sabotage_mode: bool = Field(default=False, description="When true, NPCs prioritize framing each other over avoiding the auditor.")

    npc_goals: Dict[str, str] = Field(
        default_factory=lambda: {"CFO": "minimize_detection_and_protect_chain", "WHISTLEBLOWER": "expose_truth_but_avoid_retaliation"},
        description="The internal incentives driving NPC responses."
    )
    trust_scores: Dict[str, float] = Field(
        default_factory=lambda: {"CFO": 0.5, "WHISTLEBLOWER": 0.5},
        description="Tracks the auditor's rapport with NPCs."
    )

    # 🛠️ EMERGENT INTELLIGENCE
    argument_graph: List[Dict[str, Any]] = Field(default_factory=list, description="Claim-Counter-Evidence mapping.")
    belief_about_other: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {"CFO": {"reliability": 0.8, "history": []}, "WHISTLEBLOWER": {"reliability": 0.5, "history": []}},
        description="Probabilistic tracking of the other agent's reliability."
    )

    config: AfaaConfig = Field(default_factory=AfaaConfig, description="Feature flags for curriculum training.")