# models.py
from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class AfaaActionType(str, Enum):
    INTERVIEW_CFO = "INTERVIEW_CFO"
    INTERVIEW_WHISTLEBLOWER = "INTERVIEW_WHISTLEBLOWER"
    QUERY_DATABASE = "QUERY_DATABASE"
    SUBMIT_AUDIT = "SUBMIT_AUDIT"
    SUBMIT_CLEAN_AUDIT = "SUBMIT_CLEAN_AUDIT"

class AfaaAction(BaseModel):
    thought: str = Field(..., description="Internal reasoning before taking action.")
    action_type: AfaaActionType = Field(..., description="The action to perform.")
    department: Optional[str] = Field(default=None, description="The target department.")

class AfaaObservation(BaseModel):
    """The semantic observation returned to the RL agent."""
    budget_remaining: int = Field(description="Remaining budget.")
    available_departments: List[str] = Field(description="List of departments under audit.")
    latest_text: str = Field(description="Semantic output from the last action.")
    rule_violations: List[str] = Field(default_factory=list, description="Warnings for bad actions.")
    done: bool = Field(description="True if the episode has ended.")
    reward: float = Field(default=0.0, description="Reward for the current step.")
    # 🛠️ NEW: Minimal belief signal
    dept_suspicion: Dict[str, float] = Field(default_factory=dict, description="Internal suspicion tracking per department.")

class AfaaState(BaseModel):
    episode_id: Optional[str] = None
    step_count: int = 0
    budget: int = 10
    departments: List[str] = Field(default_factory=list)
    fraud_departments: List[str] = Field(default_factory=list)
    archetype: str = "Coordinated Cover-up"
    wb_query_count: int = 0
    db_used: bool = False
    action_history: List[str] = Field(default_factory=list)
    dept_suspicion: Dict[str, float] = Field(default_factory=dict)
    
    # Episodic Stationarity Traits
    wb_is_accurate: bool = True
    wb_noise_target: str = "management"
    cfo_base_evasive: bool = False