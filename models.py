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
    budget_remaining: int = Field(description="Remaining budget.")
    available_departments: List[str] = Field(description="List of departments under audit.")
    latest_text: str = Field(description="Semantic output from the last action.")
    rule_violations: List[str] = Field(default_factory=list, description="Warnings for bad actions.")
    done: bool = Field(description="True if the episode has ended.")
    reward: float = Field(default=0.0, description="Reward for the current step.")

class AfaaState(BaseModel):
    episode_id: Optional[str] = None
    step_count: int = 0
    budget: int = 20 # 🛠️ FIXED 6: Scaled budget to 20 for deeper exploration
    departments: List[str] = Field(default_factory=list)
    fraud_chain: List[str] = Field(default_factory=list) 
    archetype: str = "Coordinated Cover-up"
    wb_query_count: int = 0
    db_used: bool = False
    action_history: List[str] = Field(default_factory=list)
    dept_suspicion: Dict[str, float] = Field(default_factory=dict) 
    
    wb_is_accurate: bool = True
    cfo_base_evasive: bool = False