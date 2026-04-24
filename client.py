# client.py
from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from models import AfaaAction, AfaaObservation, AfaaState

class AfaaEnvClient(EnvClient[AfaaAction, AfaaObservation, AfaaState]):
    def __init__(self, base_url: str, **kwargs):
        kwargs.setdefault("message_timeout_s", 60.0)
        super().__init__(base_url=base_url, **kwargs)

    def _step_payload(self, action: AfaaAction) -> Dict:
        return action.model_dump(mode='json')

    def _parse_result(self, payload: Dict) -> StepResult[AfaaObservation]:
        # OpenEnv wrapper looks like: {"observation": {...}, "reward": 0.0, "done": False}
        obs_data = payload.get("observation", {})
        
        # 🛠️ INJECT MISSING FIELDS: 
        # OpenEnv stripped 'done' and 'reward' from the inner dict. Put them back for Pydantic.
        obs_data["done"] = payload.get("done", False)
        obs_data["reward"] = payload.get("reward", 0.0)
        
        return StepResult(
            observation=AfaaObservation.model_validate(obs_data),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> AfaaState:
        return AfaaState.model_validate(payload)