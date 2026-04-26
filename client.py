# client.py
from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from models import AfaaAction, AfaaObservation, AfaaState

class AfaaEnvClient(EnvClient[AfaaAction, AfaaObservation, AfaaState]):
    def __init__(self, base_url: str, **kwargs):
        kwargs.setdefault("message_timeout_s", 120.0)
        super().__init__(base_url=base_url, **kwargs)

    def _step_payload(self, action: AfaaAction) -> Dict:
        return action.model_dump(mode='json')

    def _parse_result(self, payload: Dict) -> StepResult[AfaaObservation]:
        obs_data = payload.get("observation", {}).copy()

        # Defensive injection (avoid mutating original payload)
        obs_data.setdefault("done", payload.get("done", False))
        obs_data.setdefault("reward", payload.get("reward", 0.0))

        if "available_actions" not in obs_data:
            raise ValueError("Malformed observation: missing available_actions")

        return StepResult(
            observation=AfaaObservation.model_validate(obs_data),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )
        
        return StepResult(
            observation=AfaaObservation.model_validate(obs_data),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> AfaaState:
        return AfaaState.model_validate(payload)