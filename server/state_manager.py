import random
from typing import Tuple

try:
    from ..models import AfaaState
except (ImportError, ValueError):
    from models import AfaaState

class StateManager:
    """
    Manages the dynamic topology shifts (mutations) of the fraud graph.
    Ensures strict RL stability by capping mutations and respecting trigger bounds.
    """
    
    @staticmethod
    def attempt_mutation(state: AfaaState) -> bool:
        if not state.config.enable_dynamic_chain:
            return False

        if state.shift_count >= 1:
            return False

        if state.step_count < 3:
            return False

        state.post_mutation_instability_steps = 2

        mutable_nodes = [node for node, targets in state.fraud_graph.items() if targets]
        if not mutable_nodes:
            return False

        mutate_node = random.choice(mutable_nodes)
        old_targets = state.fraud_graph[mutate_node]

        available_depts = [
            d for d in state.departments
            if d not in old_targets and d != mutate_node and d not in state.root_causes
        ]
        if not available_depts:
            return False

        new_node = random.choice(available_depts)

        if old_targets:
            old_node = old_targets[0]
            state.outdated_nodes.append(old_node)
            state.fraud_graph[mutate_node][0] = new_node

            # 🔥 STORE STRUCTURED MUTATION INFO
            state.last_mutation_info = {
                "event": "STATE_SHIFT",
                "from_node": mutate_node,
                "old_target": old_node,
                "new_target": new_node,
                "step": state.step_count
            }

        state.shift_count += 1
        state.last_mutation_step = state.step_count

        return True