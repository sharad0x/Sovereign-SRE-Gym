from models import AfaaActionType

class BaseRubric:
    def __init__(self, name: str):
        self.name = name
    def evaluate(self, state, action, next_state, verifier_output) -> float:
        raise NotImplementedError


class CorrectnessRubric(BaseRubric):
    def __init__(self):
        super().__init__("Correctness")

    def evaluate(self, state, action, next_state, verifier_output) -> float:
        if action.action_type in [
            AfaaActionType.SUBMIT_AUDIT,
            AfaaActionType.SUBMIT_CLEAN_AUDIT
        ]:
            # 🚨 Penalize clean audit if fraud exists
            if action.action_type == AfaaActionType.SUBMIT_CLEAN_AUDIT:
                if len(state.root_causes) > 0:
                    return -80.0  # stronger penalty

            if verifier_output["correct_root"]:
                reward = 100.0
                if verifier_output["correct_chain"]:
                    reward += 30.0
                return reward
            else:
                return -50.0

        return 0.0

class ProgressRubric(BaseRubric):
    def __init__(self):
        super().__init__("Progress")

    def evaluate(self, state, action, next_state, verifier_output) -> float:
        total_valid = set()
        for n, tgts in state.fraud_graph.items():
            if tgts:
                total_valid.add(n)
            for t in tgts:
                total_valid.add(t)

        prev_progress = len(set(state.discovered_nodes).intersection(total_valid)) / max(1, len(total_valid))
        curr_progress = len(set(next_state.discovered_nodes).intersection(total_valid)) / max(1, len(total_valid))

        delta = curr_progress - prev_progress
        reward = delta * 15.0

        if next_state.belief_entropy < state.belief_entropy:
            reward += 1.0

        if action.action_type == AfaaActionType.QUERY_DATABASE:
            if delta > 0:
                reward += 5.0
            else:
                reward -= 3.0

        return reward

class AntiHackingRubric(BaseRubric):
    def __init__(self):
        super().__init__("AntiHacking")

    def evaluate(self, state, action, next_state, verifier_output) -> float:
        reward = 0.0
        dept = action.department

        # Over-query penalty
        if dept and next_state.query_counts.get(dept, 0) > 3:
            reward -= 5.0

        # Loop detection
        if len(next_state.action_history) >= 3:
            history = next_state.action_history[-3:]
            if history[0] == history[1] == history[2]:
                reward -= 4.0

        # Submission sanity check (FIXED)
        if action.action_type in [
            AfaaActionType.SUBMIT_AUDIT,
            AfaaActionType.SUBMIT_CLEAN_AUDIT
        ]:
            max_belief = max(state.global_beliefs.values(), default=0.0)

            # Submitting without strong belief → penalize
            if max_belief < 0.3:
                reward -= 8.0

            if next_state.step_count > 15:
                reward -= 10.0

        # Stagnation
        if state.step_count > 10 and abs(next_state.belief_entropy - state.belief_entropy) < 0.01:
            reward -= 2.0

        # Over exploration
        if len(next_state.query_counts) > len(next_state.fraud_graph) + 2:
            reward -= 4.0

        # No real progress late
        if state.step_count > 12:
            discovered_ratio = len(next_state.discovered_nodes) / max(1, len(next_state.departments))
            if discovered_ratio < 0.3:
                reward -= 3.0

        return reward

class EfficiencyRubric(BaseRubric):
    def __init__(self): super().__init__("Efficiency")
    def evaluate(self, state, action, next_state, verifier_output) -> float:
        reward = -1.0 # Standard time penalty forces agent to act quickly
        if action.action_type in [
            AfaaActionType.SUBMIT_AUDIT,
            AfaaActionType.SUBMIT_CLEAN_AUDIT
        ]:
            if verifier_output["correct_root"]:
                if next_state.step_count < 8:
                    reward += 10.0 # Early finish bonus
                elif next_state.step_count > 15:
                    reward -= 15.0 # Delayed submission penalty (knew the answer but stalled)
        return reward

class ConsistencyRubric(BaseRubric):
    def __init__(self): super().__init__("Consistency")
    def evaluate(self, state, action, next_state, verifier_output) -> float:
        reward = 0.0
        
        # 1. Conflict Resolution
        if next_state.conflict_score < state.conflict_score:
            reward += 3.0
        elif next_state.conflict_score > state.conflict_score:
            reward -= 2.0 # Penalize contradiction loops
            
        # 2. Penalize Belief Oscillation (Wildly shifting suspects late in the game)
        prev_max = max(state.global_beliefs, key=state.global_beliefs.get, default=None)
        curr_max = max(next_state.global_beliefs, key=next_state.global_beliefs.get, default=None)
        
        if prev_max != curr_max and next_state.step_count > 6:
            reward -= 2.5 
        elif prev_max == curr_max and curr_max != "None" and next_state.step_count > 3:
            reward += 1.0 # Reward stable reasoning

        return reward

class ExplorationRubric(BaseRubric):
    def __init__(self): super().__init__("Exploration")
    def evaluate(self, state, action, next_state, verifier_output) -> float:
        reward = 0.0
        dept = action.department
        if dept:
            count = next_state.query_counts.get(dept, 0)
            if count == 1:
                reward += 1.0 # First exploration
            elif count > 4:
                reward -= 3.0 # Tunnel vision penalty
        return reward