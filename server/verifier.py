# server/verifier.py

def verify_submission(state, submitted_dept: str) -> dict:
    """
    Deterministically evaluates the agent's submission against the ground truth.
    """
    correct_root = submitted_dept in state.root_causes
    
    # Calculate the full valid path space
    total_valid_nodes = set()
    for n, targets in state.fraud_graph.items():
        if targets: total_valid_nodes.add(n)
        for t in targets: total_valid_nodes.add(t)
            
    visited_correct = set(state.discovered_nodes).intersection(total_valid_nodes)
    partial_progress = len(visited_correct) / max(1, len(total_valid_nodes))
    missed_critical = len(total_valid_nodes) - len(visited_correct)
    
    # A chain is considered 'correct' if the root is found and >50% of the path was explicitly discovered
    correct_chain = correct_root and partial_progress > 0.5

    return {
        "correct_root": correct_root,
        "correct_chain": correct_chain,
        "partial_progress": partial_progress,
        "visited_correct_nodes": len(visited_correct),
        "missed_critical_nodes": missed_critical
    }