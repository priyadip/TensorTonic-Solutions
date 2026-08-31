import numpy as np

def priority_replay_sample(priorities: list, alpha: float, beta: float) -> list:
    """
    Returns sampling probabilities and normalized importance weights.
    """

    priorities = np.asarray(priorities, dtype=float)

    # 1. Powered priorities
    powered = priorities ** alpha

    # 2. Sampling probabilities
    probabilities = powered / np.sum(powered)

    # 3. Raw importance sampling weights
    N = len(priorities)
    weights = (N * probabilities) ** (-beta)

    # 4. Normalize weights
    weights = weights / np.max(weights)

    return [probabilities.tolist(), weights.tolist()]
