import numpy as np

def manhattan_distance(x: list, y: list) -> float:
    """
    Returns the Manhattan distance as a Python float.
    """
    # Write code here
    X = np.asarray(x)
    Y = np.asarray(y)

    return np.sum(np.abs(X - Y)).astype(float)