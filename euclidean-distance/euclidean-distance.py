import numpy as np

def euclidean_distance(x: list, y: list) -> float:
    """
    Returns the Euclidean distance as a Python float.
    """
    # Write code here
    X = np.asarray(x)
    Y = np.asarray(y)
    return np.linalg.norm(X - Y)