import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    # Dot product
    dot = np.dot(a, b)

    # Euclidean norms
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # If either vector is zero
    if norm_a == 0 or norm_b == 0:
        return 0.0

    # Cosine similarity
    return float(dot / (norm_a * norm_b))
