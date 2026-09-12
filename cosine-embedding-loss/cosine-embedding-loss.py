import numpy as np
def cosine_embedding_loss(x1: list, x2: list, label: int, margin: float) -> float:
    """Returns the cosine embedding loss as a float."""

    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    
    # Cosine similarity
    cos_sim = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    
    # Loss computation
    if label == 1:
        return float(1 - cos_sim)
    else:  # label == -1
        return float(max(0.0, cos_sim - margin))

    


