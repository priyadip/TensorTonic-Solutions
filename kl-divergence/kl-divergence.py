import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """Returns the divergence as a float."""
    # Convert to numpy arrays
    p = np.array(p)
    q = np.array(q)
    
    # Numerical stability
    p = p + eps
    q = q + eps
    
    # KL divergence
    return float(np.sum(p * np.log(p / q)))
