import numpy as np

def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    x = np.asarray(x, dtype=float)

    # Ensure scalar input becomes shape (1,)
    if x.ndim == 0:
        x = x.reshape(1)

    out = np.where(x > 0, x, alpha * (np.exp(x) - 1))
    return out.tolist()

