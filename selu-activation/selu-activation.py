import numpy as np

def selu(x):
    """
    Apply SELU activation to each element.
    """
    # Fixed SELU constants
    lambda_ = 1.0507
    alpha = 1.6733

    x = np.asarray(x, dtype=float)

    # Ensure scalar input becomes shape (1,)
    if x.ndim == 0:
        x = x.reshape(1)

    out = np.where(
        x > 0,
        lambda_ * x,
        lambda_ * alpha * (np.exp(x) - 1)
    )

    return out.tolist()
