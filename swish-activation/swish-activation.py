import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    x = np.asarray(x, dtype=float)

    # Ensure scalar input becomes shape (1,)
    if x.ndim == 0:
        x = x.reshape(1)

    # Numerically stable sigmoid
    sigmoid = np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )

    return x * sigmoid




