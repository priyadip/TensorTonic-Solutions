import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.asarray(x, dtype=float)

    # Ensure scalar input becomes shape (1,)
    if x.ndim == 0:
        x = x.reshape(1)

    return np.tanh(x)
