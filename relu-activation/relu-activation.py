import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x = np.asarray(x, dtype=float)

    # Ensure scalar input becomes shape (1,)
    if x.ndim == 0:
        x = x.reshape(1)

    return np.maximum(0, x)
