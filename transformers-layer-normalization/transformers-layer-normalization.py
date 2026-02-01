import numpy as np

def layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Mean over last dimension
    mean = np.mean(x, axis=-1, keepdims=True)

    # Variance over last dimension
    var = np.var(x, axis=-1, keepdims=True)

    # Normalize
    x_hat = (x - mean) / np.sqrt(var + eps)

    # Scale and shift
    return gamma * x_hat + beta
