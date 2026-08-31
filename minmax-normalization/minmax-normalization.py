import numpy as np

def minmax_scale(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns a floating-point NumPy array matching the shape of X.
    """
    X = np.asarray(X, dtype=float)

    xmin = np.min(X, axis=axis, keepdims=True)
    xmax = np.max(X, axis=axis, keepdims=True)
    range_ = xmax - xmin

    # Avoid division by zero for near-constant slices
    scaled = np.zeros_like(X, dtype=float)
    np.divide(X - xmin, range_, out=scaled, where=range_ > eps)

    return scaled