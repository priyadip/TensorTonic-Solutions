import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns population Z-scores as a NumPy array matching the shape of X.
    """
    X = np.asarray(X, dtype=float)

    # Calculate mean and population standard deviation
    mu = np.mean(X, axis=axis, keepdims=True)
    sigma = np.std(X, axis=axis, keepdims=True)

    # Start with zeros
    z = np.zeros_like(X, dtype=float)

    # Divide only where standard deviation is greater than eps
    np.divide(X - mu, sigma, out=z, where=sigma > eps)

    return z