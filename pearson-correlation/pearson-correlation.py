import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    X = np.asarray(X, dtype=float)

    # Center each feature
    X_centered = X - X.mean(axis=0)

    # Sample covariance matrix
    covariance = (X_centered.T @ X_centered) / (X.shape[0] - 1)

    # Sample standard deviations
    std = np.sqrt(np.diag(covariance))

    # Normalize covariance by the outer product of standard deviations
    denominator = np.outer(std, std)
    correlation = np.full_like(covariance, np.nan, dtype=float)

    valid = denominator != 0
    correlation[valid] = covariance[valid] / denominator[valid]

    return correlation