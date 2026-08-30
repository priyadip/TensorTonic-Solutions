import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """
    X = np.asarray(X, dtype=float)

    # Center the data
    X_centered = X - X.mean(axis=0)

    # Sample covariance matrix
    covariance = (X_centered.T @ X_centered) / (X.shape[0] - 1)

    # Covariance matrix is symmetric, so use eigh.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Select eigenvectors with the largest eigenvalues
    order = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, order[:k]]

    # Fix the arbitrary sign of eigenvectors for deterministic output:
    # make the largest-magnitude element in each component positive.
    for j in range(k):
        idx = np.argmax(np.abs(components[:, j]))
        if components[idx, j] < 0:
            components[:, j] *= -1

    projected = X_centered @ components

    return projected.tolist()