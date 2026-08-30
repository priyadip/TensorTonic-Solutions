import numpy as np

def calculate_eigenvalues(matrix: list) -> np.ndarray:
    """
    Returns a sorted NumPy array of real eigenvalues.
    """
    matrix = np.asarray(matrix, dtype=float)

    eigenvalues = np.linalg.eigvals(matrix)

    # Eigenvalues are guaranteed to be real; discard tiny numerical
    # imaginary components that can arise from floating-point arithmetic.
    

    return np.sort(eigenvalues).astype(float)