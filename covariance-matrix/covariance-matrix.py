import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    # Write code here
    n = len(X) - 1
    
    X = np.asarray(X)
    miu = np.mean(X, axis = 0)
    Xc = X - miu

    return 1/n*Xc.T@Xc


