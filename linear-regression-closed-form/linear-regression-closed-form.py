import numpy as np

def linear_regression_closed_form(X: list, y: list) -> list:
    """
    Returns the optimal weight vector as a list.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    XT = X.T
    XTX = XT @ X
    XTy = XT @ y

    w = np.linalg.inv(XTX) @ XTy

    return w.astype(float).tolist()