import numpy as np

def mean_squared_error(y_pred: list, y_true: list) -> float:
    """Returns the error as a float."""
    yp = np.asarray(y_pred)
    yt = np.asarray(y_true)

    return float(np.mean((yp - yt)**2))