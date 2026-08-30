import numpy as np

def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as X.
    """
    result = np.asarray(X, dtype=float).copy()

    if strategy not in ("mean", "median"):
        raise ValueError("strategy must be 'mean' or 'median'")

    if result.ndim == 1:
        valid = result[~np.isnan(result)]

        if len(valid) == 0:
            fill_value = 0.0
        elif strategy == "mean":
            fill_value = np.mean(valid)
        else:
            fill_value = np.median(valid)

        result[np.isnan(result)] = fill_value

    else:
        for j in range(result.shape[1]):
            column = result[:, j]
            valid = column[~np.isnan(column)]

            if len(valid) == 0:
                fill_value = 0.0
            elif strategy == "mean":
                fill_value = np.mean(valid)
            else:
                fill_value = np.median(valid)

            column[np.isnan(column)] = fill_value

    return result