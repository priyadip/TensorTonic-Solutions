import numpy as np

def impute_missing(X, strategy="mean"):
    # Convert to numpy float array copy
    X = np.array(X, dtype=float, copy=True)

    # Handle 1D separately
    if X.ndim == 1:
        mask = np.isnan(X)
        if np.any(~mask):
            val = np.mean(X[~mask]) if strategy == "mean" else np.median(X[~mask])
        else:
            val = 0.0
        X[mask] = val
        return X

    # 2D case
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col)

        if np.any(~mask):
            if strategy == "mean":
                val = np.mean(col[~mask])
            else:
                val = np.median(col[~mask])
        else:
            val = 0.0

        col[mask] = val
        X[:, j] = col

    return X