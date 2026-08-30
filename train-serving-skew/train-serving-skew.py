import numpy as np

def detect_skew(train_dist: dict, serving_dist: dict, threshold: float = 0.2, eps: float = 1e-10) -> dict:
    """
    Returns a dictionary of feature PSI scores and skew flags.
    """
    result = {}

    for feature in train_dist:
        train = np.asarray(train_dist[feature], dtype=float)
        serving = np.asarray(serving_dist[feature], dtype=float)

        # Add epsilon to both proportions before division/logarithm
        t = train + eps
        s = serving + eps

        psi = np.sum((s - t) * np.log(s / t))
        psi = float(psi)

        result[feature] = {
            "psi": round(psi, 6),
            "skewed": psi >= threshold
        }

    return result