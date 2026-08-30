import numpy as np

def detect_drift(reference_counts: list, production_counts: list, threshold: float) -> dict:
    """
    Returns a dictionary with score and drift_detected.
    """
    reference = np.asarray(reference_counts, dtype=float)
    production = np.asarray(production_counts, dtype=float)

    # Normalize each histogram independently
    p = reference / reference.sum()
    q = production / production.sum()

    # Total variation distance
    score = 0.5 * np.sum(np.abs(p - q))

    return {
        "score": float(score),
        "drift_detected": bool(score > threshold),
    }