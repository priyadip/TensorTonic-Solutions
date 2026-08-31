import numpy as np

def streaming_minmax(D: int, batches: list, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with normalized_batches, min, and max.
    """
    # Initialize running minimum and maximum
    running_min = np.full(D, np.inf, dtype=float)
    running_max = np.full(D, -np.inf, dtype=float)

    normalized_batches = []

    for batch in batches:
        batch = np.asarray(batch, dtype=float)

        # Update min and max BEFORE normalizing
        running_min = np.minimum(running_min, np.min(batch, axis=0))
        running_max = np.maximum(running_max, np.max(batch, axis=0))

        # Calculate range
        range_ = running_max - running_min

        # Normalize safely
        normalized = np.zeros_like(batch, dtype=float)
        np.divide(
            batch - running_min,
            np.maximum(range_, eps),
            out=normalized
        )

        normalized_batches.append(normalized)

    return {
        "normalized_batches": normalized_batches,
        "min": running_min,
        "max": running_max
    }