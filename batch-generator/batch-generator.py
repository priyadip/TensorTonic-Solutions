import numpy as np

def batch_generator(X: list, y: list, batch_size: int, seed: int = 42, drop_last: bool = False):
    """
    Returns a generator of (X_batch, y_batch) tuples.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))

    for start in range(0, len(X), batch_size):
        end = start + batch_size

        if end > len(X) and drop_last:
            break

        batch_indices = indices[start:end]
        yield X[batch_indices], y[batch_indices]