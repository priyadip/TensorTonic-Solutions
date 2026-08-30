import numpy as np

def kfold_split(N: int, k: int, shuffle: bool = True, seed: int = 0) -> list:
    """
    Returns a list of dictionaries with train_idx and val_idx.
    """
    indices = np.arange(N)

    if shuffle:
        indices = np.random.default_rng(seed).permutation(indices)

    # First N % k folds get one extra element
    fold_sizes = np.full(k, N // k, dtype=int)
    fold_sizes[:N % k] += 1

    folds = []
    start = 0

    for size in fold_sizes:
        folds.append(indices[start:start + size])
        start += size

    result = []

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate(folds[:i] + folds[i + 1:])

        result.append({
            "train_idx": train_idx.astype(int),
            "val_idx": val_idx.astype(int)
        })

    return result