import numpy as np

def stratified_split(X: list, y: list, test_size: float = 0.2, seed: int = 42) -> dict:
    """
    Returns a dictionary with X_train, X_test, y_train, and y_test.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    rng = np.random.default_rng(seed)

    test_indices = []

    # Find unique classes and split each class independently
    for cls in np.unique(y):
        class_indices = np.flatnonzero(y == cls)
        class_indices = rng.permutation(class_indices)

        n = len(class_indices)
        n_test = round(n * test_size)

        # Keep at least one sample in training when possible
        if n > 1:
            n_test = min(n_test, n - 1)

        test_indices.extend(class_indices[:n_test])

    test_indices = np.sort(np.asarray(test_indices, dtype=int))

    # Everything not selected for test goes to training
    all_indices = np.arange(len(y))
    train_indices = np.setdiff1d(all_indices, test_indices)
    train_indices = np.sort(train_indices)

    return {
        "X_train": X[train_indices],
        "X_test": X[test_indices],
        "y_train": y[train_indices],
        "y_test": y[test_indices],
    }