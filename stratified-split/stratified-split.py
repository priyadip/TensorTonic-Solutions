import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    X = np.array(X)
    y = np.array(y)

    train_idx = []
    test_idx = []

    classes = np.unique(y)

    for c in classes:
        idx = np.where(y == c)[0]

        if rng is None:
            np.random.shuffle(idx)
        else:
            rng.shuffle(idx)

        n = len(idx)
        n_test = int(round(n * test_size))

        if n_test >= n:
            n_test = n - 1
        if n_test < 0:
            n_test = 0

        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

   
    train_idx = np.array(sorted(train_idx))
    test_idx = np.array(sorted(test_idx))

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]