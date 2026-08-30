import numpy as np

def naive_bayes_bernoulli(X_train: list, y_train: list, X_test: list) -> np.ndarray:
    """
    Returns a NumPy array of unnormalized log posteriors.
    Columns are ordered by ascending class label.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    classes = np.sort(np.unique(y_train))
    n_train = len(y_train)

    log_posteriors = []

    for c in classes:
        X_c = X_train[y_train == c]
        Nc = len(X_c)

        # Class prior
        log_prior = np.log(Nc / n_train)

        # Laplace-smoothed probability of feature being 1
        theta = (np.sum(X_c, axis=0) + 1) / (Nc + 2)

        # Log likelihood for every test sample
        log_likelihood = (
            X_test * np.log(theta)
            + (1 - X_test) * np.log(1 - theta)
        ).sum(axis=1)

        log_posteriors.append(log_prior + log_likelihood)

    # Shape: (n_test, n_classes)
    result = np.column_stack(log_posteriors)

    return np.round(result, 4)