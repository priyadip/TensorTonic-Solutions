import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Bernoulli Naive Bayes log-posterior.
    Returns numpy array of shape (n_test, n_classes).
    """

    # ---- Convert & enforce 2D ----
    X_train = np.asarray(X_train, dtype=float)
    X_test  = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_train, d = X_train.shape
    n_test = X_test.shape[0]

    classes = np.unique(y_train)
    n_classes = len(classes)

    # ---- Output ----
    log_post = np.zeros((n_test, n_classes), dtype=float)

    # ---- Train ----
    log_priors = np.zeros(n_classes)
    log_p = np.zeros((n_classes, d))
    log_1mp = np.zeros((n_classes, d))

    for idx, c in enumerate(classes):
        Xc = X_train[y_train == c]
        nc = Xc.shape[0]

        # Prior
        log_priors[idx] = np.log(nc / n_train)

        # Laplace-smoothed Bernoulli parameters
        p = (np.sum(Xc, axis=0) + 1.0) / (nc + 2.0)

        log_p[idx] = np.log(p)
        log_1mp[idx] = np.log(1.0 - p)

    # ---- Predict (log-space) ----
    for i in range(n_test):
        x = X_test[i]
        for j in range(n_classes):
            log_post[i, j] = (
                log_priors[j]
                + np.sum(x * log_p[j] + (1.0 - x) * log_1mp[j])
            )

    return log_post
