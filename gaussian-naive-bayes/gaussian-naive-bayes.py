import math

def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.
    
    Args:
        X_train: list of lists (n_train, d)
        y_train: list (n_train,)
        X_test: list of lists (n_test, d)
        
    Returns:
        list of predicted class labels
    """
    eps = 1e-9

    n = len(X_train)
    d = len(X_train[0])

    # ---- Collect class indices ----
    classes = list(set(y_train))
    class_indices = {c: [] for c in classes}
    for i, c in enumerate(y_train):
        class_indices[c].append(i)

    # ---- Compute priors, means, variances ----
    priors = {}
    means = {}
    variances = {}

    for c in classes:
        idxs = class_indices[c]
        nc = len(idxs)

        priors[c] = nc / n

        mu = [0.0] * d
        for j in range(d):
            mu[j] = sum(X_train[i][j] for i in idxs) / nc

        var = [0.0] * d
        for j in range(d):
            var[j] = sum((X_train[i][j] - mu[j]) ** 2 for i in idxs) / nc
            var[j] += eps  # numerical stability

        means[c] = mu
        variances[c] = var

    # ---- Predict ----
    predictions = []

    for x in X_test:
        best_class = None
        best_log_prob = -float("inf")

        for c in classes:
            log_prob = math.log(priors[c])

            for j in range(d):
                mu = means[c][j]
                var = variances[c][j]

                log_prob += (
                    -0.5 * math.log(2 * math.pi * var)
                    - (x[j] - mu) ** 2 / (2 * var)
                )

            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_class = c

        predictions.append(best_class)

    return predictions
