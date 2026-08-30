import math

def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns a predicted class label for every test sample.
    """
    n = len(X_train)
    n_features = len(X_train[0])
    epsilon = 1e-9

    # Group samples by class
    classes = sorted(set(y_train))
    stats = {}

    for c in classes:
        samples = [X_train[i] for i in range(n) if y_train[i] == c]
        nc = len(samples)

        means = [
            sum(row[j] for row in samples) / nc
            for j in range(n_features)
        ]

        variances = [
            sum((row[j] - means[j]) ** 2 for row in samples) / nc
            for j in range(n_features)
        ]

        stats[c] = (nc, means, variances)

    predictions = []

    for x in X_test:
        best_class = None
        best_log_posterior = float("-inf")

        for c in classes:
            nc, means, variances = stats[c]

            # log prior
            log_posterior = math.log(nc / n)

            # Gaussian log likelihood for each feature
            for j in range(n_features):
                var = variances[j] + epsilon
                diff = x[j] - means[j]

                log_posterior += (
                    -0.5 * math.log(2 * math.pi * var)
                    - (diff * diff) / (2 * var)
                )

            if log_posterior > best_log_posterior:
                best_log_posterior = log_posterior
                best_class = c

        predictions.append(int(best_class))

    return predictions