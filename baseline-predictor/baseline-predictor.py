import numpy as np

def baseline_predict(ratings_matrix: list, target_pairs: list) -> list:
    """
    Returns the baseline predictions for the requested user-item pairs.
    """
    ratings = np.asarray(ratings_matrix, dtype=float)

    # Global mean from observed ratings only
    observed = ratings != 0
    global_mean = ratings.sum() / observed.sum()

    # User means and biases
    user_counts = observed.sum(axis=1)
    user_sums = ratings.sum(axis=1)
    user_means = np.divide(
        user_sums,
        user_counts,
        out=np.zeros_like(user_sums),
        where=user_counts != 0
    )
    user_bias = np.where(user_counts != 0, user_means - global_mean, 0.0)

    # Item means and biases
    item_counts = observed.sum(axis=0)
    item_sums = ratings.sum(axis=0)
    item_means = np.divide(
        item_sums,
        item_counts,
        out=np.zeros_like(item_sums),
        where=item_counts != 0
    )
    item_bias = np.where(item_counts != 0, item_means - global_mean, 0.0)

    # Preserve target_pairs order
    pairs = np.asarray(target_pairs, dtype=int)
    predictions = (
        global_mean
        + user_bias[pairs[:, 0]]
        + item_bias[pairs[:, 1]]
    )

    return predictions.tolist()