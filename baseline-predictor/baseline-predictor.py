def baseline_predict(ratings_matrix, target_pairs):
    """
    Compute baseline predictions using global mean and user/item biases.
    """

    n_users = len(ratings_matrix)
    n_items = len(ratings_matrix[0])

    # ---- Collect all non-zero ratings ----
    all_ratings = []
    for u in range(n_users):
        for i in range(n_items):
            if ratings_matrix[u][i] != 0:
                all_ratings.append(ratings_matrix[u][i])

    # Global mean
    mu = sum(all_ratings) / len(all_ratings) if all_ratings else 0.0

    # ---- User means ----
    user_means = [None] * n_users
    for u in range(n_users):
        vals = [r for r in ratings_matrix[u] if r != 0]
        user_means[u] = sum(vals) / len(vals) if vals else None

    # ---- Item means ----
    item_means = [None] * n_items
    for i in range(n_items):
        vals = []
        for u in range(n_users):
            if ratings_matrix[u][i] != 0:
                vals.append(ratings_matrix[u][i])
        item_means[i] = sum(vals) / len(vals) if vals else None

    # ---- Biases ----
    user_bias = [
        (user_means[u] - mu) if user_means[u] is not None else 0.0
        for u in range(n_users)
    ]

    item_bias = [
        (item_means[i] - mu) if item_means[i] is not None else 0.0
        for i in range(n_items)
    ]

    # ---- Predictions ----
    predictions = []
    for u, i in target_pairs:
        pred = mu + user_bias[u] + item_bias[i]
        predictions.append(pred)

    return predictions
