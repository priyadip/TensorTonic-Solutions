def decision_tree_split(X, y):
    """
    Find the best feature index and threshold using Gini impurity.
    Returns (best_feature, best_threshold).
    """

    n = len(y)
    d = len(X[0])

    # ---------- Helper: Gini impurity ----------
    def gini(labels):
        total = len(labels)
        if total == 0:
            return 0.0
        counts = {}
        for v in labels:
            counts[v] = counts.get(v, 0) + 1
        impurity = 1.0
        for c in counts.values():
            p = c / total
            impurity -= p * p
        return impurity

    parent_gini = gini(y)

    best_gain = -1.0
    best_feature = 0
    best_threshold = 0.0

    # ---------- Try all features ----------
    for f in range(d):
        values = sorted(set(row[f] for row in X))

        # Try midpoints between consecutive values
        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2.0

            left_y = []
            right_y = []

            for xi, yi in zip(X, y):
                if xi[f] <= threshold:
                    left_y.append(yi)
                else:
                    right_y.append(yi)

            # Skip invalid splits
            if not left_y or not right_y:
                continue

            g_left = gini(left_y)
            g_right = gini(right_y)

            weighted_gini = (
                len(left_y) / n * g_left +
                len(right_y) / n * g_right
            )

            gain = parent_gini - weighted_gini

            # ---------- Tie-breaking ----------
            if (
                gain > best_gain or
                (abs(gain - best_gain) < 1e-12 and
                 (f < best_feature or
                  (f == best_feature and threshold < best_threshold)))
            ):
                best_gain = gain
                best_feature = f
                best_threshold = threshold
    feature, threshold=best_feature, best_threshold
    return [feature, threshold]
