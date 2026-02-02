def compute_monitoring_metrics(system_type, y_true, y_pred):
    import math

    n = len(y_true)
    metrics = []

    if system_type == "classification":
        tp = fp = fn = tn = 0

        for yt, yp in zip(y_true, y_pred):
            if yt == 1 and yp == 1:
                tp += 1
            elif yt == 0 and yp == 1:
                fp += 1
            elif yt == 1 and yp == 0:
                fn += 1
            else:
                tn += 1

        accuracy = (tp + tn) / n if n > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics = [
            ("accuracy", accuracy),
            ("f1", f1),
            ("precision", precision),
            ("recall", recall),
        ]

    elif system_type == "regression":
        abs_err = 0.0
        sq_err = 0.0

        for yt, yp in zip(y_true, y_pred):
            diff = yt - yp
            abs_err += abs(diff)
            sq_err += diff * diff

        mae = abs_err / n if n > 0 else 0.0
        rmse = math.sqrt(sq_err / n) if n > 0 else 0.0

        metrics = [
            ("mae", mae),
            ("rmse", rmse),
        ]

    elif system_type == "ranking":
        paired = list(zip(y_true, y_pred))
        paired.sort(key=lambda x: x[1], reverse=True)

        top_k = paired[:3]
        relevant_top_k = sum(yt for yt, _ in top_k)
        total_relevant = sum(y_true)

        precision_at_3 = relevant_top_k / 3
        recall_at_3 = (
            relevant_top_k / total_relevant
            if total_relevant > 0
            else 0.0
        )

        metrics = [
            ("precision_at_3", precision_at_3),
            ("recall_at_3", recall_at_3),
        ]

    return sorted(metrics, key=lambda x: x[0])
