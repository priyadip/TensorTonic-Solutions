import numpy as np

def compute_monitoring_metrics(system_type: str, y_true: list, y_pred: list) -> dict:
    """
    Returns a dictionary of metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if system_type == "classification":
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        accuracy = (tp + tn) / len(y_true)

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    elif system_type == "regression":
        errors = y_true.astype(float) - y_pred.astype(float)

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))

        return {
            "mae": float(mae),
            "rmse": float(rmse),
        }

    elif system_type == "ranking":
        # Stable descending sort: ties retain their original input order.
        order = np.argsort(-y_pred, kind="stable")
        top3 = order[:3]

        relevant_total = np.sum(y_true > 0)
        relevant_top3 = np.sum(y_true[top3] > 0)

        precision_at_3 = relevant_top3 / 3.0
        recall_at_3 = (
            relevant_top3 / relevant_total
            if relevant_total > 0
            else 0.0
        )

        return {
            "precision_at_3": float(precision_at_3),
            "recall_at_3": float(recall_at_3),
        }

    else:
        raise ValueError("system_type must be 'classification', 'regression', or 'ranking'")