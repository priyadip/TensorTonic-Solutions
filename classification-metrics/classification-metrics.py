import numpy as np

def classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    average: str = "micro",
    pos_label: int = 1
) -> dict:
    """
    Returns a dictionary containing accuracy, precision, recall, and f1
    rounded to six decimals.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = np.mean(y_true == y_pred)

    # Include every class appearing in either array.
    classes = np.unique(np.concatenate([y_true, y_pred]))

    tp = np.array([
        np.sum((y_true == c) & (y_pred == c))
        for c in classes
    ], dtype=float)

    fp = np.array([
        np.sum((y_true != c) & (y_pred == c))
        for c in classes
    ], dtype=float)

    fn = np.array([
        np.sum((y_true == c) & (y_pred != c))
        for c in classes
    ], dtype=float)

    support = tp + fn

    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(tp),
        where=(tp + fp) != 0
    )

    recall = np.divide(
        tp,
        tp + fn,
        out=np.zeros_like(tp),
        where=(tp + fn) != 0
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0
    )

    if average == "micro":
        total_tp = np.sum(tp)
        total_fp = np.sum(fp)
        total_fn = np.sum(fn)

        p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
        r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0

    elif average == "macro":
        p = np.mean(precision)
        r = np.mean(recall)
        f = np.mean(f1)

    elif average == "weighted":
        total_support = np.sum(support)
        if total_support == 0:
            p = r = f = 0.0
        else:
            weights = support / total_support
            p = np.sum(precision * weights)
            r = np.sum(recall * weights)
            f = np.sum(f1 * weights)

    elif average == "binary":
        matches = np.where(classes == pos_label)[0]

        if len(matches) == 0:
            p = r = f = 0.0
        else:
            i = matches[0]
            p = precision[i]
            r = recall[i]
            f = f1[i]

    else:
        raise ValueError(
            "average must be 'micro', 'macro', 'weighted', or 'binary'"
        )

    return {
        "accuracy": round(float(accuracy), 6),
        "precision": round(float(p), 6),
        "recall": round(float(r), 6),
        "f1": round(float(f), 6)
    }