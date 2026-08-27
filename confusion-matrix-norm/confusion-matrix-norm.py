import numpy as np

def confusion_matrix_norm(
    y_true: list,
    y_pred: list,
    num_classes: int | None = None,
    normalize: str = "none"
) -> np.ndarray:
    """
    Returns the confusion matrix as a NumPy array.
    Rows = true classes, columns = predicted classes.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max()) + 1)

    # Count true/predicted class pairs.
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)

    if normalize == "none":
        return matrix

    if normalize == "true":
        row_sums = matrix.sum(axis=1, keepdims=True)
        return np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix, dtype=float),
            where=row_sums != 0
        )

    if normalize == "pred":
        col_sums = matrix.sum(axis=0, keepdims=True)
        return np.divide(
            matrix,
            col_sums,
            out=np.zeros_like(matrix, dtype=float),
            where=col_sums != 0
        )

    if normalize == "all":
        total = matrix.sum()
        if total == 0:
            return np.zeros_like(matrix, dtype=float)
        return matrix.astype(float) / total

    raise ValueError("normalize must be 'none', 'true', 'pred', or 'all'")