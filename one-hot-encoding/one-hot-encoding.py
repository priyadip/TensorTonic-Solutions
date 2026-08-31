import numpy as np

def one_hot(y: list, num_classes=None) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, K).
    """
    y = np.asarray(y, dtype=int)

    # Infer number of classes if not provided
    if num_classes is None:
        num_classes = np.max(y) + 1

    # Create N x K matrix filled with zeros
    result = np.zeros((len(y), num_classes), dtype=float)

    # Put 1 at the corresponding class position
    result[np.arange(len(y)), y] = 1.0

    return result