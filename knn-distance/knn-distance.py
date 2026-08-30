import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    # Handle 1D input: make each scalar a separate sample
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    # Calculate squared Euclidean distances
    distances = np.sum((X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ** 2, axis=2)

    # Sort training indices by distance
    indices = np.argsort(distances, axis=1)

    # Take k nearest neighbors
    result = indices[:, :k]

    # If k > number of training samples, pad with -1
    if k > len(X_train):
        padding = np.full((len(X_test), k - len(X_train)), -1, dtype=int)
        
        result = np.hstack((result, padding))

    return result.astype(int)

