import numpy as np

def mean_rating_imputation(ratings_matrix: list, mode: str) -> list:
    """
    Returns a copy with missing ratings replaced by user or item means.
    """
    ratings = np.asarray(ratings_matrix, dtype=float).copy()

    if mode == "user":
        counts = np.count_nonzero(ratings, axis=1)
        sums = ratings.sum(axis=1)
        means = np.divide(
            sums,
            counts,
            out=np.zeros_like(sums),
            where=counts != 0
        )
        missing = ratings == 0
        ratings[missing] = means[np.where(missing)[0]]

    elif mode == "item":
        counts = np.count_nonzero(ratings, axis=0)
        sums = ratings.sum(axis=0)
        means = np.divide(
            sums,
            counts,
            out=np.zeros_like(sums),
            where=counts != 0
        )
        missing = ratings == 0
        ratings[missing] = means[np.where(missing)[1]]

    else:
        raise ValueError("mode must be 'user' or 'item'")

    return ratings.tolist()