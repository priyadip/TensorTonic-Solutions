import numpy as np

def adjusted_cosine_similarity( ratings_matrix: list, item_i: int, item_j: int ) -> float:
    """
    Returns the adjusted cosine similarity between the requested items.
    """

    R = np.asarray(ratings_matrix, dtype=float)

    # Users who rated BOTH items
    mask = (R[:, item_i] != 0) & (R[:, item_j] != 0)

    # Keep only those users
    ratings = R[mask]

    if len(ratings) == 0:
        return 0.0

    # Calculate each user's mean from nonzero ratings
    means = np.sum(ratings * (ratings != 0), axis=1) / np.sum(ratings != 0, axis=1)

    # Ratings for the two requested items
    a = ratings[:, item_i] - means
    b = ratings[:, item_j] - means

    # Adjusted cosine similarity
    numerator = np.sum(a * b)
    denominator = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)
