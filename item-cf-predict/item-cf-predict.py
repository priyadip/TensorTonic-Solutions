import numpy as np

def item_cf_predict(user_ratings: list, item_similarities: list, target: int) -> float:
    """
    Returns the similarity-weighted rating prediction.
    """
    ratings = np.asarray(user_ratings, dtype=float)
    similarities = np.asarray(item_similarities, dtype=float)

    mask = (
        (np.arange(len(ratings)) != target)
        & (ratings != 0)
        & (similarities > 0)
    )

    if not np.any(mask):
        return 0.0

    weighted_sum = np.sum(ratings[mask] * similarities[mask])
    similarity_sum = np.sum(similarities[mask])

    return float(weighted_sum / similarity_sum)