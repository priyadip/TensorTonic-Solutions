def jaccard_similarity(set_a: list, set_b: list) -> float:
    """
    Returns the Jaccard similarity of the two item collections.
    """

    A = set(set_a)
    B = set(set_b)

    intersection = A & B
    union = A | B

    if len(union) == 0:
        return 0.0

    return float(len(intersection) / len(union))
