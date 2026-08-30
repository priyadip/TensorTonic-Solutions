import math
from collections import Counter

def bleu_score(candidate: list, reference: list, max_n: int) -> float:
    """
    Returns the unsmoothed BLEU score.
    """
    c = len(candidate)
    r = len(reference)

    if c == 0:
        return 0.0

    precisions = []

    for n in range(1, max_n + 1):
        # No candidate n-grams means precision is zero
        if c < n:
            return 0.0

        candidate_ngrams = Counter(
            tuple(candidate[i:i + n])
            for i in range(c - n + 1)
        )
        reference_ngrams = Counter(
            tuple(reference[i:i + n])
            for i in range(max(0, r - n + 1))
        )

        clipped_matches = sum(
            min(count, reference_ngrams[ngram])
            for ngram, count in candidate_ngrams.items()
        )

        total = sum(candidate_ngrams.values())
        precision = clipped_matches / total

        if precision == 0:
            return 0.0

        precisions.append(precision)

    # Brevity penalty
    bp = 1.0 if c >= r else math.exp(1.0 - r / c)

    # Uniform-weight geometric mean
    log_mean = sum(math.log(p) for p in precisions) / max_n

    return float(bp * math.exp(log_mean))