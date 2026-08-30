import math
from collections import Counter
import numpy as np

def bm25_score(query_tokens: list[str], docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """
    Returns a NumPy array with one score per document.
    """
    n_docs = len(docs)
    doc_lengths = np.array([len(doc) for doc in docs], dtype=float)
    avgdl = doc_lengths.mean()

    # Document frequency for each term
    df = Counter()
    for doc in docs:
        df.update(set(doc))

    scores = np.zeros(n_docs, dtype=float)

    # Repeated query terms are counted once
    for term in set(query_tokens):
        term_df = df.get(term, 0)

        # Unseen query terms contribute nothing
        if term_df == 0:
            continue

        idf = math.log(
            (n_docs - term_df + 0.5) / (term_df + 0.5) + 1
        )

        for i, doc in enumerate(docs):
            tf = doc.count(term)

            if tf == 0:
                continue

            denominator = (
                tf
                + k1 * (1 - b + b * doc_lengths[i] / avgdl)
            )

            scores[i] += (
                idf
                * (tf * (k1 + 1))
                / denominator
            )

    return scores