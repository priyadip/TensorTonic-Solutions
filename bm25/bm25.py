import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    N = len(docs)
    doc_lengths = np.array([len(doc) for doc in docs], dtype=float)
    avgdl = doc_lengths.mean() if N > 0 else 0.0

    # Term frequencies per document
    doc_tf = [Counter(doc) for doc in docs]

    # Document frequency for each query term
    df = {}
    for term in query_tokens:
        df[term] = sum(1 for doc in docs if term in doc)

    scores = np.zeros(N, dtype=float)

    for i, doc in enumerate(docs):
        score = 0.0
        dl = doc_lengths[i]

        for term in query_tokens:
            tf = doc_tf[i].get(term, 0)
            if tf == 0:
                continue

            # IDF as defined in the prompt
            idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)

            denom = tf + k1 * (1 - b + b * dl / avgdl)
            score += idf * (tf * (k1 + 1)) / denom

        scores[i] = score

    return scores
