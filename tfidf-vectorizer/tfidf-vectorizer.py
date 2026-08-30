import math
from collections import Counter
import numpy as np

def tfidf_vectorizer(documents: list[str]) -> dict:
    """
    Returns a dictionary with tfidf_matrix and vocabulary.
    """
    tokenized = [doc.lower().split() for doc in documents]
    n_docs = len(documents)

    # Sorted unique vocabulary
    vocabulary = sorted(set(
        token for tokens in tokenized for token in tokens
    ))

    vocab_index = {term: i for i, term in enumerate(vocabulary)}
    n_terms = len(vocabulary)

    # Document frequency
    df = np.zeros(n_terms, dtype=int)
    for tokens in tokenized:
        for term in set(tokens):
            df[vocab_index[term]] += 1

    # Unsmoothed natural-log IDF
    idf = np.log(n_docs / df)

    # TF-IDF matrix
    tfidf = np.zeros((n_docs, n_terms), dtype=float)

    for d, tokens in enumerate(tokenized):
        counts = Counter(tokens)
        doc_len = len(tokens)

        for term, count in counts.items():
            j = vocab_index[term]
            tfidf[d, j] = (count / doc_len) * idf[j]

    return {
        "tfidf_matrix": tfidf,
        "vocabulary": vocabulary,
    }