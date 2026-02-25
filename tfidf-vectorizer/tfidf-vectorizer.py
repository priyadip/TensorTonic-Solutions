import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    if not documents:
        return np.zeros((0, 0)), []

    # Tokenize documents
    tokenized = [doc.lower().split() for doc in documents]
    N = len(tokenized)

    # Build sorted vocabulary
    vocab = sorted(set(word for doc in tokenized for word in doc))
    vocab_index = {w: i for i, w in enumerate(vocab)}

    # Document frequency
    df = Counter()
    for doc in tokenized:
        for w in set(doc):
            df[w] += 1

    # IDF values
    idf = {w: math.log(N / df[w]) for w in vocab}

    # TF-IDF matrix
    tfidf = np.zeros((N, len(vocab)))

    for i, doc in enumerate(tokenized):
        if not doc:
            continue

        counts = Counter(doc)
        total = len(doc)

        for w, c in counts.items():
            j = vocab_index[w]
            tf = c / total
            tfidf[i, j] = tf * idf[w]

    return tfidf, vocab