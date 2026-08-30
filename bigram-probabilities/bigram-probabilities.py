import numpy as np

def bigram_probabilities(tokens: list) -> dict:
    """
    Returns a dictionary with vocab, counts, and probabilities.
    """
    vocab = sorted(set(tokens))
    V = len(vocab)

    # Map tokens to their matrix indices
    token_to_idx = {token: i for i, token in enumerate(vocab)}

    # Count adjacent bigrams
    counts = np.zeros((V, V), dtype=int)

    for i in range(len(tokens) - 1):
        row = token_to_idx[tokens[i]]
        col = token_to_idx[tokens[i + 1]]
        counts[row, col] += 1

    # Add-one smoothing
    smoothed = counts + 1
    probabilities = smoothed / smoothed.sum(axis=1, keepdims=True)

    return {
        "vocab": vocab,
        "counts": counts,
        "probabilities": probabilities
    }