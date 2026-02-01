import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is included.
    """
    pe = np.zeros((seq_len, d_model), dtype=float)

    positions = np.arange(seq_len)[:, None]          # (seq_len, 1)
    dims = np.arange(d_model)[None, :]                # (1, d_model)

    div_term = base ** (2 * (dims // 2) / d_model)

    pe[:, 0::2] = np.sin(positions / div_term[:, 0::2])
    pe[:, 1::2] = np.cos(positions / div_term[:, 1::2])

    return pe


def add_positional_encoding(x, base=10000.0):
    """
    Add PE to input x of shape (B, T, d_model); return same shape.
    """
    B, T, d_model = x.shape
    pe = positional_encoding(T, d_model, base)
    return x + pe
