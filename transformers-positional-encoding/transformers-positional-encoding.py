import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    Returns array of shape (seq_length, d_model).
    """
    # Position indices (seq_length, 1)
    pos = np.arange(seq_length).reshape(-1, 1)

    # Dimension indices (1, d_model)
    i = np.arange(d_model).reshape(1, -1)

    # Compute angle rates
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
    angles = pos * angle_rates

    # Initialize PE matrix
    pe = np.zeros((seq_length, d_model))

    # Apply sin to even indices, cos to odd indices
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])

    return pe
