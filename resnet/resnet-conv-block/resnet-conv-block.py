import numpy as np

def conv_block(x, W1, W2, Ws):
    """
    Returns the output of a residual block with a projection shortcut.
    """
    x = np.asarray(x)
    W1 = np.asarray(W1)
    W2 = np.asarray(W2)
    Ws = np.asarray(Ws)

    # Main path: Linear -> ReLU -> Linear
    h = np.maximum(0, x @ W1)
    z = h @ W2

    # Projection shortcut
    s = x @ Ws

    # Residual addition followed by ReLU
    return np.maximum(0, z + s)