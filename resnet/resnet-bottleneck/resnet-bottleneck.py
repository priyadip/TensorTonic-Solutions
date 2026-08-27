import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns the bottleneck residual block output.
    """
    x = np.asarray(x)
    W1 = np.asarray(W1)
    W2 = np.asarray(W2)
    W3 = np.asarray(W3)

    def relu(a):
        return np.maximum(0, a)

    # Main path: 1x1 reduce -> 3x3 process -> 1x1 expand
    h = relu(x @ W1)
    h = relu(h @ W2)
    h = h @ W3

    # Skip connection
    if Ws is None:
        shortcut = x
    else:
        shortcut = x @ np.asarray(Ws)

    # Add residual and apply final ReLU
    return relu(h + shortcut)