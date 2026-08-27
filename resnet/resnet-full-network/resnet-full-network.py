import numpy as np

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns: np.ndarray of shape (batch, num_classes) with classification logits
    """
    x = np.asarray(x)
    conv1 = np.asarray(conv1)
    W1_b1 = np.asarray(W1_b1)
    W2_b1 = np.asarray(W2_b1)
    W1_b2 = np.asarray(W1_b2)
    W2_b2 = np.asarray(W2_b2)
    fc = np.asarray(fc)

    def relu(a):
        return np.maximum(0, a)

    # Initial convolution + ReLU
    out = relu(x @ conv1)

    # BasicBlock 1: identity shortcut
    shortcut = out
    h = relu(out @ W1_b1)
    h = h @ W2_b1
    out = relu(h + shortcut)

    # BasicBlock 2: projection shortcut
    shortcut = out @ np.asarray(Ws_b2)
    h = relu(out @ W1_b2)
    h = h @ W2_b2
    out = relu(h + shortcut)

    # Fully connected classification layer
    return out @ fc