import numpy as np

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns a dictionary containing the block output and mode.
    """
    x = np.asarray(x)
    W1 = np.asarray(W1)
    W2 = np.asarray(W2)
    gamma1 = np.asarray(gamma1)
    beta1 = np.asarray(beta1)
    gamma2 = np.asarray(gamma2)
    beta2 = np.asarray(beta2)

    eps = 1e-5

    def bn(a, gamma, beta):
        mean = np.mean(a, axis=0, keepdims=True)
        var = np.var(a, axis=0, keepdims=True)
        normalized = (a - mean) / np.sqrt(var + eps)
        return gamma * normalized + beta

    def relu(a):
        return np.maximum(0, a)

    if mode == "post":
        # Conv -> BN -> ReLU
        h = x @ W1
        h = bn(h, gamma1, beta1)
        h = relu(h)

        # Conv -> BN
        h = h @ W2
        h = bn(h, gamma2, beta2)

        # Add skip -> ReLU
        output = relu(h + x)

    elif mode == "pre":
        # BN -> ReLU -> Conv
        h = bn(x, gamma1, beta1)
        h = relu(h)
        h = h @ W1

        # BN -> ReLU -> Conv
        h = bn(h, gamma2, beta2)
        h = relu(h)
        h = h @ W2

        # Add identity shortcut
        output = h + x

    else:
        raise ValueError("mode must be 'post' or 'pre'")

    return {
        "output": output.tolist(),
        "mode": mode
    }