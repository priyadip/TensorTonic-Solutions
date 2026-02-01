import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    B, T, d_model = Q.shape
    d_head = d_model // num_heads

    # Linear projections
    Q = Q @ W_q
    K = K @ W_k
    V = V @ W_v

    # Split heads: (B, num_heads, T, d_head)
    Q = Q.reshape(B, T, num_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, T, num_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, T, num_heads, d_head).transpose(0, 2, 1, 3)

    # Scaled dot-product attention
    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(d_head)
    weights = softmax(scores, axis=-1)
    attn = weights @ V

    # Concatenate heads
    attn = attn.transpose(0, 2, 1, 3).reshape(B, T, d_model)

    # Output projection
    return attn @ W_o

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    hidden = np.maximum(0, x @ W1 + b1)   # ReLU
    return hidden @ W2 + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Multi-Head Attention + Residual + LN
    mha_out = multi_head_attention(
        x, x, x, W_q, W_k, W_v, W_o, num_heads
    )
    z = layer_norm(x + mha_out, gamma1, beta1)

    # Feed-Forward + Residual + LN
    ffn_out = feed_forward(z, W1, b1, W2, b2)
    out = layer_norm(z + ffn_out, gamma2, beta2)

    return out