import numpy as np

def vit_encoder_block(
    x: np.ndarray,
    embed_dim: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    Wq: np.ndarray = None,
    Wk: np.ndarray = None,
    Wv: np.ndarray = None,
    Wo: np.ndarray = None,
    W1: np.ndarray = None,
    W2: np.ndarray = None
) -> np.ndarray:
    """
    ViT Transformer encoder block with Pre-LayerNorm.
    Weight matrices are provided as inputs for deterministic testing.
    """
    B, N, D = x.shape
    assert D == embed_dim
    assert embed_dim % num_heads == 0

    head_dim = embed_dim // num_heads
    hidden_dim = int(embed_dim * mlp_ratio)

    # Initialize weights when not provided.
    def init_weight(shape):
        return np.random.randn(*shape) * 0.02

    if Wq is None:
        Wq = init_weight((embed_dim, embed_dim))
    if Wk is None:
        Wk = init_weight((embed_dim, embed_dim))
    if Wv is None:
        Wv = init_weight((embed_dim, embed_dim))
    if Wo is None:
        Wo = init_weight((embed_dim, embed_dim))
    if W1 is None:
        W1 = init_weight((embed_dim, hidden_dim))
    if W2 is None:
        W2 = init_weight((hidden_dim, embed_dim))

    # LayerNorm over the embedding dimension.
    def layer_norm(t):
        mean = np.mean(t, axis=-1, keepdims=True)
        var = np.var(t, axis=-1, keepdims=True)
        return (t - mean) / np.sqrt(var + 1e-6)

    # ---- Pre-LN + Multi-Head Self-Attention ----
    x_norm = layer_norm(x)

    Q = x_norm @ Wq
    K = x_norm @ Wk
    V = x_norm @ Wv

    # (B, N, D) -> (B, num_heads, N, head_dim)
    Q = Q.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)

    # Scaled dot-product attention.
    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)

    # Numerically stable softmax.
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    attention = np.exp(scores)
    attention /= np.sum(attention, axis=-1, keepdims=True)

    attn_output = attention @ V

    # (B, num_heads, N, head_dim) -> (B, N, D)
    attn_output = attn_output.transpose(0, 2, 1, 3)
    attn_output = attn_output.reshape(B, N, embed_dim)

    # Output projection + first residual.
    attn_output = attn_output @ Wo
    x_prime = x + attn_output

    # ---- Pre-LN + MLP ----
    x_norm = layer_norm(x_prime)

    hidden = x_norm @ W1

    # GELU approximation.
    hidden = 0.5 * hidden * (
        1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) *
            (hidden + 0.044715 * hidden**3)
        )
    )

    mlp_output = hidden @ W2

    # Second residual.
    return x_prime + mlp_output