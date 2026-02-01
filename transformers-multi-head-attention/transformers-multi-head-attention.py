import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def multi_head_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    W_o: np.ndarray,
    num_heads: int
) -> np.ndarray:
    """
    Compute multi-head attention (NumPy).
    """
    B, T, d_model = Q.shape
    d_head = d_model // num_heads

    # 1. Linear projections
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    # 2. Split into heads: (B, num_heads, T, d_head)
    def split_heads(x):
        return x.reshape(B, T, num_heads, d_head).transpose(0, 2, 1, 3)

    Qh = split_heads(Q_proj)
    Kh = split_heads(K_proj)
    Vh = split_heads(V_proj)

    # 3. Scaled dot-product attention
    scores = np.matmul(Qh, Kh.transpose(0, 1, 3, 2)) / np.sqrt(d_head)
    attn = softmax(scores, axis=-1)
    head_out = np.matmul(attn, Vh)  # (B, num_heads, T, d_head)

    # 4. Concatenate heads
    concat = head_out.transpose(0, 2, 1, 3).reshape(B, T, d_model)

    # 5. Final linear projection
    output = concat @ W_o

    return output
