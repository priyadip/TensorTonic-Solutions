import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor
) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # d_k = key dimension
    d_k = Q.size(-1)

    # QK^T
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Softmax over keys
    attn_weights = F.softmax(scores, dim=-1)

    # Weighted sum of values
    output = torch.matmul(attn_weights, V)

    return output
