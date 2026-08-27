import numpy as np

def classification_head(
    encoder_output: np.ndarray,
    num_classes: int,
    W_head: np.ndarray = None
) -> np.ndarray:
    """
    Classification head for ViT. Extract [CLS], LayerNorm, linear projection.
    W_head: projection matrix (D, num_classes). If None, initialize randomly.
    """
    # Extract [CLS] token at position 0.
    cls_token = encoder_output[:, 0, :]  # (B, D)

    # LayerNorm over the embedding dimension.
    mean = np.mean(cls_token, axis=-1, keepdims=True)
    var = np.var(cls_token, axis=-1, keepdims=True)
    cls_norm = (cls_token - mean) / np.sqrt(var + 1e-6)

    # Initialize classification head if not provided.
    embed_dim = encoder_output.shape[-1]
    if W_head is None:
        W_head = np.random.randn(embed_dim, num_classes) * 0.02

    # Linear projection to raw class logits.
    return cls_norm @ W_head