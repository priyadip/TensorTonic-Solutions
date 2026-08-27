import numpy as np

def prepend_class_token(
    patches: np.ndarray,
    embed_dim: int,
    cls_token: np.ndarray = None
) -> np.ndarray:
    """
    Prepend learnable [CLS] token to patch sequence.
    cls_token: shape (1, 1, D). If None, initialize randomly.
    """
    B = patches.shape[0]

    if cls_token is None:
        cls_token = np.random.randn(1, 1, embed_dim) * 0.02

    # Replicate the same learnable CLS token across the batch.
    cls_tokens = np.tile(cls_token, (B, 1, 1))

    # Put CLS at position 0, before all patch embeddings.
    return np.concatenate([cls_tokens, patches], axis=1)