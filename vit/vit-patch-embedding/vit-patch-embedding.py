import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int, W_proj: np.ndarray = None) -> np.ndarray:
    """
    Convert image to patch embeddings.
    W_proj: projection matrix of shape (patch_dim, embed_dim). If None, initialize randomly.
    """
    B, H, W, C = image.shape
    P = patch_size

    # Number of patches along each spatial dimension
    h_patches = H // P
    w_patches = W // P

    # Split into patches:
    # (B, H, W, C)
    # -> (B, H/P, P, W/P, P, C)
    # -> (B, H/P, W/P, P, P, C)
    # -> (B, N, P*P*C)
    patches = image.reshape(B, h_patches, P, w_patches, P, C)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    patches = patches.reshape(B, h_patches * w_patches, P * P * C)

    # Initialize projection matrix if one wasn't supplied
    patch_dim = P * P * C
    if W_proj is None:
        W_proj = np.random.randn(patch_dim, embed_dim) * 0.02

    # Linear projection: (B, N, patch_dim) @ (patch_dim, D)
    return patches @ W_proj