import numpy as np

def unet(x: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    Complete U-Net: trace shape through 4 encoder blocks, bottleneck,
    4 decoder blocks, and the output layer.

    Returns a zero array with the correct final shape.
    """
    B, H, W, _ = x.shape

    # Encoder: each block applies two valid 3x3 convolutions (-4),
    # followed by 2x2 max pooling (//2).
    for _ in range(4):
        H = (H - 4) // 2
        W = (W - 4) // 2

    # Bottleneck: two valid 3x3 convolutions (-4).
    H -= 4
    W -= 4

    # Decoder: each block upsamples by 2, then applies two
    # valid 3x3 convolutions (-4).
    for _ in range(4):
        H = 2 * H - 4
        W = 2 * W - 4

    # 1x1 output convolution preserves spatial dimensions.
    return np.zeros((B, H, W, num_classes))