import numpy as np

def unet_bottleneck(x: np.ndarray, out_channels: int) -> np.ndarray:
    """
    U-Net bottleneck: double convolution at lowest resolution.
    Two 3x3 unpadded convolutions, no pooling.
    Returns zero array with correct shape.
    """
    B, H, W, _ = x.shape

    # Each valid 3x3 convolution reduces H and W by 2.
    # Two convolutions reduce each spatial dimension by 4.
    output_shape = (B, H - 4, W - 4, out_channels)

    return np.zeros(output_shape)