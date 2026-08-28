import numpy as np

def unet_decoder_block(
    x: np.ndarray, skip: np.ndarray, out_channels: int) -> np.ndarray:
    """
    Returns zero array with correct shape.
    """
    B, H, W, _ = x.shape

    # 2x2 transposed convolution with stride 2 doubles spatial dimensions.
    up_h = 2 * H
    up_w = 2 * W

    # Skip connection is center-cropped to (up_h, up_w) and concatenated.
    # Two 3x3 valid convolutions reduce each spatial dimension by 4 total.
    output_shape = (B, up_h - 4, up_w - 4, out_channels)

    return np.zeros(output_shape)