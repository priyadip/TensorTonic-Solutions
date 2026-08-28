import numpy as np

def unet_encoder_block(x: np.ndarray, out_channels: int) -> tuple:
    """
    Returns (pool_out, skip_out) as zero arrays with correct shapes.
    """
    B, H, W, _ = x.shape

    # Two 3x3 valid convolutions reduce H and W by 4 total.
    skip_shape = (B, H - 4, W - 4, out_channels)

    # 2x2 max pooling with stride 2 halves the spatial dimensions.
    pool_shape = (B, (H - 4) // 2, (W - 4) // 2, out_channels)

    skip_out = np.zeros(skip_shape)
    pool_out = np.zeros(pool_shape)

    return pool_out, skip_out