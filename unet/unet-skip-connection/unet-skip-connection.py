import numpy as np

def crop_and_concat(encoder_features: np.ndarray, decoder_features: np.ndarray) -> np.ndarray:
    """
    Crop encoder features to match decoder spatial dims,
    then concatenate along channels.
    """
    _, H_enc, W_enc, _ = encoder_features.shape
    _, H_dec, W_dec, _ = decoder_features.shape

    # Compute symmetric crop amounts.
    crop_h = (H_enc - H_dec) // 2
    crop_w = (W_enc - W_dec) // 2

    # Center-crop the encoder feature map.
    encoder_cropped = encoder_features[
        :,
        crop_h:crop_h + H_dec,
        crop_w:crop_w + W_dec,
        :
    ]

    # Concatenate encoder and decoder features along channels.
    return np.concatenate((encoder_cropped, decoder_features), axis=3)