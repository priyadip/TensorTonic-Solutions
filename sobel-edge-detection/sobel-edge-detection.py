import numpy as np

def sobel_edges(image):
    """
    Apply the Sobel operator to detect edges.
    Returns the gradient magnitude image.
    """
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    # Sobel kernels (exact as given)
    Kx = np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ])

    Ky = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])

    # Zero padding (1 pixel border)
    padded = np.pad(image, pad_width=1, mode='constant', constant_values=0)

    # Output gradient magnitude
    G = np.zeros((h, w), dtype=float)

    # Convolution
    for i in range(h):
        for j in range(w):
            region = padded[i:i+3, j:j+3]
            gx = np.sum(region * Kx)
            gy = np.sum(region * Ky)
            G[i, j] = np.sqrt(gx**2 + gy**2)

    return G.tolist()
