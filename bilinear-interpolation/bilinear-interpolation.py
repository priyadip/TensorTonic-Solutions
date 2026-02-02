import math

def bilinear_resize(image, new_h, new_w):
    """
    Resize a 2D grid using bilinear interpolation.
    Returns a 2D list of floats.
    """
    H = len(image)
    W = len(image[0])

    # Handle degenerate cases
    if new_h == 0 or new_w == 0:
        return []

    output = [[0.0 for _ in range(new_w)] for _ in range(new_h)]

    for i in range(new_h):
        # Map output row to source row
        if new_h == 1:
            src_y = 0.0
        else:
            src_y = i * (H - 1) / (new_h - 1)

        y0 = int(math.floor(src_y))
        y1 = min(y0 + 1, H - 1)
        dy = src_y - y0

        for j in range(new_w):
            # Map output column to source column
            if new_w == 1:
                src_x = 0.0
            else:
                src_x = j * (W - 1) / (new_w - 1)

            x0 = int(math.floor(src_x))
            x1 = min(x0 + 1, W - 1)
            dx = src_x - x0

            # Bilinear interpolation
            val = (
                image[y0][x0] * (1 - dy) * (1 - dx)
                + image[y1][x0] * dy * (1 - dx)
                + image[y0][x1] * (1 - dy) * dx
                + image[y1][x1] * dy * dx
            )

            output[i][j] = float(val)

    return output
