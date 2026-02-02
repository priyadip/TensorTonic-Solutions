import math

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.

    Args:
        feature_size (int): size of feature grid (feature_size x feature_size)
        image_size (int): original image size (assumed square)
        scales (list[float]): list of scales
        aspect_ratios (list[float]): list of aspect ratios

    Returns:
        list of [x1, y1, x2, y2] anchor boxes
    """
    anchors = []

    # 1. Compute stride
    stride = image_size / feature_size

    # 2. Iterate over grid cells (row-major: i then j)
    for i in range(feature_size):
        for j in range(feature_size):
            # Center coordinates
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride

            # 3. Iterate over scales and aspect ratios
            for s in scales:
                for r in aspect_ratios:
                    w = s * math.sqrt(r)
                    h = s / math.sqrt(r)

                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2

                    anchors.append([x1, y1, x2, y2])

    return anchors
