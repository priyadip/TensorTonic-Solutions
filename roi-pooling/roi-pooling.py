import math

def roi_pool(feature_map, rois, output_size):
    """
    Apply ROI Pooling to extract fixed-size features.
    
    Args:
        feature_map: 2D list (H x W)
        rois: list of [x1, y1, x2, y2] in feature-map coordinates
        output_size: int (e.g., 3 for 3x3)
        
    Returns:
        list of 2D lists, one per ROI (each output_size x output_size)
    """
    H = len(feature_map)
    W = len(feature_map[0])

    outputs = []

    for roi in rois:
        x1, y1, x2, y2 = roi

        roi_h = y2 - y1
        roi_w = x2 - x1

        pooled = [[0.0 for _ in range(output_size)] for _ in range(output_size)]

        for i in range(output_size):
            for j in range(output_size):

                # Bin boundaries (exact formula)
                h_start = y1 + math.floor(i * roi_h / output_size)
                h_end   = y1 + math.floor((i + 1) * roi_h / output_size)
                w_start = x1 + math.floor(j * roi_w / output_size)
                w_end   = x1 + math.floor((j + 1) * roi_w / output_size)

                # Ensure at least one pixel
                if h_end <= h_start:
                    h_end = h_start + 1
                if w_end <= w_start:
                    w_end = w_start + 1

                # Clamp to feature map
                h_start = max(0, min(h_start, H))
                h_end   = max(0, min(h_end, H))
                w_start = max(0, min(w_start, W))
                w_end   = max(0, min(w_end, W))

                # Max pooling in the bin
                max_val = -float("inf")
                for y in range(h_start, h_end):
                    for x in range(w_start, w_end):
                        max_val = max(max_val, feature_map[y][x])

                pooled[i][j] = max_val

        outputs.append(pooled)

    return outputs
