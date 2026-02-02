def morphological_op(image, kernel, operation):
    """
    Apply morphological erosion or dilation to a binary image.
    
    Args:
        image: 2D list of 0/1
        kernel: 2D list of 0/1 (structuring element)
        operation: "erode" or "dilate"
        
    Returns:
        2D list of 0/1
    """
    H = len(image)
    W = len(image[0])

    kH = len(kernel)
    kW = len(kernel[0])

    pad_h = kH // 2
    pad_w = kW // 2

    # Zero padding
    padded = [[0] * (W + 2 * pad_w) for _ in range(H + 2 * pad_h)]
    for i in range(H):
        for j in range(W):
            padded[i + pad_h][j + pad_w] = image[i][j]

    output = [[0 for _ in range(W)] for _ in range(H)]

    for i in range(H):
        for j in range(W):
            if operation == "erode":
                result = 1
                for ki in range(kH):
                    for kj in range(kW):
                        if kernel[ki][kj] == 1:
                            if padded[i + ki][j + kj] == 0:
                                result = 0
                                break
                    if result == 0:
                        break
                output[i][j] = result

            elif operation == "dilate":
                result = 0
                for ki in range(kH):
                    for kj in range(kW):
                        if kernel[ki][kj] == 1:
                            if padded[i + ki][j + kj] == 1:
                                result = 1
                                break
                    if result == 1:
                        break
                output[i][j] = result

            else:
                raise ValueError("operation must be 'erode' or 'dilate'")

    return output
