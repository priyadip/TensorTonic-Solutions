def linear_interpolation(values: list) -> list:
    """Returns a copy with every missing value interpolated."""
    result = values.copy()
    n = len(result)
    i = 0
    while i < n:
        if result[i] is not None: # Find the beginning of a missing block
            i += 1
            continue
        left = i - 1

        right = i
        while result[right] is None: # Find the first known value after the missing block
            right += 1

        for j in range(left + 1, right):
            result[j] = (result[left] + (j - left) / (right - left) * (result[right] - result[left]))

        i = right

    return result