def calibrate_isotonic(cal_labels: list, cal_probs: list, new_probs: list) -> list:
    """
    Fit isotonic regression using the Pool Adjacent Violators algorithm
    and linearly interpolate the fitted values for new probabilities.
    """
    pairs = sorted(zip(cal_probs, cal_labels))
    probs = [p for p, _ in pairs]
    labels = [float(y) for _, y in pairs]

    # PAVA: each block stores [sum, count, start, end]
    blocks = []

    for i, y in enumerate(labels):
        blocks.append([y, 1, i, i])

        # Merge while adjacent block means violate monotonicity.
        while len(blocks) >= 2:
            prev = blocks[-2]
            curr = blocks[-1]

            prev_mean = prev[0] / prev[1]
            curr_mean = curr[0] / curr[1]

            if prev_mean <= curr_mean:
                break

            merged = [
                prev[0] + curr[0],
                prev[1] + curr[1],
                prev[2],
                curr[3],
            ]

            blocks[-2:] = [merged]

    # Assign each calibration point its final block mean.
    fitted = [0.0] * len(labels)

    for total, count, start, end in blocks:
        mean = total / count
        for i in range(start, end + 1):
            fitted[i] = mean

    # Clamp outside range and linearly interpolate inside.
    result = []

    for p in new_probs:
        if p <= probs[0]:
            result.append(float(fitted[0]))
            continue

        if p >= probs[-1]:
            result.append(float(fitted[-1]))
            continue

        # Find interval containing p.
        for i in range(len(probs) - 1):
            if probs[i] <= p <= probs[i + 1]:
                x0, x1 = probs[i], probs[i + 1]
                y0, y1 = fitted[i], fitted[i + 1]

                if x1 == x0:
                    value = y0
                else:
                    value = y0 + (p - x0) * (y1 - y0) / (x1 - x0)

                result.append(float(value))
                break

    return result