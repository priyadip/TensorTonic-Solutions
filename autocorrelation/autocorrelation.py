import numpy as np

def autocorrelation(series: list, max_lag: int) -> list:
    """
    Returns normalized autocorrelation from lag zero through max_lag.
    """
    x = np.asarray(series, dtype=float)
    mean = np.mean(x)
    centered = x - mean

    # Total variance (unnormalized).
    variance = np.sum(centered ** 2)

    if variance == 0:
        return [1.0] + [0.0] * max_lag

    result = [1.0]

    for lag in range(1, max_lag + 1):
        covariance = np.sum(
            centered[:-lag] * centered[lag:]
        )
        result.append(round(float(covariance / variance), 6))

    return result