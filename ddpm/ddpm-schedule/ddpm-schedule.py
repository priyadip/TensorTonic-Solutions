import numpy as np

def linear_beta_schedule(T, beta_1=0.0001, beta_T=0.02):
    """
    Linear noise schedule from beta_1 to beta_T.
    Returns list of floats rounded to 6 decimals.
    """
    betas = np.linspace(beta_1, beta_T, T)
    return np.round(betas, 6)


def cosine_alpha_bar_schedule(T, s=0.008):
    """
    Cosine schedule for alpha_bar (cumulative signal retention).
    Returns an array of floats rounded to 6 decimals,
    clipped to [0.0001, 0.9999].
    """
    t = np.arange(1, T + 1)

    alpha_bars = (
        np.cos(
            ((t / T) + s) / (1 + s) * np.pi / 2
        ) ** 2
    )

    # Normalize so alpha_bar starts from 1
    alpha_bars = alpha_bars / np.cos(
        (s / (1 + s)) * np.pi / 2
    ) ** 2

    alpha_bars = np.clip(alpha_bars, 0.0001, 0.9999)

    return np.round(alpha_bars, 6)


def alpha_bar_to_betas(alpha_bars):
    """
    Convert alpha_bar schedule to beta schedule.
    Returns array of floats rounded to 6 decimals,
    clipped to [0.0001, 0.9999].
    """
    alpha_bars = np.asarray(alpha_bars, dtype=float)

    # alpha_bar_0 is conceptually 1
    previous = np.concatenate(([1.0], alpha_bars[:-1]))

    betas = 1.0 - (alpha_bars / previous)

    betas = np.clip(betas, 0.0001, 0.9999)

    return np.round(betas, 6)