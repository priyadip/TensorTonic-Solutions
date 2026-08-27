import numpy as np

def get_alpha_bar(betas):
    betas = np.asarray(betas, dtype=float)
    return np.cumprod(1 - betas)


def forward_diffusion(x_0, t, betas, epsilon):
    x_0 = np.asarray(x_0, dtype=float)
    epsilon = np.asarray(epsilon, dtype=float)
    betas = np.asarray(betas, dtype=float)

    alpha_bar_t = get_alpha_bar(betas)[t - 1]

    x_t = (
        np.sqrt(alpha_bar_t) * x_0
        + np.sqrt(1 - alpha_bar_t) * epsilon
    )

    return np.round(x_t, 4).tolist()