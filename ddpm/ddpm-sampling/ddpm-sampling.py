import numpy as np

def ddpm_sample(x_T, betas, epsilon_preds, z_values):
    x = np.asarray(x_T, dtype=float)
    betas = np.asarray(betas, dtype=float)
    epsilon_preds = np.asarray(epsilon_preds, dtype=float)
    z_values = np.asarray(z_values, dtype=float)

    T = len(betas)
    alpha_bars = np.cumprod(1.0 - betas)

    for i, t in enumerate(range(T, 0, -1)):
        beta_t = betas[t - 1]
        alpha_t = 1.0 - beta_t
        alpha_bar_t = alpha_bars[t - 1]

        epsilon_pred = epsilon_preds[i]

        x = (1 / np.sqrt(alpha_t)) * (x - (beta_t / np.sqrt(1 - alpha_bar_t)) * epsilon_pred)

        if t > 1:
            z = z_values[i]
            x = x + np.sqrt(beta_t) * z

    return np.round(x, 4)