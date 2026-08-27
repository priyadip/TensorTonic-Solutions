import numpy as np

def reverse_step(x_t, t, epsilon_pred, betas, z=None):
    """
    Returns: np.ndarray x_{t-1} after one reverse diffusion step
    """
    x_t = np.asarray(x_t, dtype=float)
    epsilon_pred = np.asarray(epsilon_pred, dtype=float)
    betas = np.asarray(betas, dtype=float)

    # alpha_t = 1 - beta_t
    beta_t = betas[t - 1]
    alpha_t = 1.0 - beta_t

    # Compute alpha_bar_t
    alpha_bar = np.cumprod(1.0 - betas)
    alpha_bar_t = alpha_bar[t - 1]

    # Posterior mean:
    # mu = 1/sqrt(alpha_t) *
    #      (x_t - beta_t/sqrt(1-alpha_bar_t) * epsilon_pred)
    mu = (
        1.0 / np.sqrt(alpha_t)
        * (
            x_t
            - (beta_t / np.sqrt(1.0 - alpha_bar_t))
            * epsilon_pred
        )
    )

    # At t=1, no random noise is added
    if t == 1:
        return mu

    # For t > 1, add stochastic noise
    if z is None:
        z = np.random.randn(*x_t.shape)
    else:
        z = np.asarray(z, dtype=float)

    x_prev = mu + np.sqrt(beta_t) * z

    return x_prev