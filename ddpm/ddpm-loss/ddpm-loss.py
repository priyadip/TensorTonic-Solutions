import numpy as np

def compute_ddpm_loss(x_0, betas, t_values, epsilon, epsilon_pred):
    """
    Returns: float scalar MSE loss between true noise and predicted noise
    """
    epsilon = np.asarray(epsilon, dtype=float)
    epsilon_pred = np.asarray(epsilon_pred, dtype=float)

    loss = np.mean((epsilon - epsilon_pred) ** 2)

    return float(loss)