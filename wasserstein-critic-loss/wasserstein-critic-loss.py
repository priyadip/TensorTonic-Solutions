import numpy as np

def wasserstein_critic_loss(real_scores: list, fake_scores: list) -> float:
    """ Returns the loss as a float. """
    r = np.asarray(real_scores)
    f = np.asarray(fake_scores)

    return float(np.mean(f) - np.mean(r))