import numpy as np

def focal_loss(p: list, y: list, gamma: float = 2.0) -> float:
    """ Returns the loss as a float. """
    p = np.asarray(p)
    y = np.asarray(y)
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)

    
    loss = -(1 - p) ** gamma * y * np.log(p) \
           - p ** gamma * (1 - y) * np.log(1 - p)
    
    return float(np.mean(loss))