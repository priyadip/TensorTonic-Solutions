import numpy as np
def binary_focal_loss(predictions: list, targets: list, alpha: float, gamma: float) -> float:
    """Returns the mean binary focal loss as a float."""
    p = np.asarray(predictions)
    t = np.asarray(targets)

    pt = np.where(t==1, p, 1-p)

    l = - alpha * ((1 - pt)**gamma) * np.log(pt)

    return float(np.mean(l))