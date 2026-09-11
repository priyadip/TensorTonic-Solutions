import numpy as np

def info_nce_loss(Z1: list, Z2: list, temperature: float = 0.1) -> float:
    #Returns the loss as a float.
    z1 = np.asarray(Z1)
    z2 = np.asarray(Z2)

    s = (z1@z2.T)/temperature

    s -= np.max(s, axis = -1, keepdims = True)

    exps = np.exp(s)

    positive = np.diag(exps)

    denom = np.sum(exps, axis = -1)

    loss = -np.log(positive/denom)

    return float(np.mean(loss))
