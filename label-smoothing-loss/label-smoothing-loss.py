import numpy as np

def label_smoothing_loss(predictions: list, target: int, epsilon: float) -> float:
    """Returns cross-entropy loss for the smoothed target distribution."""
    
    p = np.asarray(predictions)
    q = np.full(k := p.shape[0], epsilon/k)

    q[target] += (1 - epsilon)

    return - float(np.sum(q*np.log(p)))
    
    

    
    