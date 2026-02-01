import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit .
    x: scalar, list, or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    x = np.asarray(x, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * x * (1.0 + erf_vec(x / math.sqrt(2.0)))

