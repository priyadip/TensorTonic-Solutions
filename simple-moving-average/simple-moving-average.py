import numpy as np
def simple_moving_average(values: list, window_size: int) -> list:
    """Returns the mean of every complete sliding window."""
    v = np.asarray(values)
    sma = np.convolve(v, np.ones(window_size)*(1/window_size), mode = 'valid').tolist()

    return sma

        
