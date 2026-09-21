from statistics import mean
def simple_moving_average(values: list, window_size: int) -> list:
    """Returns the mean of every complete sliding window."""
    # v = np.asarray(values)
    sma = []
    for i in range( len(values)-window_size+1):
        sma.append(mean(values[i:i+window_size]))
    return sma
        
