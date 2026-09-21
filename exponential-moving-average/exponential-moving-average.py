def exponential_moving_average(values: list, alpha: float) -> list:
    """ Returns the exponential moving average at every position. """
    ans = [0]*len(values)
    ans[0] = values[0]
    for i in range(1,len(values)):
        ans[i] = (alpha*values[i] + (1-alpha)*ans[i-1])
    return ans
        
