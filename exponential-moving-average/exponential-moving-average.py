def exponential_moving_average(values: list, alpha: float) -> list:
    """ Returns the exponential moving average at every position. """
    ans = [va
    for i in range(1,len(values)):
        ans.append(alpha*values[i] + (1-alpha)*ans[i-1])
    return ans
        
