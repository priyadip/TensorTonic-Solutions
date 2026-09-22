def differencing(series: list, order: int) -> list:
    """ Returns the series after the requested differencing order. """
    
    for j in range(1, order + 1):
        ans = [0] * (len(series) - 1)

        for i in range(1, len(series)):
            ans[i - 1] = series[i] - series[i - 1]

        series = ans

    return series