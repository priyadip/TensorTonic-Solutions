from statistics import mean
import numpy as np
def autocorrelation(series: list, max_lag: int) -> list:
    ' Returns normalized autocorrelation from lag zero through max_lag.'
    # n = len(series)
    # gama = 0
    # r = [1]
    # sm = mean(series)
    # for i in range(n):
    #     gama += (sm - series[i])**2
    # for k in range(1, max_lag+1):
    #     result = 0
    #     for j in range(n-k):
    #         auto = (series[j] - sm)*(series[j+k] - sm)
    #         result += auto
    #     result /= gama
    #     result = round(result, 6)
    #     r.append(result)
    # return r



    x = np.asarray(series, dtype=float)
    x = x - np.mean(x)
    gamma = np.sum(x ** 2)
    r = [1.0]
    
    if gamma == 0:
        return r + [0.0] * max_lag
    
    for k in range(1, max_lag + 1):
        result = np.sum(x[:-k] * x[k:]) / gamma
        r.append(round(result, 6))
    return r
        
