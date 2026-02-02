import numpy as np

def lbfgs_direction(grad, s_list, y_list):
    """
    Compute the L-BFGS search direction using the two-loop recursion.

    Args:
        grad: current gradient (1D list or array)
        s_list: list of past step differences s_i
        y_list: list of past gradient differences y_i

    Returns:
        search direction (same shape as grad)
    """
    # Convert to numpy arrays
    q = np.array(grad, dtype=float)
    s_list = [np.array(s, dtype=float) for s in s_list]
    y_list = [np.array(y, dtype=float) for y in y_list]

    m = len(s_list)
    alpha = [0.0] * m
    rho = [0.0] * m

    # Precompute rho_i = 1 / (y_i^T s_i)
    for i in range(m):
        ys = np.dot(y_list[i], s_list[i])
        rho[i] = 1.0 / ys

    # ---- First loop (backward: newest → oldest) ----
    for i in range(m - 1, -1, -1):
        alpha[i] = rho[i] * np.dot(s_list[i], q)
        q = q - alpha[i] * y_list[i]

    # ---- Initial Hessian scaling ----
    if m > 0:
        y_last = y_list[-1]
        s_last = s_list[-1]
        gamma = np.dot(s_last, y_last) / np.dot(y_last, y_last)
    else:
        gamma = 1.0

    r = gamma * q

    # ---- Second loop (forward: oldest → newest) ----
    for i in range(m):
        beta = rho[i] * np.dot(y_list[i], r)
        r = r + s_list[i] * (alpha[i] - beta)

    # Descent direction
    return (-r).tolist()
