import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    # Convert inputs to numpy arrays
    w = np.array(w)
    g = np.array(g)
    s = np.array(s)

    # Update running average
    s = beta * s + (1 - beta) * (g ** 2)

    # Update parameters
    w = w - lr * g / (np.sqrt(s + eps))

    return w, s