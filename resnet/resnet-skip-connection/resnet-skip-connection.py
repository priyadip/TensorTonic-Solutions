import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    Each layer has Jacobian (J + I).
    """
    g = np.asarray(x)

    for J in gradients_F:
        J = np.asarray(J)
        g = g @ (J + np.eye(J.shape[0]))

    return g


def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    """
    g = np.asarray(x)

    for J in gradients_F:
        J = np.asarray(J)
        g = g @ J

    return g