import numpy as np

def relu(x):
    return np.maximum(0, x)

class IdentityBlock:
    """
    Identity Block: F(x) + x
    Used when input and output dimensions match.
    """
    
    def __init__(self, channels: int):
        self.channels = channels
        # Simplified: using dense layers instead of conv for demo
        self.W1 = np.random.randn(channels, channels) * 0.01
        self.W2 = np.random.randn(channels, channels) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = ReLU(W2 @ ReLU(W1 @ x)) + x
        Supports both:
        - x shape (batch, channels)
        - x shape (batch, channels, height, width)
        """
        if x.ndim == 2:
            # ----- Dense case -----
            # x: (batch, channels)
            z1 = relu(x @ self.W1.T)
            z2 = relu(z1 @ self.W2.T)
            return z2 + x

        elif x.ndim == 4:
            # ----- CNN-style case -----
            b, c, h, w = x.shape

            # Flatten spatial dims
            x_flat = x.reshape(b, c, -1)  # (b, c, h*w)

            # Linear + ReLU
            z1 = relu(np.matmul(self.W1, x_flat))
            z2 = relu(np.matmul(self.W2, z1))

            # Restore shape
            z2 = z2.reshape(b, c, h, w)

            return z2 + x

        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")


