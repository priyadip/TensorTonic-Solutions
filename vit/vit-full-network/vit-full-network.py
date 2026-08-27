import numpy as np

class VisionTransformer:
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        W_patch=None,
        cls_token=None,
        pos_embed=None,
        encoder_weights=None,
        W_head=None
    ):
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes

        hidden_dim = int(embed_dim * mlp_ratio)

        def init(shape):
            return np.random.randn(*shape) * 0.02

        # Patch projection: RGB input -> embedding dimension
        self.W_patch = (
            W_patch
            if W_patch is not None
            else init((patch_size * patch_size * 3, embed_dim))
        )

        self.cls_token = (
            cls_token
            if cls_token is not None
            else init((1, 1, embed_dim))
        )

        # One position for every patch plus CLS.
        self.pos_embed = (
            pos_embed
            if pos_embed is not None
            else init((1, self.num_patches + 1, embed_dim))
        )

        # Store one dictionary of weights per encoder block.
        if encoder_weights is not None:
            self.encoder_weights = encoder_weights
        else:
            self.encoder_weights = []

            for _ in range(depth):
                self.encoder_weights.append({
                    "Wq": init((embed_dim, embed_dim)),
                    "Wk": init((embed_dim, embed_dim)),
                    "Wv": init((embed_dim, embed_dim)),
                    "Wo": init((embed_dim, embed_dim)),
                    "W1": init((embed_dim, hidden_dim)),
                    "W2": init((hidden_dim, embed_dim)),
                })

        self.W_head = (
            W_head
            if W_head is not None
            else init((embed_dim, num_classes))
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, H, W, C = x.shape
        P = self.patch_size

        # -------------------------
        # Patch embedding
        # -------------------------
        n_h = H // P
        n_w = W // P

        patches = x.reshape(B, n_h, P, n_w, P, C)
        patches = patches.transpose(0, 1, 3, 2, 4, 5)
        patches = patches.reshape(B, n_h * n_w, P * P * C)

        z = patches @ self.W_patch

        # -------------------------
        # CLS token
        # -------------------------
        cls = np.tile(self.cls_token, (B, 1, 1))
        z = np.concatenate((cls, z), axis=1)

        # -------------------------
        # Position embedding
        # -------------------------
        z = z + self.pos_embed

        # -------------------------
        # Helpers
        # -------------------------
        def layer_norm(a):
            mean = np.mean(a, axis=-1, keepdims=True)
            var = np.var(a, axis=-1, keepdims=True)
            return (a - mean) / np.sqrt(var + 1e-6)

        def softmax(a):
            a = a - np.max(a, axis=-1, keepdims=True)
            e = np.exp(a)
            return e / np.sum(e, axis=-1, keepdims=True)

        def gelu(a):
            return 0.5 * a * (
                1.0 + np.tanh(
                    np.sqrt(2.0 / np.pi) *
                    (a + 0.044715 * a**3)
                )
            )

        # -------------------------
        # Transformer blocks
        # -------------------------
        for weights in self.encoder_weights:

            # Pre-LayerNorm
            normalized = layer_norm(z)

            Q = normalized @ weights["Wq"]
            K = normalized @ weights["Wk"]
            V = normalized @ weights["Wv"]

            seq_len = z.shape[1]
            head_dim = self.embed_dim // self.num_heads

            # Split heads
            Q = Q.reshape(
                B, seq_len, self.num_heads, head_dim
            ).transpose(0, 2, 1, 3)

            K = K.reshape(
                B, seq_len, self.num_heads, head_dim
            ).transpose(0, 2, 1, 3)

            V = V.reshape(
                B, seq_len, self.num_heads, head_dim
            ).transpose(0, 2, 1, 3)

            # Attention
            scores = (
                Q @ K.transpose(0, 1, 3, 2)
            ) / np.sqrt(head_dim)

            attention = softmax(scores)
            attended = attention @ V

            # Merge heads
            attended = attended.transpose(0, 2, 1, 3)
            attended = attended.reshape(
                B, seq_len, self.embed_dim
            )

            # First residual
            z_prime = z + attended @ weights["Wo"]

            # Pre-LN MLP
            normalized = layer_norm(z_prime)
            hidden = normalized @ weights["W1"]
            hidden = gelu(hidden)
            mlp = hidden @ weights["W2"]

            # Second residual
            z = z_prime + mlp

        # -------------------------
        # Classification head
        # -------------------------
        cls_output = z[:, 0, :]
        cls_output = layer_norm(cls_output)

        return cls_output @ self.W_head