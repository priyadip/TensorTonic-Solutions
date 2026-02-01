import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer mapping vocab indices to d_model vectors.
    """
    return nn.Embedding(vocab_size, d_model)


def embed_tokens(
    embedding: nn.Embedding,
    tokens: torch.Tensor,
    d_model: int
) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    # Lookup embeddings
    emb = embedding(tokens)

    # Scale by sqrt(d_model)
    emb = emb * math.sqrt(d_model)

    return emb
