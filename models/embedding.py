import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim): # embed dim is d_model
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

    def forward(self, x):
        return self.embedding(x)