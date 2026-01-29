import math
import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        # store weight with shape (num_embeddings, embedding_dim)
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        # initialization: truncated normal with std = 1/sqrt(embedding_dim)
        std = 1.0 / math.sqrt(embedding_dim) if embedding_dim > 0 else 1.0
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Ensure token_ids is a long tensor on the same device as weight
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.weight.device)
        return self.weight[ids]
