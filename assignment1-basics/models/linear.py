import math
import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        # store W with shape (out_features, in_features)
        self.W = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        # initialization: truncated normal with std = 1/sqrt(in_features)
        std = 1.0 / math.sqrt(in_features) if in_features > 0 else 1.0
        nn.init.trunc_normal_(self.W, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (..., in_features) -> output (..., out_features)
        return x.matmul(self.W.t())
