import math
import torch
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        # weights shapes:
        # w1: (d_ff, d_model)
        # w2: (d_model, d_ff)
        # w3: (d_ff, d_model)
        self.w1 = nn.Parameter(torch.empty((d_ff, d_model), **factory_kwargs))
        self.w2 = nn.Parameter(torch.empty((d_model, d_ff), **factory_kwargs))
        self.w3 = nn.Parameter(torch.empty((d_ff, d_model), **factory_kwargs))
        # initialize weights with truncated normal similar to Linear
        std_w1 = 1.0 / math.sqrt(d_model) if d_model > 0 else 1.0
        std_w2 = 1.0 / math.sqrt(d_ff) if d_ff > 0 else 1.0
        std_w3 = 1.0 / math.sqrt(d_model) if d_model > 0 else 1.0
        nn.init.trunc_normal_(self.w1, mean=0.0, std=std_w1)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=std_w2)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=std_w3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        # project to two d_ff tensors
        u = x.matmul(self.w1.t())  # (..., d_ff)
        v = x.matmul(self.w3.t())  # (..., d_ff)
        # SwiGLU: SiLU(u) * v. Use x * sigmoid(x) for numerical stability
        gate = u * torch.sigmoid(u)
        activated = gate * v
        out = activated.matmul(self.w2.t())  # (..., d_model)
        return out
