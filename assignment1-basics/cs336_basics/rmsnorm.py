import torch
from torch import nn
import torch.nn.init as init


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        self.weight = nn.Parameter(torch.empty((d_model,), **factory_kwargs))
        with torch.no_grad():
            init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_float = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x_float * x_float, dim=-1, keepdim=True) + self.eps)
        x_norm = x_float / rms
        w = self.weight
        w_float = w.to(x_norm.dtype)
        out = x_norm * w_float
        return out.to(orig_dtype)
