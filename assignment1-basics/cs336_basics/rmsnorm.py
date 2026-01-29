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
        # weight parameter with shape (d_model,)
        self.weight = nn.Parameter(torch.empty((d_model,), **factory_kwargs))
        # initialize to ones
        with torch.no_grad():
            init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        orig_dtype = x.dtype
        # Upcast to float32 for stable computation
        x_float = x.to(torch.float32)
        # compute rms across last dim
        rms = torch.sqrt(torch.mean(x_float * x_float, dim=-1, keepdim=True) + self.eps)
        x_norm = x_float / rms
        # apply affine weight: broadcast over leading dims
        w = self.weight
        # ensure weight is float32 for multiplication
        w_float = w.to(x_norm.dtype)
        out = x_norm * w_float
        # cast back to original dtype
        return out.to(orig_dtype)
