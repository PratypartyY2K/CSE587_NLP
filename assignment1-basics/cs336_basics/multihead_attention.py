import math
import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        self.q_proj = nn.Parameter(torch.empty((d_model, d_model), **factory_kwargs))
        self.k_proj = nn.Parameter(torch.empty((d_model, d_model), **factory_kwargs))
        self.v_proj = nn.Parameter(torch.empty((d_model, d_model), **factory_kwargs))
        self.o_proj = nn.Parameter(torch.empty((d_model, d_model), **factory_kwargs))
        std = 1.0 / math.sqrt(d_model) if d_model > 0 else 1.0
        nn.init.trunc_normal_(self.q_proj, mean=0.0, std=std)
        nn.init.trunc_normal_(self.k_proj, mean=0.0, std=std)
        nn.init.trunc_normal_(self.v_proj, mean=0.0, std=std)
        nn.init.trunc_normal_(self.o_proj, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Input last dim must be d_model={self.d_model}")
        q = torch.matmul(x, self.q_proj.t())
        k = torch.matmul(x, self.k_proj.t())
        v = torch.matmul(x, self.v_proj.t())
        *lead_dims, seq_len, _ = q.shape
        q = q.view(*lead_dims, seq_len, self.num_heads, self.d_k).permute(*list(range(len(lead_dims))), -2, -3, -1)
        k = k.view(*lead_dims, seq_len, self.num_heads, self.d_k).permute(*list(range(len(lead_dims))), -2, -3, -1)
        v = v.view(*lead_dims, seq_len, self.num_heads, self.d_k).permute(*list(range(len(lead_dims))), -2, -3, -1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        device = scores.device
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
        while causal_mask.dim() < scores.dim():
            causal_mask = causal_mask.unsqueeze(0)
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(*list(range(len(lead_dims))), -2, -3, -1).contiguous()
        out = out.view(*lead_dims, seq_len, self.d_model)
        out = torch.matmul(out, self.o_proj.t())
        return out
