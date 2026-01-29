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
        # projection weights stored as (out_features, in_features) to match tests
        # q,k,v projections have shape (d_model, d_model)
        self.q_proj = nn.Parameter(torch.empty((d_model, d_model), **factory_kwargs))
        self.k_proj = nn.Parameter(torch.empty((d_model, d_model), **factory_kwargs))
        self.v_proj = nn.Parameter(torch.empty((d_model, d_model), **factory_kwargs))
        # output projection
        self.o_proj = nn.Parameter(torch.empty((d_model, d_model), **factory_kwargs))
        # initialize weights
        std_q = 1.0 / math.sqrt(d_model) if d_model > 0 else 1.0
        std_k = 1.0 / math.sqrt(d_model) if d_model > 0 else 1.0
        std_v = 1.0 / math.sqrt(d_model) if d_model > 0 else 1.0
        std_o = 1.0 / math.sqrt(d_model) if d_model > 0 else 1.0
        nn.init.trunc_normal_(self.q_proj, mean=0.0, std=std_q)
        nn.init.trunc_normal_(self.k_proj, mean=0.0, std=std_k)
        nn.init.trunc_normal_(self.v_proj, mean=0.0, std=std_v)
        nn.init.trunc_normal_(self.o_proj, mean=0.0, std=std_o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_model)
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Input last dim must be d_model={self.d_model}")
        # project
        # Use linear: x @ W.T where W shape (d_model, d_model)
        q = torch.matmul(x, self.q_proj.t())
        k = torch.matmul(x, self.k_proj.t())
        v = torch.matmul(x, self.v_proj.t())
        # reshape to (..., seq_len, num_heads, d_k)
        *lead_dims, seq_len, _ = q.shape
        q = q.view(*lead_dims, seq_len, self.num_heads, self.d_k).permute(*list(range(len(lead_dims))), -2, -3, -1)
        # After permute, shape: (..., num_heads, seq_len, d_k)
        k = k.view(*lead_dims, seq_len, self.num_heads, self.d_k).permute(*list(range(len(lead_dims))), -2, -3, -1)
        v = v.view(*lead_dims, seq_len, self.num_heads, self.d_k).permute(*list(range(len(lead_dims))), -2, -3, -1)
        # compute scores: (..., num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        # causal mask: allow positions <= query index
        device = scores.device
        seq = seq_len
        causal_mask = torch.tril(torch.ones((seq, seq), dtype=torch.bool, device=device))
        # broadcast mask to scores shape
        while causal_mask.dim() < scores.dim():
            causal_mask = causal_mask.unsqueeze(0)
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (..., num_heads, seq_len, d_k)
        # permute back to (..., seq_len, num_heads, d_k)
        out = out.permute(*list(range(len(lead_dims))), -2, -3, -1).contiguous()
        # reshape to (..., seq_len, d_model)
        out = out.view(*lead_dims, seq_len, self.d_model)
        # final output projection
        out = torch.matmul(out, self.o_proj.t())
        return out
