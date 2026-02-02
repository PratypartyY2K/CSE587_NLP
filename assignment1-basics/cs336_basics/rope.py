import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for rotary embeddings")
        self.theta = float(theta)
        self.d_k = int(d_k)
        self.max_seq_len = int(max_seq_len)
        self.half = d_k // 2
        inv_freq = torch.pow(self.theta, -2.0 * torch.arange(0, self.half, dtype=torch.float32) / float(self.d_k))
        positions = torch.arange(0, self.max_seq_len, dtype=torch.float32).unsqueeze(1)
        freqs = positions * inv_freq.unsqueeze(0)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        if device is not None:
            cos = cos.to(device)
            sin = sin.to(device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_k:
            raise ValueError(f"last dim of x must be d_k={self.d_k}, got {x.shape[-1]}")
        pos = torch.as_tensor(token_positions, dtype=torch.long, device=self.cos.device)
        cos_sel = self.cos[pos]
        sin_sel = self.sin[pos]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        lead_dims = x.dim() - 2
        if cos_sel.dim() == 2:
            cos_sel = cos_sel.view((1,) * lead_dims + cos_sel.shape)
            sin_sel = sin_sel.view((1,) * lead_dims + sin_sel.shape)
        out_even = x_even * cos_sel - x_odd * sin_sel
        out_odd = x_even * sin_sel + x_odd * cos_sel
        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out
