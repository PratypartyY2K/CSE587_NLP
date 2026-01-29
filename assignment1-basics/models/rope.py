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
        # compute inverse frequencies: 1 / theta^(2i/d_k)
        inv_freq = torch.pow(self.theta, -2.0 * torch.arange(0, self.half, dtype=torch.float32) / float(self.d_k))
        # positions 0..max_seq_len-1
        positions = torch.arange(0, self.max_seq_len, dtype=torch.float32).unsqueeze(1)
        freqs = positions * inv_freq.unsqueeze(0)  # (max_seq_len, half)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        if device is not None:
            cos = cos.to(device)
            sin = sin.to(device)
        # register buffers so they move with the module/device
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len) or (seq_len,) or (1, seq_len)
        if x.shape[-1] != self.d_k:
            raise ValueError(f"last dim of x must be d_k={self.d_k}, got {x.shape[-1]}")
        seq_len = x.shape[-2]
        # normalize token_positions to long tensor on same device as cos
        pos = torch.as_tensor(token_positions, dtype=torch.long, device=self.cos.device)
        # If pos has shape (seq_len,), expand to match leading dims of x excluding last dim if needed
        # We will index cos/sin with pos directly; cos has shape (max_seq_len, half)
        # After indexing, cos_indexed shape == pos.shape + (half,)
        cos_sel = self.cos[pos]
        sin_sel = self.sin[pos]
        # Now split x into even and odd components
        x_shape = x.shape
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        # Ensure cos_sel/sin_sel broadcast to x_even/x_odd
        # cos_sel has shape (..., seq_len, half) where leading dims match pos's leading dims
        # If pos was 1D (seq_len,), cos_sel shape = (seq_len, half); we need to align it to x_even's leading dims
        # Use unsqueeze to add missing leading dims
        # Determine number of leading dims before seq dim in x
        lead_dims = x.dim() - 2
        # cos_sel.dim() may be 2 or more; if it's 2 (seq_len, half), expand to (1,...,1, seq_len, half)
        if cos_sel.dim() == 2:
            # add lead_dims number of singleton dims at front
            cos_sel = cos_sel.view((1,) * lead_dims + cos_sel.shape)
            sin_sel = sin_sel.view((1,) * lead_dims + sin_sel.shape)
        # Now broadcast multiply
        # x_even, x_odd shape: (..., seq_len, half)
        out_even = x_even * cos_sel - x_odd * sin_sel
        out_odd = x_even * sin_sel + x_odd * cos_sel
        # interleave even and odd back
        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out
