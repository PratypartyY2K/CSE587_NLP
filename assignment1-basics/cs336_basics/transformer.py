import math
import torch
from torch import nn

from .embedding import Embedding
from .rope import RotaryPositionalEmbedding
from .rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = float(theta)

        factory_kwargs = {}
        self.q_proj = nn.Parameter(torch.empty((d_model, d_model), **factory_kwargs))
        self.k_proj = nn.Parameter(torch.empty((d_model, d_model), **factory_kwargs))
        self.v_proj = nn.Parameter(torch.empty((d_model, d_model), **factory_kwargs))
        self.o_proj = nn.Parameter(torch.empty((d_model, d_model), **factory_kwargs))

        self.ln1 = RMSNorm(d_model=d_model)
        self.ln2 = RMSNorm(d_model=d_model)

        self.w1 = nn.Parameter(torch.empty((d_ff, d_model), **factory_kwargs))
        self.w2 = nn.Parameter(torch.empty((d_model, d_ff), **factory_kwargs))
        self.w3 = nn.Parameter(torch.empty((d_ff, d_model), **factory_kwargs))

        std = 1.0 / math.sqrt(d_model) if d_model > 0 else 1.0
        nn.init.trunc_normal_(self.q_proj, mean=0.0, std=std)
        nn.init.trunc_normal_(self.k_proj, mean=0.0, std=std)
        nn.init.trunc_normal_(self.v_proj, mean=0.0, std=std)
        nn.init.trunc_normal_(self.o_proj, mean=0.0, std=std)

        std_w1 = 1.0 / math.sqrt(d_model) if d_model > 0 else 1.0
        std_w2 = 1.0 / math.sqrt(d_ff) if d_ff > 0 else 1.0
        std_w3 = 1.0 / math.sqrt(d_model) if d_model > 0 else 1.0
        nn.init.trunc_normal_(self.w1, mean=0.0, std=std_w1)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=std_w2)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=std_w3)

        self.rope = RotaryPositionalEmbedding(theta=self.theta, d_k=self.d_k, max_seq_len=self.max_seq_len)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        device = x.device
        seq_len = x.shape[-2]
        if token_positions is None:
            pos = torch.arange(0, seq_len, device=device)
        else:
            pos = token_positions

        x_ln = self.ln1(x)

        q = torch.matmul(x_ln, self.q_proj.t())
        k = torch.matmul(x_ln, self.k_proj.t())
        v = torch.matmul(x_ln, self.v_proj.t())

        *lead_dims, s_len, _ = q.shape
        q = q.view(*lead_dims, s_len, self.num_heads, self.d_k).permute(*list(range(len(lead_dims))), -2, -3, -1)
        k = k.view(*lead_dims, s_len, self.num_heads, self.d_k).permute(*list(range(len(lead_dims))), -2, -3, -1)
        v = v.view(*lead_dims, s_len, self.num_heads, self.d_k).permute(*list(range(len(lead_dims))), -2, -3, -1)

        q = self.rope(q, pos)
        k = self.rope(k, pos)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        causal = torch.tril(torch.ones((s_len, s_len), dtype=torch.bool, device=device))
        while causal.dim() < scores.dim():
            causal = causal.unsqueeze(0)
        scores = scores.masked_fill(~causal, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.permute(*list(range(len(lead_dims))), -2, -3, -1).contiguous()
        out = out.view(*lead_dims, s_len, self.d_model)
        out = torch.matmul(out, self.o_proj.t())

        x = x + out

        x_ln2 = self.ln2(x)
        u = torch.matmul(x_ln2, self.w1.t())
        vff = torch.matmul(x_ln2, self.w3.t())
        gate = u * torch.sigmoid(u)
        ff = gate * vff
        ff_out = torch.matmul(ff, self.w2.t())
        x = x + ff_out
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.context_length = int(context_length)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)
        self.rope_theta = float(rope_theta)

        self.token_embeddings = Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_ff,
                    max_seq_len=self.context_length,
                    theta=self.rope_theta,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model=self.d_model)
        self.lm_head = nn.Parameter(torch.empty((self.vocab_size, self.d_model)))
        nn.init.trunc_normal_(self.lm_head, mean=0.0, std=1.0 / math.sqrt(self.d_model))

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(in_indices)
        seq_len = x.shape[-2]
        pos = torch.arange(0, seq_len, device=x.device)
        for block in self.blocks:
            x = block(x, token_positions=pos)
        x = self.ln_final(x)
        logits = torch.matmul(x, self.lm_head.t())
        return logits
