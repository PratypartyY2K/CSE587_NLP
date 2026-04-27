from __future__ import annotations

import math

import torch


def _flash_attention_forward_tiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    *,
    q_tile_size: int = 32,
    k_tile_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if is_causal:
        raise NotImplementedError("Causal masking is intentionally not implemented for this pure PyTorch reference.")

    n_queries = q.shape[-2]
    n_keys = k.shape[-2]
    value_dim = v.shape[-1]
    scale = 1.0 / math.sqrt(q.shape[-1])

    output = torch.empty((*q.shape[:-1], value_dim), device=q.device, dtype=q.dtype)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=q.dtype)

    for q_start in range(0, n_queries, q_tile_size):
        q_end = min(q_start + q_tile_size, n_queries)
        q_block = q[..., q_start:q_end, :]

        running_max = torch.full(
            q_block.shape[:-1],
            float("-inf"),
            device=q.device,
            dtype=q.dtype,
        )
        running_l = torch.zeros(q_block.shape[:-1], device=q.device, dtype=q.dtype)
        running_o = torch.zeros(
            (*q_block.shape[:-1], value_dim),
            device=q.device,
            dtype=q.dtype,
        )

        for k_start in range(0, n_keys, k_tile_size):
            k_end = min(k_start + k_tile_size, n_keys)
            k_block = k[..., k_start:k_end, :]
            v_block = v[..., k_start:k_end, :]

            scores = torch.matmul(q_block, k_block.transpose(-1, -2)) * scale
            block_max = scores.max(dim=-1).values
            new_max = torch.maximum(running_max, block_max)

            exp_scores = torch.exp(scores - new_max.unsqueeze(-1))
            correction = torch.exp(running_max - new_max)

            running_l = correction * running_l + exp_scores.sum(dim=-1)
            running_o = correction.unsqueeze(-1) * running_o + torch.matmul(exp_scores, v_block)
            running_max = new_max

        output[..., q_start:q_end, :] = running_o / running_l.unsqueeze(-1)
        lse[..., q_start:q_end] = running_max + torch.log(running_l)

    return output, lse


class FlashAttentionPytorchFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        output, lse = _flash_attention_forward_tiled(q, k, v, is_causal=is_causal)
        ctx.save_for_backward(lse, q, k, v, output)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
        _lse, q, k, v, _output = ctx.saved_tensors

        with torch.enable_grad():
            q_ref = q.detach().requires_grad_(True)
            k_ref = k.detach().requires_grad_(True)
            v_ref = v.detach().requires_grad_(True)
            output_ref, _ = _flash_attention_forward_tiled(
                q_ref,
                k_ref,
                v_ref,
                is_causal=ctx.is_causal,
            )
            dq, dk, dv = torch.autograd.grad(output_ref, (q_ref, k_ref, v_ref), grad_output)

        return dq, dk, dv, None
