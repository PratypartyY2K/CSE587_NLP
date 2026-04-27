from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - Triton is unavailable on macOS dev machines.
    triton = None
    tl = None


def _flash_attention_forward_tiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    *,
    q_tile_size: int = 32,
    k_tile_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
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
            if is_causal:
                q_positions = torch.arange(q_start, q_end, device=q.device).unsqueeze(-1)
                k_positions = torch.arange(k_start, k_end, device=q.device).unsqueeze(0)
                scores = torch.where(q_positions >= k_positions, scores, torch.full_like(scores, -1e6))
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


def _flash_attention_backward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    grad_output: torch.Tensor,
    lse: torch.Tensor,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale

    if is_causal:
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]
        q_positions = torch.arange(n_queries, device=q.device).unsqueeze(-1)
        k_positions = torch.arange(n_keys, device=q.device).unsqueeze(0)
        scores = torch.where(q_positions >= k_positions, scores, torch.full_like(scores, -1e6))

    p = torch.exp(scores - lse.unsqueeze(-1))
    dv = torch.matmul(p.transpose(-1, -2), grad_output)
    dp = torch.matmul(grad_output, v.transpose(-1, -2))
    d_vec = torch.sum(o * grad_output, dim=-1)
    ds = p * (dp - d_vec.unsqueeze(-1))
    dq = torch.matmul(ds, k) * scale
    dk = torch.matmul(ds.transpose(-1, -2), q) * scale
    return dq, dk, dv


if hasattr(torch, "compile"):
    _flash_attention_backward_compiled = torch.compile(_flash_attention_backward_reference, backend="eager")
else:  # pragma: no cover - present for older PyTorch compatibility.
    _flash_attention_backward_compiled = _flash_attention_backward_reference


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
        lse, q, k, v, output = ctx.saved_tensors
        dq, dk, dv = _flash_attention_backward_compiled(
            q,
            k,
            v,
            output,
            grad_output,
            lse,
            ctx.is_causal,
        )
        return dq, dk, dv, None


if triton is not None:
    @triton.jit
    def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        is_causal: tl.constexpr,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
    ):
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        Q_block_ptr = tl.make_block_ptr(
            base=Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        q = tl.load(Q_block_ptr)
        m_i = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

        q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        k_offsets = tl.arange(0, K_TILE_SIZE)

        for _ in range(0, N_KEYS, K_TILE_SIZE):
            k = tl.load(K_block_ptr)
            v = tl.load(V_block_ptr)

            scores = tl.dot(q, tl.trans(k)) * scale
            if is_causal:
                causal_mask = q_offsets[:, None] >= k_offsets[None, :]
                scores = scores + tl.where(causal_mask, 0.0, -1e6)

            scores_f32 = scores.to(tl.float32)
            m_ij = tl.maximum(m_i, tl.max(scores_f32, axis=1))
            p_tilde = tl.exp(scores_f32 - m_ij[:, None])
            alpha = tl.exp(m_i - m_ij)

            l_i = alpha * l_i + tl.sum(p_tilde, axis=1)
            o_i = o_i * alpha[:, None]
            o_i = tl.dot(p_tilde.to(v.dtype), v, acc=o_i)
            m_i = m_ij

            K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
            V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
            k_offsets += K_TILE_SIZE

        o_i = o_i / l_i[:, None]
        lse = m_i + tl.log(l_i)

        tl.store(O_block_ptr, o_i.to(O_block_ptr.type.element_ty))
        l_ptrs = L_ptr + batch_index * stride_lb + q_offsets * stride_lq
        tl.store(l_ptrs, lse.to(L_ptr.type.element_ty))


def _flash_attention_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    *,
    q_tile_size: int = 32,
    k_tile_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is not installed in this environment.")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise RuntimeError("FlashAttention Triton forward requires CUDA tensors.")
    if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
        raise ValueError("This reference Triton kernel expects Q, K, and V to share the same head dimension.")

    batch_size, n_queries, d = q.shape
    n_keys = k.shape[-2]
    scale = 1.0 / math.sqrt(d)

    output = torch.empty_like(v[:, :n_queries, :])
    lse = torch.empty((batch_size, n_queries), device=q.device, dtype=q.dtype)

    grid = (triton.cdiv(n_queries, q_tile_size), batch_size)
    flash_fwd_kernel[grid](
        q, k, v,
        output, lse,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        n_queries, n_keys,
        scale,
        is_causal=is_causal,
        D=d,
        Q_TILE_SIZE=q_tile_size,
        K_TILE_SIZE=k_tile_size,
    )
    return output, lse


class FlashAttentionTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        output, lse = _flash_attention_forward_triton(q, k, v, is_causal=is_causal)
        ctx.save_for_backward(lse, q, k, v, output)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
        lse, q, k, v, output = ctx.saved_tensors
        dq, dk, dv = _flash_attention_backward_compiled(
            q,
            k,
            v,
            output,
            grad_output,
            lse,
            ctx.is_causal,
        )
        return dq, dk, dv, None
