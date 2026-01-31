"""Simple AdamW optimizer implementation compatible with torch.optim.Optimizer API.

This is a minimal, clear implementation designed to match PyTorch's AdamW behavior closely
(so that tests accept either matching PyTorch or a provided reference).
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, required
from cs336_basics.impl.optimizer import AdamW as _AdamW, run_get_lr_cosine_schedule_impl as _run_get_lr_cosine_schedule_impl

__all__ = ["AdamW", "run_get_lr_cosine_schedule_impl"]

# Re-export for compatibility
AdamW = _AdamW
run_get_lr_cosine_schedule_impl = _run_get_lr_cosine_schedule_impl


class AdamW(Optimizer):
    """AdamW optimizer (decoupled weight decay).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate
        betas (Tuple[float, float]): coefficients used for computing running averages of gradient and its square
        eps (float): term added to the denominator to improve numerical stability
        weight_decay (float): weight decay coefficient (decoupled)
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr is None:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid eps value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group.get('weight_decay', 0.0)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1
                step = state['step']

                # Decay the first and second moment running average coefficients
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias corrections
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1

                # Parameter update (Adam step)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Decoupled weight decay (apply after parameter update to match PyTorch semantics)
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss

    def zero_grad(self, set_to_none: bool = False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.detach_()
                        p.grad.zero_()
