"""Optimizer helpers: AdamW and LR schedule implementations."""
from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class AdamW(Optimizer):
    """AdamW optimizer (decoupled weight decay).

    Minimal implementation compatible with torch.optim.Optimizer API used in tests.
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

    def step(self, closure=None):
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

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1
                step = state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

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


def run_get_lr_cosine_schedule_impl(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """Linear warmup followed by cosine annealing schedule."""
    t = int(it)
    alpha_max = float(max_learning_rate)
    alpha_min = float(min_learning_rate)
    Tw = int(warmup_iters)
    Tc = int(cosine_cycle_iters)

    if Tw <= 0:
        if Tc <= 0 or t >= Tc:
            return alpha_min
        num_cosine_steps = max(1, Tc)
        progress = max(0, t)
        cos_val = 0.5 * (1.0 + math.cos(math.pi * progress / num_cosine_steps))
        return alpha_min + (alpha_max - alpha_min) * cos_val

    if t <= Tw:
        return alpha_max * (t / Tw)

    if t <= Tc:
        num_cosine_steps = max(1, Tc - Tw)
        progress = t - Tw
        cos_val = 0.5 * (1.0 + math.cos(math.pi * progress / num_cosine_steps))
        return alpha_min + (alpha_max - alpha_min) * cos_val

    return alpha_min
