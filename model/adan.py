import math

import torch
from torch.optim import Optimizer


def exists(val):
    return val is not None


class Adan(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.02, 0.08, 0.01),
        eps=1e-8,
        weight_decay=0,
        restart_cond: callable = None,
    ):
        assert len(betas) == 3

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            restart_cond=restart_cond,
        )

        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if exists(closure):
            loss = closure()

        for group in self.param_groups:

            lr = group["lr"]
            beta1, beta2, beta3 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            restart_cond = group["restart_cond"]

            for p in group["params"]:
                if not exists(p.grad):
                    continue

                data, grad = p.data, p.grad.data
                assert not grad.is_sparse

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["prev_grad"] = torch.zeros_like(grad)
                    state["m"] = torch.zeros_like(grad)
                    state["v"] = torch.zeros_like(grad)
                    state["n"] = torch.zeros_like(grad)

                step, m, v, n, prev_grad = (
                    state["step"],
                    state["m"],
                    state["v"],
                    state["n"],
                    state["prev_grad"],
                )

                if step > 0:
                    prev_grad = state["prev_grad"]

                    # main algorithm

                    m.mul_(1 - beta1).add_(grad, alpha=beta1)

                    grad_diff = grad - prev_grad

                    v.mul_(1 - beta2).add_(grad_diff, alpha=beta2)

                    next_n = (grad + (1 - beta2) * grad_diff) ** 2

                    n.mul_(1 - beta3).add_(next_n, alpha=beta3)

                # bias correction terms

                step += 1

                correct_m, correct_v, correct_n = map(
                    lambda n: 1 / (1 - (1 - n) ** step), (beta1, beta2, beta3)
                )

                # gradient step

                def grad_step_(data, m, v, n):
                    weighted_step_size = lr / (n * correct_n).sqrt().add_(eps)

                    denom = 1 + weight_decay * lr

                    data.addcmul_(
                        weighted_step_size,
                        (m * correct_m + (1 - beta2) * v * correct_v),
                        value=-1.0,
                    ).div_(denom)

                grad_step_(data, m, v, n)

                # restart condition

                if exists(restart_cond) and restart_cond(state):
                    m.data.copy_(grad)
                    v.zero_()
                    n.data.copy_(grad ** 2)

                    grad_step_(data, m, v, n)

                # set new incremented step

                prev_grad.copy_(grad)
                state["step"] = step

        return loss
