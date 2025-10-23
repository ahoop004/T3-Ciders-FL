"""Projected Gradient Descent attack implementation."""

from __future__ import annotations

import torch


def pgd_attack(
    model: torch.nn.Module,
    criterion,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    step_size: float,
    iters: int,
) -> torch.Tensor:
    """Run an iterative PGD attack with an L-infinity constraint."""
    ori = images.clone().detach()
    adv = ori.clone().detach()

    for _ in range(iters):
        adv.requires_grad_(True)
        outputs = model(adv)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        adv = adv + step_size * adv.grad.sign()
        eta = torch.clamp(adv - ori, min=-eps, max=eps)
        adv = torch.clamp(ori + eta, 0, 1).detach()

    return adv


__all__ = ["pgd_attack"]
