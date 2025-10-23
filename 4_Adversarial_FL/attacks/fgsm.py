"""Fast Gradient Sign Method implementation."""

from __future__ import annotations

import torch


def fgsm_attack(
    model: torch.nn.Module,
    criterion,
    images: torch.Tensor,
    labels: torch.Tensor,
    step_size: float,
) -> torch.Tensor:
    """Craft adversarial examples by taking a single gradient sign step."""
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = criterion(outputs, labels)

    model.zero_grad()
    loss.backward()

    adv_images = images + step_size * images.grad.sign()
    return torch.clamp(adv_images, 0, 1).detach()


__all__ = ["fgsm_attack"]
