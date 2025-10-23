"""Random noise baseline attack."""

from __future__ import annotations

import torch


def random_noise_attack(images: torch.Tensor, step_size: float) -> torch.Tensor:
    """Add random signed noise to the batch and clamp to image bounds."""
    perturb = torch.randn_like(images)
    adv_images = images + step_size * perturb.sign()
    return torch.clamp(adv_images, 0, 1).detach()


__all__ = ["random_noise_attack"]
