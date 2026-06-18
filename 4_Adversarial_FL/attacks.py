"""Attack primitives used by the adversarial FL labs.

The dataloaders feed ImageNet-normalized tensors to MobileNetV2/V3.  The
public attack budgets, however, are workshop-friendly pixel-space values such
as 2/255, 4/255, and 8/255.  These helpers therefore denormalize to pixel
space for clipping/projection, then normalize again before model evaluation.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch

AttackFn = Callable[..., torch.Tensor]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _channel_tensor(values, ref: torch.Tensor) -> torch.Tensor:
    return torch.tensor(values, dtype=ref.dtype, device=ref.device).view(1, -1, 1, 1)


def normalize_pixels(
    images: torch.Tensor,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> torch.Tensor:
    return (images - _channel_tensor(mean, images)) / _channel_tensor(std, images)


def denormalize_pixels(
    images: torch.Tensor,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> torch.Tensor:
    return images * _channel_tensor(std, images) + _channel_tensor(mean, images)


def _to_pixel_space(
    images: torch.Tensor,
    normalized: bool,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> torch.Tensor:
    return denormalize_pixels(images, mean, std) if normalized else images


def _from_pixel_space(
    images: torch.Tensor,
    normalized: bool,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> torch.Tensor:
    return normalize_pixels(images, mean, std) if normalized else images


def _project_pixel_linf(
    original: torch.Tensor,
    adversarial: torch.Tensor,
    eps: float,
    normalized: bool,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> torch.Tensor:
    original_px = _to_pixel_space(original, normalized, mean, std)
    adversarial_px = _to_pixel_space(adversarial, normalized, mean, std)
    eta = torch.clamp(adversarial_px - original_px, min=-eps, max=eps)
    projected_px = torch.clamp(original_px + eta, 0.0, 1.0)
    return _from_pixel_space(projected_px, normalized, mean, std)


def pixel_linf_norm(
    original: torch.Tensor,
    adversarial: torch.Tensor,
    normalized: bool = True,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> torch.Tensor:
    """Return the per-example L-infinity perturbation in pixel space."""
    original_px = _to_pixel_space(original, normalized, mean, std)
    adversarial_px = _to_pixel_space(adversarial, normalized, mean, std)
    diff = (adversarial_px - original_px).detach().abs()
    return diff.flatten(start_dim=1).amax(dim=1)


def pgd_attack(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    step_size: float,
    iters: int,
    targeted: bool = False,
    normalized: bool = True,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> torch.Tensor:
    """Projected Gradient Descent with a pixel-space L-infinity constraint."""
    ori = images.clone().detach()
    adv = ori.clone().detach()

    for _ in range(iters):
        adv.requires_grad_(True)
        outputs = model(adv)
        loss = criterion(outputs, labels)

        model.zero_grad(set_to_none=True)
        loss.backward()

        direction = -1 if targeted else 1
        adv_px = _to_pixel_space(adv.detach(), normalized, mean, std)
        adv_px = adv_px + direction * step_size * adv.grad.sign()
        adv = _from_pixel_space(torch.clamp(adv_px, 0.0, 1.0), normalized, mean, std)
        adv = _project_pixel_linf(ori, adv, eps, normalized, mean, std).detach()

    return adv


def fgsm_attack(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step_size: float,
    targeted: bool = False,
    normalized: bool = True,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> torch.Tensor:
    """Single gradient-sign step using a pixel-space step size."""
    adv = images.clone().detach().requires_grad_(True)
    loss = criterion(model(adv), labels)

    model.zero_grad(set_to_none=True)
    loss.backward()

    direction = -1 if targeted else 1
    adv_px = _to_pixel_space(adv.detach(), normalized, mean, std)
    adv_px = adv_px + direction * step_size * adv.grad.sign()
    adv_px = torch.clamp(adv_px, 0.0, 1.0)
    return _from_pixel_space(adv_px, normalized, mean, std).detach()


def random_noise_attack(
    images: torch.Tensor,
    step_size: float,
    normalized: bool = True,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> torch.Tensor:
    """Add uniform pixel-space L-infinity noise and clamp to valid bounds."""
    images_px = _to_pixel_space(images, normalized, mean, std)
    perturb = torch.empty_like(images_px).uniform_(-step_size, step_size)
    adv_px = torch.clamp(images_px + perturb, 0.0, 1.0)
    return _from_pixel_space(adv_px, normalized, mean, std).detach()


ATTACK_FUNCTIONS: Dict[str, AttackFn] = {
    "pgd": pgd_attack,
    "fgsm": fgsm_attack,
    "random": random_noise_attack,
    "random_noise": random_noise_attack,
}


def get_attack(name: str) -> AttackFn:
    key = name.lower()
    if key not in ATTACK_FUNCTIONS:
        raise KeyError(f"Unknown attack '{name}'. Available: {sorted(ATTACK_FUNCTIONS)}")
    return ATTACK_FUNCTIONS[key]


__all__ = [
    "AttackFn",
    "denormalize_pixels",
    "fgsm_attack",
    "get_attack",
    "normalize_pixels",
    "pgd_attack",
    "pixel_linf_norm",
    "random_noise_attack",
]
