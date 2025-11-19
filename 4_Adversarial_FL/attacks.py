"""Attack primitives used by the adversarial FL labs."""

from __future__ import annotations

from typing import Callable, Dict

import torch

AttackFn = Callable[..., torch.Tensor]


def pgd_attack(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    step_size: float,
    iters: int,
    targeted: bool = False,
) -> torch.Tensor:
    """Projected Gradient Descent with an L-infinity constraint."""
    ori = images.clone().detach()
    adv = ori.clone().detach()

    for _ in range(iters):
        adv.requires_grad_(True)
        outputs = model(adv)
        loss = criterion(outputs, labels)

        model.zero_grad(set_to_none=True)
        loss.backward()

        direction = -1 if targeted else 1
        adv = adv + direction * step_size * adv.grad.sign()
        eta = torch.clamp(adv - ori, min=-eps, max=eps)
        adv = torch.clamp(ori + eta, 0, 1).detach()

    return adv


def fgsm_attack(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step_size: float,
    targeted: bool = False,
) -> torch.Tensor:
    """Single gradient-sign step."""
    adv = images.clone().detach().requires_grad_(True)
    loss = criterion(model(adv), labels)

    model.zero_grad(set_to_none=True)
    loss.backward()

    direction = -1 if targeted else 1
    adv = adv + direction * step_size * adv.grad.sign()
    return torch.clamp(adv, 0, 1).detach()


def random_noise_attack(images: torch.Tensor, step_size: float) -> torch.Tensor:
    """Add random signed noise to the batch and clamp to valid bounds."""
    perturb = torch.randn_like(images).sign()
    adv = images + step_size * perturb
    return torch.clamp(adv, 0, 1).detach()


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


__all__ = ["AttackFn", "fgsm_attack", "get_attack", "pgd_attack", "random_noise_attack"]
