"""Attack registry for adversarial federated learning experiments."""

from __future__ import annotations

from typing import Callable, Dict

import torch

from .fgsm import fgsm_attack
from .pgd import pgd_attack
from .random_noise import random_noise_attack

AttackFn = Callable[..., torch.Tensor]


def _build_registry() -> Dict[str, AttackFn]:
    registry: Dict[str, AttackFn] = {
        "fgsm": fgsm_attack,
        "pgd": pgd_attack,
        "rand_noise": random_noise_attack,
        "random_noise": random_noise_attack,
        "random": random_noise_attack,
    }
    return registry


ATTACK_REGISTRY: Dict[str, AttackFn] = _build_registry()


def get_attack(name: str) -> AttackFn:
    """Return a callable attack implementation from the registry."""
    key = name.lower()
    if key not in ATTACK_REGISTRY:
        available = ", ".join(sorted(set(ATTACK_REGISTRY)))
        raise ValueError(f"Unknown attack '{name}'. Available attacks: {available}")
    return ATTACK_REGISTRY[key]


__all__ = ["ATTACK_REGISTRY", "fgsm_attack", "get_attack", "pgd_attack", "random_noise_attack"]
