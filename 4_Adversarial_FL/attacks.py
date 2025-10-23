"""Backwards-compatible shims for attack utilities.

The real implementations now live under ``4_Adversarial_FL.attacks``.
"""

from __future__ import annotations

import warnings

from .attacks import fgsm_attack as _fgsm_attack
from .attacks import pgd_attack as _pgd_attack
from .attacks import random_noise_attack as _random_noise_attack


def _warn_deprecated() -> None:
    warnings.warn(
        "Importing attacks from '4_Adversarial_FL.attacks' is deprecated. "
        "Please switch to '4_Adversarial_FL.attacks.<name>' or the registry interface.",
        DeprecationWarning,
        stacklevel=2,
    )


def fgsm(*args, **kwargs):
    _warn_deprecated()
    return _fgsm_attack(*args, **kwargs)


def rand_noise_attack(*args, **kwargs):
    _warn_deprecated()
    return _random_noise_attack(*args, **kwargs)


def pgd_attack(*args, **kwargs):
    _warn_deprecated()
    return _pgd_attack(*args, **kwargs)


__all__ = ["fgsm", "pgd_attack", "rand_noise_attack"]
