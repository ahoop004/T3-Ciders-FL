"""Model definitions used in the adversarial FL experiments."""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as tv_models

try:  # torchvision>=0.13 prefers explicit weight enums
    from torchvision.models import MobileNet_V2_Weights, MobileNet_V3_Small_Weights
except ImportError:  # pragma: no cover - fallback for older torchvision
    MobileNet_V2_Weights = MobileNet_V3_Small_Weights = None  # type: ignore[assignment]


def _resolve_weights(pretrained: bool, enum_cls):
    if not pretrained:
        return None
    if enum_cls is None:
        return "DEFAULT"
    return enum_cls.DEFAULT


class MobileNetV2Transfer(nn.Module):
    def __init__(self, pretrained: bool = True, num_classes: int = 10):
        super().__init__()

        weights = _resolve_weights(pretrained, MobileNet_V2_Weights)
        backbone = tv_models.mobilenet_v2(weights=weights)
        in_feats = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_feats, num_classes)
        self.v2model = backbone

    def forward(self, x):
        return self.v2model(x)


class MobileNetV3Transfer(nn.Module):
    def __init__(self, pretrained: bool = True, num_classes: int = 10):
        super().__init__()

        weights = _resolve_weights(pretrained, MobileNet_V3_Small_Weights)
        backbone = tv_models.mobilenet_v3_small(weights=weights)
        in_feats = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_feats, num_classes)
        self.v3model = backbone

    def forward(self, x):
        return self.v3model(x)


__all__ = ["MobileNetV2Transfer", "MobileNetV3Transfer"]
