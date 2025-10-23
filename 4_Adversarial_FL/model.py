"""Model definitions used in the adversarial FL experiments."""

import torch.nn as nn
import torchvision.models as tv_models


class MobileNetV2Transfer(nn.Module):
    def __init__(self, pretrained: bool = True, num_classes: int = 10):
        super().__init__()

        weights = "DEFAULT" if pretrained else None
        backbone = tv_models.mobilenet_v2(weights=weights)
        in_feats = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_feats, num_classes)
        self.v2model = backbone

    def forward(self, x):
        return self.v2model(x)


class MobileNetV3Transfer(nn.Module):
    def __init__(self, pretrained: bool = True, num_classes: int = 10):
        super().__init__()

        weights = "DEFAULT" if pretrained else None
        backbone = tv_models.mobilenet_v3_small(weights=weights)
        in_feats = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_feats, num_classes)
        self.v3model = backbone

    def forward(self, x):
        return self.v3model(x)


__all__ = ["MobileNetV2Transfer", "MobileNetV3Transfer"]
