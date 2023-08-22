import copy

import torch
from torch import nn


__all__ = [
    "BridgeModelBase",
]


class BridgeModelBase(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()

        self.backbone   = copy.deepcopy(backbone)    # Base
        self.classifier = copy.deepcopy(classifier)  # Base

    def forward(self, *features):
        x = features[0] if len(features) == 1 else torch.cat(features, dim=1)
        x = self.backbone(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))

        features = x.view(x.size(0), -1)
        outputs = self.classifier(features)

        return {
            "features": features,  #    [batch, feature_dim]
            **outputs,             # k: [batch, num_classes]
        }
