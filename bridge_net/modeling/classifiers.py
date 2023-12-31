from typing import Dict

import torch
import torch.nn as nn

from nxcl.config import ConfigDict as CfgNode

from .layers import *


__all__ = [
    "build_softmax_classifier",
]


class SoftmaxClassifier(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        num_heads: int,
        use_bias: bool,
        linear: nn.Module = Linear,
        **kwargs,
    ) -> None:
        super(SoftmaxClassifier, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_heads   = num_heads
        self.use_bias    = use_bias
        self.linear      = linear

        self.fc = linear(
            in_features=self.feature_dim,
            out_features=self.num_classes * self.num_heads,
            bias=self.use_bias,
            **kwargs
        )

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:

        outputs = dict()

        # make predictions
        logits = self.fc(x, **kwargs)
        if self.num_heads > 1:
            logits = torch.stack(torch.split(logits, self.num_classes, dim=1), dim=1)

        outputs["logits"] = logits
        outputs["confidences"] = torch.softmax(logits, dim=-1)
        outputs["log_confidences"] = torch.log_softmax(logits, dim=-1)

        return outputs


def build_softmax_classifier(cfg: CfgNode) -> nn.Module:

    kwargs = {}

    # Linear layers may be replaced by its variants
    _linear_layers = cfg.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER.LINEAR_LAYERS
    if _linear_layers == "Linear":
        linear_layers = Linear
    elif _linear_layers == "Linear_Bezier":
        linear_layers = Linear_Bezier
    elif _linear_layers == "Linear_BatchEnsemble":
        linear_layers = Linear_BatchEnsemble

        if cfg.MODEL.BATCH_ENSEMBLE.ENABLED is False:
            raise AssertionError(f"Set MODEL.BATCH_ENSEMBLE.ENABLED=True to use {_linear_layers}")

        kwargs.update({
            "ensemble_size": cfg.MODEL.BATCH_ENSEMBLE.ENSEMBLE_SIZE,
            "use_ensemble_bias": cfg.MODEL.BATCH_ENSEMBLE.USE_ENSEMBLE_BIAS,
            "alpha_initializer": {
                "initializer": cfg.MODEL.BATCH_ENSEMBLE.ALPHA_INITIALIZER.NAME,
                "init_values": cfg.MODEL.BATCH_ENSEMBLE.ALPHA_INITIALIZER.VALUES,
            },
            "gamma_initializer": {
                "initializer": cfg.MODEL.BATCH_ENSEMBLE.GAMMA_INITIALIZER.NAME,
                "init_values": cfg.MODEL.BATCH_ENSEMBLE.GAMMA_INITIALIZER.VALUES,
            },
        })
    else:
        raise NotImplementedError(
            f"Unknown MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER.LINEAR_LAYERS: {_linear_layers}"
        )

    classifier = SoftmaxClassifier(
        feature_dim = cfg.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER.FEATURE_DIM,
        num_classes = cfg.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER.NUM_CLASSES,
        use_bias    = cfg.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER.USE_BIAS,
        num_heads   = cfg.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER.NUM_HEADS,
        linear      = linear_layers,
        **kwargs,
    )

    # initialize weights
    if isinstance(classifier.fc.weight, nn.ParameterList):
        for idx in range(len(classifier.fc.weight)):
            nn.init.kaiming_normal_(classifier.fc.weight[idx], mode="fan_out", nonlinearity="relu")
    else:
        nn.init.kaiming_normal_(classifier.fc.weight, mode="fan_out", nonlinearity="relu")

    return classifier
