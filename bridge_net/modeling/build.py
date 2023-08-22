import torch.nn as nn

from nxcl.config import ConfigDict

from .architectures import *
from .backbones import *
from .classifiers import *
from .bridges import *


__all__ = [
    "build_backbone",
    "build_classifier",
    "build_model",
    "build_bridge",
]


def build_backbone(cfg: ConfigDict) -> nn.Module:
    name = cfg.MODEL.BACKBONE.NAME

    if name == "build_resnet_backbone":
        backbone = build_resnet_backbone(cfg)
    else:
        raise NotImplementedError(f"Unknown cfg.MODEL.BACKBONE.NAME = \"{name}\"")

    return backbone


def build_classifier(cfg: ConfigDict) -> nn.Module:
    name = cfg.MODEL.CLASSIFIER.NAME

    if name == "build_softmax_classifier":
        classifier = build_softmax_classifier(cfg)
    else:
        raise NotImplementedError(f"Unknown cfg.MODEL.CLASSIFIER.NAME = \"{name}\"")

    return classifier


def build_model(cfg: ConfigDict) -> nn.Module:
    name = cfg.MODEL.META_ARCHITECTURE.NAME

    if name == "ClassificationModelBase":
        model = ClassificationModelBase(
            backbone   = build_backbone(cfg),
            classifier = build_classifier(cfg),
            pixel_mean = cfg.MODEL.PIXEL_MEAN,
            pixel_std  = cfg.MODEL.PIXEL_STD,
        )
    else:
        raise NotImplementedError(f"Unknown cfg.MODEL.META_ARCHITECTURE.NAME = \"{name}\"")

    for m in model.modules():
        if m.__class__.__name__.endswith("Bezier"):
            m.add_param()
            m.add_param()

    return model


def build_bridge(cfg: ConfigDict) -> nn.Module:
    name = cfg.MODEL.META_ARCHITECTURE.NAME

    if name != "BridgeModelBase":
        raise NotImplementedError(f"Unknown cfg.MODEL.META_ARCHITECTURE.NAME = \"{name}\"")

    cfg2 = ConfigDict(MODEL=cfg.BEZIER_MODEL)
    cfg2.MODEL.BACKBONE.RESNET.WIDEN_FACTOR /= cfg.MODEL.SLIM_FACTOR
    cfg2.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER.FEATURE_DIM //= cfg.MODEL.SLIM_FACTOR
    cfg2.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER.LINEAR_LAYERS = "Linear"

    backbone = build_resnet_partial_backbone(
        cfg2,
        mode="normal",
        idx_start=cfg.MODEL.BACKBONES[0],
        in_planes=(cfg.MODEL.BASE_IN_PLANES * len(cfg.MODEL.REQUIRES) * cfg.MODEL.TYPE),
    )
    classifier = build_softmax_classifier(cfg2)

    model = BridgeModelBase(backbone, classifier)

    return model
