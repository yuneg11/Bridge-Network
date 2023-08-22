from typing import Dict, List, Tuple

import torch
import torch.nn as nn


__all__ = [
    "ClassificationModelBase",
]


class ClassificationModelBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        pixel_mean: List[float],
        pixel_std: List[float],
    ) -> None:
        super().__init__()

        self.backbone   = backbone
        self.classifier = classifier

        self.register_buffer(
            name="pixel_mean",
            tensor=torch.Tensor(pixel_mean).view(-1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            name="pixel_std",
            tensor=torch.Tensor(pixel_std).view(-1, 1, 1),
            persistent=False,
        )

    @property
    def device(self):
        return self.pixel_mean.device

    def _preprocess_images(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor]:

        images = images.to(self.device)
        images = (images - self.pixel_mean) / self.pixel_std

        return images

    def forward(
        self,
        images: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        images = self._preprocess_images(images)
        features = self.backbone(images, **kwargs)
        predictions = self.classifier(features["features"], **kwargs)

        outputs = {
            **features,
            **predictions,
        }

        return outputs
