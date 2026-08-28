"""
model.py
--------
Defines the ResNet50-based transfer learning model for binary
wildfire classification.
"""

import torch
import torch.nn as nn
from torchvision import models


class WildfireResNet50(nn.Module):
    """
    ResNet50 backbone (pre-trained on ImageNet) with a frozen feature
    extractor and a new binary classification head:

        ... -> avgpool -> Linear(2048, 1) -> Sigmoid

    The Sigmoid is applied inside the model (rather than left to the
    loss function) so this module directly outputs a probability in
    [0, 1], as required by nn.BCELoss.
    """

    def __init__(self, freeze_backbone: bool = True):
        super().__init__()

        # Load ResNet50 pretrained on ImageNet using the current
        # torchvision weights API.
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        self.backbone = models.resnet50(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the final fully-connected layer. `fc` in torchvision's
        # ResNet50 maps 2048 -> 1000 (ImageNet classes); we replace it
        # with a 2048 -> 1 layer for binary classification. Because this
        # is a freshly created layer, its parameters have requires_grad=True
        # by default, so it will be the only part of the network trained.
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        probs = self.sigmoid(logits)
        return probs

    def get_trainable_parameters(self):
        """Returns only the parameters that require gradients (the new head)."""
        return [p for p in self.parameters() if p.requires_grad]


def build_model(freeze_backbone: bool = True, device: str = "cpu") -> WildfireResNet50:
    model = WildfireResNet50(freeze_backbone=freeze_backbone)
    model = model.to(device)
    return model
