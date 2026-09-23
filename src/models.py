"""
models.py — Model factory for transfer-learning image classifiers.

Supports:
- ResNet-18 (torchvision, ImageNet-pretrained)
- EfficientNet-B0 (torchvision, ImageNet-pretrained)

Each model's final classification head is replaced with a new
nn.Linear(in_features, num_classes) layer.
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


def build_model(
    backbone_name: str = 'resnet18',
    num_classes: int = 2,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a transfer-learning classification model.

    Parameters
    ----------
    backbone_name : str
        One of 'resnet18', 'efficientnet_b0'.
    num_classes : int
        Number of output classes.
    pretrained : bool
        Whether to use ImageNet-pretrained weights.

    Returns
    -------
    nn.Module with attribute `backbone_name` set for checkpoint naming.
    """
    backbone_name = backbone_name.lower().strip()

    if backbone_name == 'resnet18':
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif backbone_name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. "
            f"Choose from: 'resnet18', 'efficientnet_b0'"
        )

    # Tag the model so checkpoints can record the backbone
    model.backbone_name = backbone_name
    model.num_classes = num_classes

    total_params = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {backbone_name} → {num_classes} classes")
    print(f"  Parameters: {total_params:,} total, {trainable:,} trainable")

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the final classification head."""
    backbone_name = getattr(model, 'backbone_name', '')

    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the classification head
    if backbone_name == 'resnet18':
        for param in model.fc.parameters():
            param.requires_grad = True
    elif backbone_name == 'efficientnet_b0':
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        # Fallback: unfreeze last layer
        children = list(model.children())
        if children:
            for param in children[-1].parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  🧊 Backbone frozen — {trainable:,} trainable params (head only)")


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all model parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  🔥 All layers unfrozen — {trainable:,} trainable params")


# ───────────────────────────────────────────────────────────────────────────
# Quick self-test
# ───────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    for name in ['resnet18', 'efficientnet_b0']:
        print(f"
--- {name} ---")
        m = build_model(name, num_classes=5, pretrained=True)
        freeze_backbone(m)
        dummy = torch.randn(2, 3, 224, 224)
        out = m(dummy)
        print(f"  Output shape: {out.shape}")
        unfreeze_all(m)
        print(f"  ✓ {name} OK")
