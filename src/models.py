"""
models.py — Model factory for transfer-learning image classifiers.

Supports:
- ResNet-18 (torchvision, ImageNet-pretrained)
- EfficientNet-B0 (torchvision, ImageNet-pretrained)

Each model's final classification head is replaced with a new
nn.Linear(in_features, num_classes) layer.
"""

import os
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

CONFIDENCE_TEMPERATURE = float(os.environ.get("CONFIDENCE_TEMPERATURE", "3.0"))


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

    elif backbone_name in ['vit_b_16', 'vit']:
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    elif backbone_name in ['convnext_tiny', 'convnext']:
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    elif backbone_name in ['swin_t', 'swin']:
        weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.swin_t(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. "
            f"Choose from: 'resnet18', 'efficientnet_b0', 'vit_b_16', 'convnext_tiny', 'swin_t'"
        )

    # Tag the model so checkpoints can record the backbone
    model.backbone_name = backbone_name
    model.num_classes = num_classes

    total_params = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {backbone_name} -> {num_classes} classes")
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
    elif backbone_name in ['vit_b_16', 'vit']:
        for param in model.heads.parameters():
            param.requires_grad = True
    elif backbone_name in ['convnext_tiny', 'convnext']:
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif backbone_name in ['swin_t', 'swin']:
        for param in model.head.parameters():
            param.requires_grad = True
    else:
        # Fallback: unfreeze last layer
        children = list(model.children())
        if children:
            for param in children[-1].parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [FROZEN] Backbone frozen - {trainable:,} trainable params (head only)")


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all model parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [UNFROZEN] All layers unfrozen - {trainable:,} trainable params")


# ───────────────────────────────────────────────────────────────────────────
# Quick self-test
# ───────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    for name in ['resnet18', 'efficientnet_b0', 'vit_b_16', 'convnext_tiny', 'swin_t']:
        print(f"\n--- {name} ---")
        m = build_model(name, num_classes=5, pretrained=True)
        freeze_backbone(m)
        dummy = torch.randn(2, 3, 224, 224)
        out = m(dummy)
        print(f"  Output shape: {out.shape}")
        unfreeze_all(m)
        print(f"  [OK] {name} OK")
