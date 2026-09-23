"""
gradcam.py — Grad-CAM (Gradient-weighted Class Activation Mapping) for ResNet18 & EfficientNet-B0.

Generates visual saliency heatmaps highlighting ROI (Region of Interest) features.
"""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.models import build_model
from src.data_utils import get_transforms


class GradCAM:
    """Grad-CAM implementation for PyTorch CNN backbones."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate normalized Grad-CAM heatmap for input tensor.
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        if target_class_idx is None:
            target_class_idx = output.argmax(dim=1).item()

        score = output[0, target_class_idx]
        score.backward()

        # Global average pooling of gradients over spatial dimensions
        weights = torch.mean(self.gradients[0], dim=(1, 2), keepdim=True)  # [C, 1, 1]
        cam = torch.sum(weights * self.activations[0], dim=0)  # [H, W]

        cam = F.relu(cam).detach().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-16)
        else:
            cam = np.zeros_like(cam)

        return cam


def get_gradcam_overlay(
    model: nn.Module,
    backbone_name: str,
    pil_image: Image.Image,
    target_class_idx: Optional[int] = None,
    alpha: float = 0.5,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Computes Grad-CAM heatmap and returns (overlay_pil_image, raw_cam_map).
    """
    # Select target layer based on backbone architecture
    if backbone_name == 'resnet18':
        target_layer = model.layer4[-1]
    elif backbone_name == 'efficientnet_b0':
        target_layer = model.features[-1]
    else:
        target_layer = list(model.children())[-2]

    transform = get_transforms(train=False, img_size=224)
    input_tensor = transform(pil_image.convert('RGB')).unsqueeze(0)

    grad_cam = GradCAM(model, target_layer)
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    cam_map = grad_cam.generate_heatmap(input_tensor, target_class_idx=target_class_idx)

    # Resize spatial heatmap (e.g. 7x7) to match image resolution (224, 224)
    cam_img = Image.fromarray((cam_map * 255.0).astype(np.uint8)).resize((224, 224), resample=Image.BILINEAR)
    cam_map_resized = np.array(cam_img, dtype=np.float32) / 255.0

    # Resize input image to (224, 224)
    resized_img = pil_image.convert('RGB').resize((224, 224))
    img_np = np.array(resized_img, dtype=np.float32) / 255.0

    # Apply JET colormap to heatmap
    colormap = cm.get_cmap('jet')
    heatmap_colored = colormap(cam_map_resized)[:, :, :3]  # Drop alpha channel

    # Blend original image and heatmap
    overlay = (1.0 - alpha) * img_np + alpha * heatmap_colored
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)

    overlay_pil = Image.fromarray(overlay)
    return overlay_pil, cam_map
