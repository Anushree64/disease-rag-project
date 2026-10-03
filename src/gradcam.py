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
    """Grad-CAM implementation for PyTorch CNN, ViT, Swin, and ConvNeXt backbones."""

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

        act = self.activations[0]
        grad = self.gradients[0]

        # Determine spatial dimensions based on activation shape
        if act.ndim == 2 and act.shape[0] == 197:
            # ViT-B/16 token sequence: [197, C] where index 0 is [CLS] token
            spatial_act = act[1:]    # [196, C]
            spatial_grad = grad[1:]  # [196, C]
            weights = torch.mean(spatial_grad, dim=0, keepdim=True) # [1, C]
            cam = torch.sum(weights * spatial_act, dim=-1) # [196]
            grid_size = int(np.sqrt(cam.shape[0])) # 14
            cam = cam.reshape(grid_size, grid_size) # [14, 14]

        elif act.ndim == 3 and act.shape[0] != 197:
            # Swin-T NHWC format: [H, W, C]
            weights = torch.mean(grad, dim=(0, 1), keepdim=True) # [1, 1, C]
            cam = torch.sum(weights * act, dim=-1) # [H, W]

        elif act.ndim == 3 and act.shape[0] == 197:
            # ViT-B/16 token sequence with leading dim
            spatial_act = act[0, 1:]
            spatial_grad = grad[0, 1:]
            weights = torch.mean(spatial_grad, dim=0, keepdim=True)
            cam = torch.sum(weights * spatial_act, dim=-1)
            grid_size = int(np.sqrt(cam.shape[0]))
            cam = cam.reshape(grid_size, grid_size)

        else:
            # Standard NCHW 2D feature map [C, H, W]
            weights = torch.mean(grad, dim=(1, 2), keepdim=True)  # [C, 1, 1]
            cam = torch.sum(weights * act, dim=0)  # [H, W]

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
    backbone_name = backbone_name.lower().strip()
    try:
        # Select target layer based on backbone architecture
        if backbone_name == 'resnet18':
            target_layer = model.layer4[-1]
        elif backbone_name in ['efficientnet_b0', 'convnext_tiny']:
            target_layer = model.features[-1]
        elif backbone_name in ['vit_b_16', 'vit']:
            target_layer = model.encoder.layers[-1].ln_1
        elif backbone_name in ['swin_t', 'swin']:
            target_layer = model.features[-1][-1].norm1
        else:
            target_layer = list(model.children())[-2]

        transform = get_transforms(train=False, img_size=224)
        input_tensor = transform(pil_image.convert('RGB')).unsqueeze(0)

        grad_cam = GradCAM(model, target_layer)
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)

        cam_map = grad_cam.generate_heatmap(input_tensor, target_class_idx=target_class_idx)

        # Resize spatial heatmap to match image resolution (224, 224)
        cam_img = Image.fromarray((cam_map * 255.0).astype(np.uint8)).resize((224, 224), resample=Image.BILINEAR)
        cam_map_resized = np.array(cam_img, dtype=np.float32) / 255.0

        # Resize input image to (224, 224)
        resized_img = pil_image.convert('RGB').resize((224, 224))
        img_np = np.array(resized_img, dtype=np.float32) / 255.0

        # Apply JET colormap to heatmap
        colormap = cm.get_cmap('jet')
        heatmap_colored = colormap(cam_map_resized)[:, :, :3]

        # Blend original image and heatmap
        overlay = (1.0 - alpha) * img_np + alpha * heatmap_colored
        overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)

        overlay_pil = Image.fromarray(overlay)
        return overlay_pil, cam_map
    except Exception as e:
        print(f"Warning: Grad-CAM overlay generation failed for backbone '{backbone_name}': {e}")
        fallback_img = pil_image.convert('RGB').resize((224, 224))
        return fallback_img, np.zeros((224, 224), dtype=np.float32)

