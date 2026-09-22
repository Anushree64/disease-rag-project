"""
attribution_xai.py — Integrated Gradients Axiomatic Visual Attribution.

Computes pixel-level attribution maps highlighting exact input features
that drive model predictions via Riemann sum integration.
"""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.cm as cm

from src.data_utils import get_transforms


def compute_integrated_gradients(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class_idx: Optional[int] = None,
    steps: int = 20,
) -> np.ndarray:
    """
    Computes Integrated Gradients: (x - x') * avg_grad(x' + alpha*(x - x'))
    """
    model.eval()
    baseline = torch.zeros_like(input_tensor)

    if target_class_idx is None:
        with torch.no_grad():
            out = model(input_tensor)
            target_class_idx = int(out.argmax(dim=1).item())

    # Generate interpolated inputs
    alphas = torch.linspace(0.0, 1.0, steps=steps).to(input_tensor.device)
    grads = []

    for alpha in alphas:
        interpolated = baseline + alpha * (input_tensor - baseline)
        interpolated.requires_grad_(True)
        out = model(interpolated)
        score = out[0, target_class_idx]
        score.backward()
        grads.append(interpolated.grad.detach().cpu().numpy())
        model.zero_grad()

    avg_grads = np.mean(grads, axis=0)  # [1, C, H, W]
    delta = (input_tensor - baseline).detach().cpu().numpy()
    integrated_grad = delta * avg_grads  # [1, C, H, W]

    # Sum across color channels
    attr = np.sum(np.abs(integrated_grad[0]), axis=0)  # [H, W]
    if attr.max() > 0:
        attr = attr / attr.max()
    return attr


def get_integrated_gradients_overlay(
    model: nn.Module,
    pil_image: Image.Image,
    target_class_idx: Optional[int] = None,
    alpha: float = 0.5,
) -> Image.Image:
    """Returns PIL overlay image for Integrated Gradients attribution."""
    transform = get_transforms(train=False, img_size=224)
    device = next(model.parameters()).device
    input_tensor = transform(pil_image.convert('RGB')).unsqueeze(0).to(device)

    attr_map = compute_integrated_gradients(model, input_tensor, target_class_idx=target_class_idx, steps=15)

    resized_img = pil_image.convert('RGB').resize((224, 224))
    img_np = np.array(resized_img, dtype=np.float32) / 255.0

    colormap = cm.get_cmap('inferno')
    colored_attr = colormap(attr_map)[:, :, :3]

    overlay = (1.0 - alpha) * img_np + alpha * colored_attr
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


if __name__ == '__main__':
    print("Integrated Gradients attribution module ready.")
