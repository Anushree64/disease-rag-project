"""
noise_robustness.py — Evaluates CNN classifier robustness against Gaussian image noise.

Features:
- Adds zero-mean Gaussian noise with std sigma in {0.0, 0.05, 0.10, 0.20} to test images.
- Evaluates degradation in test accuracy, F1 score, and confidence across noise intensities.
- Saves results to results/noise_robustness_results.json.
"""

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.data_utils import get_dataloaders, load_disease_config, BASE_DIR
from src.models import build_model


def add_gaussian_noise(tensor: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Add zero-mean Gaussian noise to image tensor and clip to [0, 1]."""
    if sigma <= 0.0:
        return tensor
    noise = torch.randn_like(tensor) * sigma
    noisy = tensor + noise
    return torch.clamp(noisy, 0.0, 1.0)


@torch.no_grad()
def evaluate_noise_robustness(
    disease_name: str,
    backbone_name: str = 'resnet18',
    sigmas: List[float] = [0.0, 0.05, 0.10, 0.20],
    device: torch.device = None,
) -> Dict[str, Dict]:
    """Evaluates a trained classifier across multiple Gaussian noise intensities."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_dir = BASE_DIR
    config = load_disease_config(disease_name)
    class_names = config['classes']
    num_classes = len(class_names)

    _, _, test_loader, _ = get_dataloaders(disease_name, batch_size=64)

    model = build_model(backbone_name=backbone_name, num_classes=num_classes, pretrained=False)
    ckpt_path = base_dir / "results" / f"{disease_name}_{backbone_name}_best.pt"

    if not ckpt_path.exists():
        print(f"  ❌ Missing checkpoint: {ckpt_path.name}")
        return {}

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    results_by_sigma = {}

    for sigma in sigmas:
        all_preds = []
        all_labels = []
        all_confs = []

        for images, labels in test_loader:
            noisy_images = add_gaussian_noise(images, sigma=sigma).to(device)
            outputs = model(noisy_images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confs.extend(confs.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0,
        )
        avg_conf = float(np.mean(all_confs))

        results_by_sigma[f"sigma_{sigma:.2f}"] = {
            'sigma': sigma,
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'avg_confidence': avg_conf,
        }
        print(f"    sigma={sigma:.2f} -> Acc: {acc:.4f}, F1: {f1:.4f}, Avg Conf: {avg_conf:.4f}")

    return results_by_sigma


def run_all_noise_robustness_tests():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diseases = ['breast_cancer', 'cad', 'diabetes', 'ckd', 'nafld', 'parkinsons']
    backbones = ['resnet18', 'efficientnet_b0']
    sigmas = [0.0, 0.05, 0.10, 0.20]

    print("\n==================================================")
    print("  RUNNING IMAGE NOISE ROBUSTNESS TESTS")
    print("==================================================")

    all_results = {}
    for d in diseases:
        for b in backbones:
            key = f"{d}_{b}"
            print(f"  Testing Noise Robustness for {d.upper()} ({b})...")
            res = evaluate_noise_robustness(d, backbone_name=b, sigmas=sigmas, device=device)
            if res:
                all_results[key] = res

    base_dir = BASE_DIR
    with open(base_dir / "results" / "noise_robustness_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n  Noise robustness evaluation complete! Saved to results/noise_robustness_results.json")

    print("\n  Noise robustness evaluation complete! Saved to results/noise_robustness_results.json")


if __name__ == '__main__':
    run_all_noise_robustness_tests()
