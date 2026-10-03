"""
conformal.py — Conformal Prediction & Calibrated Uncertainty Quantification for Medical Diagnosis.

Guarantees 95% (1 - alpha) coverage prediction set C(X) containing true disease label.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from src.data_utils import get_dataloaders, load_disease_config, BASE_DIR
from src.models import build_model


class ConformalPredictor:
    """
    Split Conformal Prediction for Multi-Class Medical Classification.
    """
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.q_hat = 1.0  # Quantile threshold

    def calibrate(self, val_probs: np.ndarray, val_labels: np.ndarray):
        """
        Calibrate quantile threshold q_hat from validation set non-conformity scores.
        Non-conformity score: s_i = 1 - P(y_i | x_i)
        """
        n = len(val_labels)
        if n == 0:
            self.q_hat = 1.0
            return

        scores = 1.0 - val_probs[np.arange(n), val_labels]
        # Quantile index with finite sample correction: ceil((n+1)*(1-alpha))/n
        q_idx = int(np.ceil((n + 1) * (1.0 - self.alpha))) - 1
        q_idx = min(max(q_idx, 0), n - 1)

        sorted_scores = np.sort(scores)
        self.q_hat = float(sorted_scores[q_idx])

    def predict_set(self, prob_dict: Dict[str, float], class_names: List[str]) -> Dict:
        """
        Compute 95% conformal prediction set for a single sample prediction.
        Includes all classes whose non-conformity score 1 - p_k <= q_hat (i.e. p_k >= 1 - q_hat).
        """
        threshold = max(0.0, 1.0 - self.q_hat)
        prediction_set = [cls for cls in class_names if prob_dict.get(cls, 0.0) >= threshold]

        # Ensure at least highest probability class is included
        if not prediction_set:
            top_cls = max(prob_dict, key=prob_dict.get)
            prediction_set = [top_cls]

        set_size = len(prediction_set)
        is_singleton = (set_size == 1)
        requires_human_review = not is_singleton

        return {
            'prediction_set': prediction_set,
            'set_size': set_size,
            'is_singleton': is_singleton,
            'requires_human_review': requires_human_review,
            'coverage_level': f"{(1.0 - self.alpha)*100:.0f}%",
            'q_hat': round(self.q_hat, 4),
            'threshold_prob': round(threshold, 4),
        }


def get_calibrated_conformal_predictor(
    disease_name: str,
    backbone_name: str = 'resnet18',
    alpha: float = 0.05,
    device: torch.device = None,
) -> ConformalPredictor:
    """
    Builds and calibrates ConformalPredictor on validation set probabilities.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = load_disease_config(disease_name)
    class_names = config['classes']
    num_classes = len(class_names)

    _, val_loader, _, _ = get_dataloaders(disease_name, batch_size=64)

    model = build_model(backbone_name=backbone_name, num_classes=num_classes, pretrained=False)
    ckpt_path = BASE_DIR / "results" / f"{disease_name}_{backbone_name}_best.pt"

    cp = ConformalPredictor(alpha=alpha)
    if not ckpt_path.exists():
        return cp

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    val_probs_list = []
    val_labels_list = []

    from src.models import CONFIDENCE_TEMPERATURE

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs / CONFIDENCE_TEMPERATURE, dim=1)
            val_probs_list.append(probs.cpu().numpy())
            val_labels_list.append(labels.numpy())

    if val_probs_list:
        val_probs = np.vstack(val_probs_list)
        val_labels = np.concatenate(val_labels_list)
        cp.calibrate(val_probs, val_labels)

    return cp


if __name__ == '__main__':
    print("Conformal prediction module ready.")
