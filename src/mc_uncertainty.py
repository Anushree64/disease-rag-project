"""
mc_uncertainty.py — Monte Carlo Dropout Epistemic Uncertainty Quantification.

Measures model epistemic uncertainty via stochastic test-time MC Dropout sampling (N=10 runs).
Computes predictive mean probabilities, epistemic variance, and predictive entropy.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn


def enable_dropout_at_test_time(model: nn.Module):
    """Enables dropout layers during inference mode."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def compute_mc_dropout_uncertainty(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_names: List[str],
    num_samples: int = 10,
) -> Dict:
    """
    Runs N stochastic forward passes with dropout to estimate epistemic uncertainty.
    """
    model.eval()
    enable_dropout_at_test_time(model)

    prob_samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
            prob_samples.append(probs)

    prob_samples = np.array(prob_samples)  # [num_samples, num_classes]

    mean_probs = np.mean(prob_samples, axis=0)  # [num_classes]
    epistemic_var = np.var(prob_samples, axis=0)  # [num_classes]

    # Compute Predictive Entropy H(Y|X) = - sum(p * log(p))
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-12))
    avg_epistemic_variance = float(np.mean(epistemic_var))

    return {
        'mean_class_probabilities': {cls: round(float(p), 4) for cls, p in zip(class_names, mean_probs)},
        'class_epistemic_variance': {cls: round(float(v), 6) for cls, v in zip(class_names, epistemic_var)},
        'predictive_entropy': round(float(entropy), 4),
        'avg_epistemic_variance': round(avg_epistemic_variance, 6),
        'epistemic_uncertainty_level': "🟢 Low Epistemic Uncertainty" if avg_epistemic_variance < 0.01 else "🟡 Moderate Uncertainty" if avg_epistemic_variance < 0.05 else "🔴 High Uncertainty",
    }


if __name__ == '__main__':
    print("Monte Carlo Dropout uncertainty module ready.")
