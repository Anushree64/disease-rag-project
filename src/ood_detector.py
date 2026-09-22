"""
ood_detector.py — Out-of-Distribution (OOD) & Anomaly Detection Engine.

Detects non-medical scans, corrupted images, or unknown out-of-domain images
using Energy-Based Scoring E(x) = -T * log sum(exp(f_k(x)/T)).
"""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn


class OODAnomalyDetector:
    """Energy-Based Out-of-Distribution Detector."""

    def __init__(self, temperature: float = 1.0, energy_threshold: float = -5.0):
        self.temperature = temperature
        self.energy_threshold = energy_threshold

    def compute_energy_score(self, logits: torch.Tensor) -> float:
        """
        Computes Energy Score: E(x) = -T * log sum(exp(logits / T))
        In-distribution samples have higher (less negative) energy scores.
        """
        scaled_logits = logits / self.temperature
        energy = -self.temperature * torch.logsumexp(scaled_logits, dim=1).item()
        return float(energy)

    def evaluate_sample(self, logits: torch.Tensor) -> Dict:
        """Evaluates whether logits correspond to In-Distribution vs OOD sample."""
        energy = self.compute_energy_score(logits)
        probs = torch.softmax(logits, dim=1).squeeze()
        max_prob = float(probs.max().item())

        is_ood = (energy < self.energy_threshold) or (max_prob < 0.40)
        status = "⚠️ OUT-OF-DISTRIBUTION / ANOMALY DETECTED" if is_ood else "✅ IN-DISTRIBUTION VALID MEDICAL SCAN"

        return {
            'energy_score': round(energy, 4),
            'max_class_probability': round(max_prob, 4),
            'is_ood': is_ood,
            'ood_status': status,
        }


if __name__ == '__main__':
    print("OOD Anomaly Detector module ready.")
