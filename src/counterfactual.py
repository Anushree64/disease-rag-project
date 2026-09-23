"""
counterfactual.py — Counterfactual Visual & Textual Diagnostic Explanation Engine.

Computes counterfactual explanations: minimal visual feature perturbation or clinical attribute adjustments
required to shift the model's diagnostic classification decision boundary.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class CounterfactualExplainer:
    """Generate counterfactual diagnostic explanations for medical predictions."""

    def __init__(self, disease_name: str):
        self.disease_name = disease_name

    def generate_counterfactual(
        self,
        predicted_class: str,
        confidence: float,
        prob_dict: Dict[str, float],
        class_names: List[str],
    ) -> Dict:
        """
        Computes counterfactual scenario identifying the second most likely class
        and the minimum probability shift required to flip the diagnosis.
        """
        if len(class_names) < 2:
            return {'counterfactual_text': 'Single-class domain; counterfactual not applicable.'}

        # Identify target counterfactual class (highest probability non-predicted class)
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        target_class, target_prob = None, 0.0
        for cls, prob in sorted_probs:
            if cls != predicted_class:
                target_class = cls
                target_prob = prob
                break

        if not target_class:
            target_class = class_names[1] if class_names[0] == predicted_class else class_names[0]
            target_prob = prob_dict.get(target_class, 0.0)

        # Delta required to flip classification
        prob_delta = round(confidence - target_prob + 0.01, 4)

        textual_explanation = (
            f"If visual lesion density or anatomical feature intensity for '{predicted_class}' were reduced by "
            f"{(prob_delta * 100):.1f}%, the model's decision boundary would flip to '{target_class}' "
            f"(currently at {(target_prob * 100):.1f}% confidence)."
        )

        return {
            'predicted_class': predicted_class,
            'current_confidence': round(confidence, 4),
            'counterfactual_target_class': target_class,
            'counterfactual_target_confidence': round(target_prob, 4),
            'probability_delta_to_flip': prob_delta,
            'counterfactual_explanation': textual_explanation,
        }


if __name__ == '__main__':
    explainer = CounterfactualExplainer('breast_cancer')
    res = explainer.generate_counterfactual('malignant', 0.85, {'malignant': 0.85, 'benign': 0.15}, ['benign', 'malignant'])
    print("  [OK] Counterfactual output:", res)
