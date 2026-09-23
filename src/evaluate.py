"""
evaluate.py — Model evaluation with comprehensive metrics.

Computes:
- Accuracy
- Precision / Recall / F1 (weighted macro)
- Confusion matrix
- AUROC (binary for 2-class, multiclass One-vs-Rest for 3+ classes)
- Full classification report
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    class_names: List[str],
    device: torch.device,
) -> Dict:
    """
    Run full evaluation on a test set.

    Returns a dict with:
    - accuracy, precision, recall, f1 (all weighted)
    - confusion_matrix (as nested list)
    - classification_report (as string)
    - auroc (binary or multiclass OvR)
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    num_classes = len(class_names)

    # ── Core metrics ────────────────────────────────────────────────────
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds).tolist()
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0,
    )

    # ── AUROC ───────────────────────────────────────────────────────────
    try:
        if num_classes == 2:
            # Binary AUROC using probability of the positive class
            auroc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            # Multiclass One-vs-Rest
            auroc = roc_auc_score(
                all_labels, all_probs, multi_class='ovr', average='weighted',
            )
    except ValueError as e:
        print(f"  ⚠️  AUROC computation failed: {e}")
        auroc = None

    # ── Print results ───────────────────────────────────────────────────
    print(f"\n  {'─'*50}")
    print(f"  EVALUATION RESULTS")
    print(f"  {'─'*50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f} (weighted)")
    print(f"  Recall:    {recall:.4f} (weighted)")
    print(f"  F1 Score:  {f1:.4f} (weighted)")
    if auroc is not None:
        print(f"  AUROC:     {auroc:.4f}")
    print(f"\n  Confusion Matrix:")
    # Print confusion matrix with class labels
    header = "  " + " " * 15 + "  ".join(f"{c:>10s}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>10d}" for v in row)
        print(f"  {class_names[i]:>13s}  {row_str}")
    print(f"\n  Classification Report:\n{report}")

    return {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auroc': float(auroc) if auroc is not None else None,
        'confusion_matrix': cm,
        'classification_report': report,
        'class_names': class_names,
        'num_test_samples': int(len(all_labels)),
    }
