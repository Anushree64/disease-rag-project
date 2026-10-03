"""
classical_baselines.py — Extracts CNN embeddings and evaluates XGBoost / LightGBM classifiers.

Features:
- Extracts deep visual embeddings from pre-trained & fine-tuned ResNet18 and EfficientNet-B0 models.
- Trains XGBoost and LightGBM models on extracted embeddings.
- Computes test set accuracy, weighted precision/recall/F1, and AUROC.
- Saves results to results/<disease>_<backbone>_<classifier>_results.json.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from src.data_utils import get_dataloaders, load_disease_config, BASE_DIR
from src.models import build_model


def extract_embeddings(
    model: nn.Module,
    backbone_name: str,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract penultimate feature representations and ground-truth labels from a PyTorch model.
    """
    model.eval()
    model.to(device)

    # Store original classification head
    orig_head = None

    if backbone_name == 'resnet18':
        orig_head = model.fc
        model.fc = nn.Identity()
    elif backbone_name == 'efficientnet_b0':
        orig_head = model.classifier[1]
        model.classifier[1] = nn.Identity()

    features_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            feats = model(images)
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())

    # Restore original head
    if backbone_name == 'resnet18':
        model.fc = orig_head
    elif backbone_name == 'efficientnet_b0':
        model.classifier[1] = orig_head

    X = np.vstack(features_list)
    y = np.concatenate(labels_list)
    return X, y


def evaluate_tabular_clf(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
) -> Dict:
    """Evaluate a fitted scikit-learn compatible classifier on test embeddings."""
    y_pred = clf.predict(X_test)
    num_classes = len(class_names)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0,
    )

    auroc = None
    try:
        if hasattr(clf, "predict_proba"):
            y_probs = clf.predict_proba(X_test)
            if num_classes == 2:
                auroc = roc_auc_score(y_test, y_probs[:, 1])
            else:
                auroc = roc_auc_score(y_test, y_probs, multi_class='ovr', average='weighted')
    except Exception as e:
        print(f"    ⚠️ AUROC error: {e}")

    return {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auroc': float(auroc) if auroc is not None else None,
        'confusion_matrix': cm,
        'classification_report': report,
        'class_names': class_names,
        'num_test_samples': int(len(y_test)),
    }


def run_classical_baselines_for_disease(
    disease_name: str,
    backbone_name: str = 'resnet18',
    device: torch.device = None,
) -> Dict[str, Dict]:
    """
    Extract embeddings from fine-tuned backbone, fit XGBoost and LightGBM, and save metrics.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_dir = BASE_DIR
    config = load_disease_config(disease_name)
    class_names = config['classes']
    num_classes = len(class_names)

    print("\n==================================================")
    print(f"  CLASSICAL BASELINES: {disease_name.upper()} ({backbone_name})")
    print("==================================================")

    # 1. Load data loaders with larger batch size for fast inference
    train_loader, val_loader, test_loader, _ = get_dataloaders(disease_name, batch_size=64)

    # 2. Build model and load trained weights
    model = build_model(backbone_name=backbone_name, num_classes=num_classes, pretrained=False)
    checkpoint_name = f"{disease_name}_{backbone_name}_best.pt"
    ckpt_path = base_dir / "results" / checkpoint_name

    if not ckpt_path.exists():
        print(f"  ❌ Checkpoint not found: {ckpt_path}")
        return {}

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    print(f"  Loaded weights from {ckpt_path.name}")

    # 3. Extract embeddings
    print("  Extracting embeddings...", flush=True)
    X_train, y_train = extract_embeddings(model, backbone_name, train_loader, device)
    X_val, y_val = extract_embeddings(model, backbone_name, val_loader, device)
    X_test, y_test = extract_embeddings(model, backbone_name, test_loader, device)

    # Combine train + val for fitting classical ML models
    X_tr = np.vstack([X_train, X_val])
    y_tr = np.concatenate([y_train, y_val])

    results = {}

    # 4. Train XGBoost
    print("  Training XGBoost...")
    start_t = time.time()
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric='logloss' if num_classes == 2 else 'mlogloss',
        random_state=42,
    )
    xgb_model.fit(X_tr, y_tr)
    xgb_time = time.time() - start_t

    xgb_metrics = evaluate_tabular_clf(xgb_model, X_test, y_test, class_names)
    xgb_metrics['training_time_seconds'] = xgb_time
    xgb_metrics['disease'] = disease_name
    xgb_metrics['backbone'] = backbone_name
    xgb_metrics['classifier'] = 'xgboost'

    xgb_res_path = base_dir / "results" / f"{disease_name}_{backbone_name}_xgb_results.json"
    with open(xgb_res_path, 'w') as f:
        json.dump(xgb_metrics, f, indent=2)
    print(f"  ✓ XGBoost Acc: {xgb_metrics['accuracy']:.4f}, F1: {xgb_metrics['f1']:.4f}, AUROC: {xgb_metrics['auroc'] or 0:.4f}")

    # 5. Train LightGBM
    print("  Training LightGBM...")
    start_t = time.time()
    lgb_model = LGBMClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(X_tr, y_tr)
    lgb_time = time.time() - start_t

    lgb_metrics = evaluate_tabular_clf(lgb_model, X_test, y_test, class_names)
    lgb_metrics['training_time_seconds'] = lgb_time
    lgb_metrics['disease'] = disease_name
    lgb_metrics['backbone'] = backbone_name
    lgb_metrics['classifier'] = 'lightgbm'

    lgb_res_path = base_dir / "results" / f"{disease_name}_{backbone_name}_lgbm_results.json"
    with open(lgb_res_path, 'w') as f:
        json.dump(lgb_metrics, f, indent=2)
    print(f"  ✓ LightGBM Acc: {lgb_metrics['accuracy']:.4f}, F1: {lgb_metrics['f1']:.4f}, AUROC: {lgb_metrics['auroc'] or 0:.4f}")

    results['xgboost'] = xgb_metrics
    results['lightgbm'] = lgb_metrics
    return results


def run_all_classical_baselines():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diseases = ['breast_cancer', 'cad', 'diabetes', 'ckd', 'nafld', 'parkinsons']
    backbones = ['resnet18', 'efficientnet_b0']

    all_results = {}
    for d in diseases:
        for b in backbones:
            key = f"{d}_{b}"
            res = run_classical_baselines_for_disease(d, backbone_name=b, device=device)
            all_results[key] = res

    # Save summary
    base_dir = BASE_DIR
    with open(base_dir / "results" / "classical_baselines_summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n==================================================")
    print("  CLASSICAL BASELINES COMPLETED FOR ALL DISEASES")
    print("==================================================")


if __name__ == '__main__':
    run_all_classical_baselines()
