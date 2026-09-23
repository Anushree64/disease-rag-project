"""
train.py — Training loop with freeze-then-unfreeze strategy.

Strategy:
  Phase 1: Freeze backbone, train only classification head at lr_head (e.g. 1e-3)
  Phase 2: Unfreeze full network, fine-tune all layers at lr_finetune (e.g. 1e-4)

Uses class-weighted CrossEntropyLoss for imbalanced datasets.
Saves the best checkpoint by validation accuracy.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from src.data_utils import (
    BASE_DIR, load_disease_config, get_dataloaders, get_class_weights
)
from src.models import build_model, freeze_backbone, unfreeze_all
from src.evaluate import evaluate_model

# ── Results directory ──────────────────────────────────────────────────────
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple:
    """Run one training epoch.  Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Print progress every 20 batches for CPU visibility
        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == num_batches:
            print(f"    batch {batch_idx+1}/{num_batches} loss={loss.item():.4f}", flush=True)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Run validation.  Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_disease(
    disease_name: str,
    backbone: str = 'resnet18',
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None,
) -> Dict:
    """
    Full training pipeline for one disease.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = load_disease_config(disease_name)
    classes       = config['classes']
    num_classes   = len(classes)
    total_epochs  = config.get('epochs', 15)
    freeze_epochs = config.get('freeze_epochs', 3)
    lr_head       = config.get('lr_head', 1e-3)
    lr_finetune   = config.get('lr_finetune', 1e-4)

    print(f"
{'#'*60}")
    print(f"  TRAINING: {disease_name} ({backbone})")
    print(f"  Device: {device}")
    print(f"  Classes: {classes}")
    print(f"  Epochs: {freeze_epochs} frozen + {total_epochs - freeze_epochs} unfrozen = {total_epochs}")
    print(f"{'#'*60}")

    # ── Data ────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, _ = get_dataloaders(disease_name, batch_size=batch_size or config.get('batch_size', 32))

    # ── Model ───────────────────────────────────────────────────────────
    model = build_model(backbone, num_classes, pretrained=True).to(device)

    # ── Class-weighted loss ─────────────────────────────────────────────
    class_weights = get_class_weights(train_loader, num_classes, device)
    print(f"  Class weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Phase 1: Frozen backbone ────────────────────────────────────────
    freeze_backbone(model)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_head,
    )

    best_val_acc = 0.0
    ckpt_path = RESULTS_DIR / f'{disease_name}_{backbone}_best.pt'
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    start_time = time.time()

    for epoch in range(1, total_epochs + 1):
        # Transition to Phase 2
        if epoch == freeze_epochs + 1:
            print(f"
  ── Unfreezing all layers at epoch {epoch} ──")
            unfreeze_all(model)
            optimizer = optim.Adam(model.parameters(), lr=lr_finetune)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        phase = "FROZEN" if epoch <= freeze_epochs else "FINETUNE"
        msg = (
            f"  [{phase}] Epoch {epoch:>2}/{total_epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'backbone': backbone,
                'num_classes': num_classes,
                'classes': classes,
                'disease': disease_name,
                'val_acc': best_val_acc,
            }, ckpt_path)
            print(f"{msg}  * Best ({val_acc:.4f})", flush=True)
        else:
            print(msg, flush=True)

    elapsed = time.time() - start_time
    print(f"
  Training complete in {elapsed:.1f}s")
    print(f"  Best val accuracy: {best_val_acc:.4f}")
    print(f"  Checkpoint saved: {ckpt_path}")

    # ── Evaluate on test set with the BEST checkpoint ───────────────────
    print(f"
  ── Evaluating best checkpoint on test set ──")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    metrics = evaluate_model(model, test_loader, classes, device)
    metrics['training_history'] = history
    metrics['training_time_seconds'] = elapsed
    metrics['best_val_accuracy'] = best_val_acc
    metrics['disease'] = disease_name
    metrics['backbone'] = backbone

    # Save metrics
    filename = f'{disease_name}_results.json' if backbone == 'resnet18' else f'{disease_name}_{backbone}_results.json'
    metrics_path = RESULTS_DIR / filename
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Metrics saved: {metrics_path}")


    return metrics


def load_trained_model(
    disease_name: str,
    backbone: str = 'resnet18',
    device: Optional[torch.device] = None,
) -> tuple:
    """
    Load a trained model from its checkpoint.

    Returns (model, classes, device).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = RESULTS_DIR / f'{disease_name}_{backbone}_best.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(
        ckpt.get('backbone', backbone),
        ckpt['num_classes'],
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    classes = ckpt.get('classes', [])
    print(f"  Loaded checkpoint: {ckpt_path}")
    print(f"  Best val acc: {ckpt.get('val_acc', 'N/A')}")

    return model, classes, device


# ───────────────────────────────────────────────────────────────────────────
# CLI entry point
# ───────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    diseases = sys.argv[1:] if len(sys.argv) > 1 else ['breast_cancer', 'cad', 'diabetes']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for disease in diseases:
        try:
            metrics = train_disease(disease, backbone='resnet18', device=device)
            print(f"
  ✅ {disease}: Test Accuracy = {metrics['accuracy']:.4f}")
        except Exception as e:
            print(f"
  ❌ {disease} FAILED: {e}")
            import traceback
            traceback.print_exc()
