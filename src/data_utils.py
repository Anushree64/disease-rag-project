"""
data_utils.py — Dataset loading, class-folder discovery, PyTorch Dataset & DataLoaders.

Handles:
- Recursive case-insensitive class folder discovery (for nested datasets like CAD)
- Stratified 70/15/15 train/val/test split
- ImageNet normalization + standard augmentation on training set only
"""

import os
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from collections import Counter

# ── Project root (one level up from src/) ──────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Image extensions to search for ─────────────────────────────────────────
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


# ───────────────────────────────────────────────────────────────────────────
# Class-folder discovery
# ───────────────────────────────────────────────────────────────────────────

def find_class_dir(root: Path, class_name: str) -> Path:
    """
    Recursively search for a directory whose name matches *class_name*
    (case-insensitive).  Returns the first match.

    Raises FileNotFoundError with a full directory-tree listing when
    the class folder cannot be found.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    target = class_name.strip().lower()

    for dirpath in sorted(root.rglob('*')):
        if dirpath.is_dir() and dirpath.name.strip().lower() == target:
            return dirpath

    # Build a helpful error message with the actual tree
    all_dirs = sorted(
        str(d.relative_to(root)) for d in root.rglob('*') if d.is_dir()
    )
    tree_str = "\n".join(f"  📁 {d}" for d in all_dirs) if all_dirs else "  (no subdirectories found)"
    raise FileNotFoundError(
        f"Class folder '{class_name}' not found under '{root}'.\n"
        f"Actual directory tree:\n{tree_str}"
    )


def discover_images(class_dir: Path) -> List[str]:
    """Return all image file paths inside *class_dir* (non-recursive)."""
    images = []
    for f in class_dir.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(str(f))
    return sorted(images)


# ───────────────────────────────────────────────────────────────────────────
# Transforms
# ───────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(train: bool = False, img_size: int = 224) -> transforms.Compose:
    """
    Return image transforms.
    - Training: resize → random flip/rotation/color-jitter → tensor → ImageNet normalize
    - Val/Test: resize → tensor → ImageNet normalize
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])


# ───────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ───────────────────────────────────────────────────────────────────────────

class DiseaseDataset(Dataset):
    """Simple image-classification dataset backed by file paths + integer labels."""

    def __init__(self, image_paths: List[str], labels: List[int],
                 transform: Optional[transforms.Compose] = None):
        assert len(image_paths) == len(labels), "Mismatched paths / labels"
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# ───────────────────────────────────────────────────────────────────────────
# Config loader
# ───────────────────────────────────────────────────────────────────────────

def load_all_configs() -> Dict:
    """Load the full diseases.yaml configuration."""
    config_path = BASE_DIR / 'configs' / 'diseases.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_disease_config(disease_name: str) -> Dict:
    """Load config for a single disease."""
    all_cfg = load_all_configs()
    if disease_name not in all_cfg:
        raise KeyError(
            f"Disease '{disease_name}' not found in config. "
            f"Available: {list(all_cfg.keys())}"
        )
    return all_cfg[disease_name]


# ───────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ───────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    disease_name: str,
    batch_size: Optional[int] = None,
    img_size: int = 224,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Build train / val / test DataLoaders for a disease.

    Returns
    -------
    train_loader, val_loader, test_loader, class_names
    """
    config = load_disease_config(disease_name)
    root   = BASE_DIR / config['root_path']
    classes = config['classes']
    bs = batch_size or config.get('batch_size', 32)

    print(f"\n{'='*60}")
    print(f"Loading dataset: {disease_name}")
    print(f"Root: {root}")
    print(f"{'='*60}")

    all_paths: List[str] = []
    all_labels: List[int] = []

    for idx, cls_name in enumerate(classes):
        cls_dir = find_class_dir(root, cls_name)
        imgs = discover_images(cls_dir)
        if not imgs:
            raise ValueError(
                f"No images found in '{cls_dir}' for class '{cls_name}'. "
                f"Supported extensions: {IMAGE_EXTENSIONS}"
            )
        print(f"  [OK] Class '{cls_name}' (label={idx}): {len(imgs)} images -> {cls_dir}")
        all_paths.extend(imgs)
        all_labels.extend([idx] * len(imgs))

    total = len(all_paths)
    print(f"  Total images: {total}")

    # Group-aware 70 / 15 / 15 split (keeps all augmented copies of one scan in the same split)
    import re
    from sklearn.model_selection import GroupShuffleSplit
    
    groups = [re.split(r'[\s(_]', Path(p).name)[0] for p in all_paths]

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss1.split(all_paths, all_labels, groups=groups))

    X_train = [all_paths[i] for i in train_idx]
    y_train = [all_labels[i] for i in train_idx]

    temp_paths = [all_paths[i] for i in temp_idx]
    temp_labels = [all_labels[i] for i in temp_idx]
    temp_groups = [groups[i] for i in temp_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(gss2.split(temp_paths, temp_labels, groups=temp_groups))

    X_val = [temp_paths[i] for i in val_idx]
    y_val = [temp_labels[i] for i in val_idx]

    X_test = [temp_paths[i] for i in test_idx]
    y_test = [temp_labels[i] for i in test_idx]

    print(f"  Group Split -> train: {len(X_train)} | val: {len(X_val)} | test: {len(X_test)} (Unique Groups: {len(set(groups))})")
    print(f"  Train class distribution: {dict(Counter(y_train))}")
    print(f"  Val   class distribution: {dict(Counter(y_val))}")
    print(f"  Test  class distribution: {dict(Counter(y_test))}")

    # Build datasets
    train_ds = DiseaseDataset(X_train, y_train, transform=get_transforms(train=True,  img_size=img_size))
    val_ds   = DiseaseDataset(X_val,   y_val,   transform=get_transforms(train=False, img_size=img_size))
    test_ds  = DiseaseDataset(X_test,  y_test,  transform=get_transforms(train=False, img_size=img_size))

    # Build loaders
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader, classes


def get_class_weights(train_loader: DataLoader, num_classes: int, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the training set labels
    for use with nn.CrossEntropyLoss(weight=...).
    """
    counts = Counter()
    for _, labels in train_loader:
        counts.update(labels.tolist())

    total = sum(counts.values())
    weights = torch.zeros(num_classes)
    for cls_idx in range(num_classes):
        if counts[cls_idx] > 0:
            weights[cls_idx] = total / (num_classes * counts[cls_idx])
        else:
            weights[cls_idx] = 1.0

    return weights.to(device)


# ───────────────────────────────────────────────────────────────────────────
# Quick self-test
# ───────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    for disease in ['breast_cancer', 'cad', 'diabetes']:
        try:
            train_ld, val_ld, test_ld, cls_names = get_dataloaders(disease)
            imgs, lbls = next(iter(train_ld))
            print(f"  Sample batch: images={imgs.shape}, labels={lbls.shape}")
            print(f"  ✓ {disease} OK\n")
        except Exception as e:
            print(f"  ✗ {disease} FAILED: {e}\n")
