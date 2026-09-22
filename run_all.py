"""
disease_project_v2 — Full local runner
Adapted from Colab notebook for Windows/CPU execution.
Runs: EDA → Data cleaning → BiomedCLIP embeddings → Training → Evaluation → Prediction
"""
import os
import sys
import re
import hashlib
import time
import csv
import glob

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── Project paths ───────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, 'src'))

from data_utils import load_config, gather_files, ImgListDataset, find_class_dir
from models import get_transforms, build_resnet18, unfreeze_all

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = os.path.join(BASE, 'data')

def sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

# ══════════════════════════════════════════════════════════════
#  PHASE 1: Exploratory Data Analysis
# ══════════════════════════════════════════════════════════════
def phase1_eda():
    sep("PHASE 1: Exploratory Data Analysis")

    print(f"Device: {DEVICE}")
    print(f"Base:   {BASE}\n")

    # 1a. Directory listing
    print("-- Data directories --")
    for d in sorted(os.listdir(DATA)):
        path = os.path.join(DATA, d)
        if os.path.isdir(path):
            total = sum(1 for _, _, fs in os.walk(path) for f in fs)
            exts = set()
            for _, _, fs in os.walk(path):
                for f in fs:
                    exts.add(os.path.splitext(f)[1].lower())
            print(f"  {d:15s} {total:5d} files   types: {exts}")

    # 1b. Class breakdown
    print("\n-- Class breakdown --")
    # NOTE: ckd, nafld, parkinsons zips contain CSV tabular data, not images.
    # Only the 3 image datasets can go through the image classification pipeline.
    diseases = ['breast_cancer', 'cad', 'diabetes']
    for name in diseases:
        root = os.path.join(DATA, name)
        if not os.path.isdir(root):
            print(f"  {name}: NOT FOUND")
            continue
        counts = Counter()
        for dirpath, _, files in os.walk(root):
            if files:
                counts[os.path.relpath(dirpath, root)] += len(files)
        print(f"\n  {name}")
        for folder, n in sorted(counts.items()):
            print(f"    {folder:35s} {n}")

    # 1c. Build metadata DataFrame
    print("\n-- Building metadata --")
    rows = []
    for disease in diseases:
        root = os.path.join(DATA, disease)
        if not os.path.isdir(root):
            continue
        for dp, _, fs in os.walk(root):
            for f in fs:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    rows.append({
                        'disease': disease,
                        'label': os.path.basename(dp),
                        'path': os.path.join(dp, f),
                        'fname': f
                    })
    df = pd.DataFrame(rows)
    print(f"  Total images: {len(df)}")

    # 1d. Group IDs for breast_cancer
    df['group'] = df['path']
    m = df['disease'] == 'breast_cancer'
    extracted = df.loc[m, 'fname'].str.extract(r'^(\d+)')
    if not extracted.empty:
        df.loc[m, 'group'] = 'bc_' + extracted[0]

    print("\n  Unique groups per disease:")
    print(df.groupby('disease')['group'].nunique().to_string())

    # 1e. Image sizes (sampled)
    print("\n  Image sizes (sampled):")
    for d, g in df.groupby('disease'):
        sample = g['path'].sample(min(20, len(g)), random_state=0)
        sizes = Counter(Image.open(p).size for p in sample)
        print(f"    {d}: {sizes.most_common(3)}")

    return df


# ══════════════════════════════════════════════════════════════
#  PHASE 2: Data Cleaning & Split
# ══════════════════════════════════════════════════════════════
def phase2_clean(df):
    sep("PHASE 2: Data Cleaning & Deduplication")

    # 2a. MD5 hashing
    print("Computing MD5 hashes...")
    df['md5'] = df['path'].map(lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest())

    dups_per_disease = df.groupby('disease')['md5'].apply(lambda s: s.duplicated().sum())
    print("Duplicates per disease:")
    print(dups_per_disease.to_string())

    # 2b. Find conflicts
    dup = df[df.duplicated('md5', keep=False)]
    if len(dup) > 0:
        g = dup.groupby('md5').agg(
            disease=('disease', 'first'),
            copies=('path', 'size'),
            n_labels=('label', 'nunique'),
            n_splits=('split', 'nunique') if 'split' in dup.columns else ('label', 'nunique')
        )
        conflict = g[g.n_labels > 1].index
        print(f"\nConflicting hashes (same image, different labels): {len(conflict)}")

        # Drop conflicts
        df = df[~df['md5'].isin(conflict)]
        df = df.drop_duplicates('md5', keep='first').reset_index(drop=True)
        print(f"After dedup: {len(df)} images")
    else:
        print("No duplicates found.")

    # 2c. Group-aware stratified split
    print("\nCreating train/test split...")
    df['group'] = df['path']
    m = df['disease'] == 'breast_cancer'
    extracted = df.loc[m, 'fname'].str.extract(r'^(\d+)')
    if not extracted.empty:
        df.loc[m, 'group'] = 'bc_' + extracted[0]

    df['split'] = 'train'
    for d, sub in df.groupby('disease'):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        _, test_idx = next(sgkf.split(sub, sub['label'], groups=sub['group']))
        df.loc[sub.index[test_idx], 'split'] = 'test'

    for d, sub in df.groupby('disease'):
        print(f"\n  {d}")
        print(pd.crosstab(sub['split'], sub['label']).to_string())

    meta_path = os.path.join(BASE, 'metadata.csv')
    df.to_csv(meta_path, index=False)
    print(f"\nMetadata saved to {meta_path}")

    return df


# ══════════════════════════════════════════════════════════════
#  PHASE 3: BiomedCLIP Embeddings
# ══════════════════════════════════════════════════════════════
def phase3_embeddings(df):
    sep("PHASE 3: BiomedCLIP Embeddings")

    emb_path = os.path.join(BASE, 'emb.npy')
    if os.path.exists(emb_path):
        emb = np.load(emb_path)
        print(f"Loaded existing embeddings: {emb.shape}")
        return emb

    try:
        import open_clip
    except ImportError:
        print("open_clip not installed, skipping embeddings.")
        return None

    print("Loading BiomedCLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model = model.to(DEVICE).eval()

    class DS(Dataset):
        def __init__(self, paths):
            self.paths = paths
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, i):
            return preprocess(Image.open(self.paths[i]).convert('RGB'))

    dl = DataLoader(DS(df['path'].tolist()), batch_size=32, num_workers=0)
    chunks = []
    total = len(dl)
    print(f"Encoding {len(df)} images in {total} batches...")
    with torch.no_grad():
        for i, x in enumerate(dl):
            f = model.encode_image(x.to(DEVICE))
            chunks.append(torch.nn.functional.normalize(f, dim=-1).cpu())
            if (i + 1) % 10 == 0 or i == total - 1:
                print(f"  batch {i+1}/{total}")

    emb = torch.cat(chunks).numpy().astype('float32')
    np.save(emb_path, emb)
    df.to_csv(os.path.join(BASE, 'metadata.csv'), index=False)
    print(f"Embeddings saved: {emb.shape}")
    return emb


# ══════════════════════════════════════════════════════════════
#  PHASE 4: Leakage Check
# ══════════════════════════════════════════════════════════════
def phase4_leakage_check(df, emb):
    sep("PHASE 4: Leakage Check (test-to-train similarity)")

    if emb is None:
        print("No embeddings available, skipping leakage check.")
        return

    for d in df.disease.unique():
        idx = np.where(df.disease.values == d)[0]
        sub, E = df.iloc[idx], emb[idx]
        is_test = sub.split.values == 'test'

        if is_test.sum() == 0 or (~is_test).sum() == 0:
            continue

        S = E[is_test] @ E[~is_test].T
        max_sim = S.max(1)
        j = S.argmax(1)

        print(f"\n  {d}: test-to-train max similarity percentiles")
        print(f"    {np.percentile(max_sim, [50, 90, 99, 100]).round(3)}")

        close = max_sim > 0.98
        mism = (sub.label.values[is_test][close] != sub.label.values[~is_test][j[close]]).sum()
        print(f"    {close.sum()} of {is_test.sum()} test images have a train neighbor >0.98 "
              f"({mism} with a different label)")


# ══════════════════════════════════════════════════════════════
#  PHASE 5: Training
# ══════════════════════════════════════════════════════════════
def train_one_disease(base_path, disease_name):
    config = load_config(base_path)
    cfg = config[disease_name]
    root = os.path.join(base_path, cfg["root"])
    classes = cfg["classes"]
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    freeze_epochs = cfg["freeze_epochs"]

    print(f"\n=== Training {disease_name} on {DEVICE} ===")
    paths, labels = gather_files(root, classes)
    print(f"  Total images: {len(paths)}")

    if len(paths) == 0:
        print(f"  No images found for {disease_name}, skipping.")
        return None

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_tf, eval_tf = get_transforms()
    train_ds = ImgListDataset(train_paths, train_labels, train_tf)
    val_ds = ImgListDataset(val_paths, val_labels, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_resnet18(num_classes=len(classes), freeze_backbone=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    results_dir = os.path.join(base_path, "results", disease_name)
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "train_log.csv")
    best_ckpt = os.path.join(results_dir, "best_model.pt")

    best_val_acc = 0.0
    with open(log_path, "w", newline="") as logf:
        writer = csv.writer(logf)
        writer.writerow(["epoch", "train_loss", "val_acc"])

        for epoch in range(1, epochs + 1):
            if epoch == freeze_epochs + 1:
                model = unfreeze_all(model)
                print(f"  -> unfroze backbone at epoch {epoch}")

            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
            )

            model.train()
            running_loss = 0.0
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(DEVICE), torch.tensor(lbls).to(DEVICE)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, lbls)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)
            train_loss = running_loss / len(train_ds)

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(DEVICE), torch.tensor(lbls).to(DEVICE)
                    out = model(imgs)
                    preds = out.argmax(dim=1)
                    correct += (preds == lbls).sum().item()
                    total += lbls.size(0)
            val_acc = correct / total

            print(f"  epoch {epoch}/{epochs} | loss={train_loss:.4f} | val_acc={val_acc:.4f}")
            writer.writerow([epoch, train_loss, val_acc])
            logf.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "val_acc": val_acc,
                }, best_ckpt)

    print(f"  Best val_acc for {disease_name}: {best_val_acc:.4f}")
    print(f"  Checkpoint saved to {best_ckpt}")
    return best_val_acc


def phase5_training():
    sep("PHASE 5: Training All Disease Models")

    diseases_to_train = ['breast_cancer', 'cad', 'diabetes']
    all_results = {}

    for d in diseases_to_train:
        try:
            all_results[d] = train_one_disease(BASE, d)
        except Exception as e:
            print(f"!! {d} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[d] = None

    print("\n=== Training Summary ===")
    for d, acc in all_results.items():
        status = f"{acc:.4f}" if acc is not None else "FAILED"
        print(f"  {d:15s}: {status}")

    return all_results


# ══════════════════════════════════════════════════════════════
#  PHASE 6: Evaluation
# ══════════════════════════════════════════════════════════════
def phase6_evaluation():
    sep("PHASE 6: Evaluation")

    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    diseases = ['breast_cancer', 'cad', 'diabetes']
    for disease_name in diseases:
        config = load_config(BASE)
        if disease_name not in config:
            continue
        cfg = config[disease_name]
        root = os.path.join(BASE, cfg["root"])
        classes = cfg["classes"]

        ckpt_path = os.path.join(BASE, "results", disease_name, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  {disease_name}: No checkpoint found, skipping.")
            continue

        try:
            paths, labels = gather_files(root, classes)
        except FileNotFoundError as e:
            print(f"  {disease_name}: {e}")
            continue

        _, val_paths, _, val_labels = train_test_split(
            paths, labels, test_size=0.2, stratify=labels, random_state=42
        )

        _, eval_tf = get_transforms()
        val_ds = ImgListDataset(val_paths, val_labels, eval_tf)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

        model = build_resnet18(num_classes=len(classes), freeze_backbone=False).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(DEVICE)
                out = model(imgs)
                preds = out.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(lbls if isinstance(lbls, list) else lbls.tolist())

        print(f"\n=== {disease_name} Evaluation ===")
        print(classification_report(all_labels, all_preds, target_names=classes))

        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{disease_name} Confusion Matrix")
        fig.colorbar(im)
        plt.tight_layout()

        fig_path = os.path.join(BASE, "results", disease_name, "confusion_matrix.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  Saved confusion matrix to {fig_path}")


# ══════════════════════════════════════════════════════════════
#  PHASE 7: Prediction Test
# ══════════════════════════════════════════════════════════════
def phase7_prediction():
    sep("PHASE 7: Prediction Test")

    for disease_name in ['breast_cancer', 'cad', 'diabetes']:
        ckpt_path = os.path.join(BASE, "results", disease_name, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  {disease_name}: No checkpoint, skipping.")
            continue

        config = load_config(BASE)
        cfg = config[disease_name]
        classes = cfg["classes"]

        model = build_resnet18(num_classes=len(classes), freeze_backbone=False).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        _, eval_tf = get_transforms()

        # Find a sample image
        sample_imgs = glob.glob(os.path.join(BASE, f'data/{disease_name}/**/*.jpg'), recursive=True) + \
                      glob.glob(os.path.join(BASE, f'data/{disease_name}/**/*.png'), recursive=True)

        if not sample_imgs:
            print(f"  {disease_name}: No sample images found.")
            continue

        img = Image.open(sample_imgs[0]).convert("RGB")
        tensor = eval_tf(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().squeeze()

        pred_idx = probs.argmax().item()
        print(f"\n  {disease_name}:")
        print(f"    Image: {os.path.basename(sample_imgs[0])}")
        print(f"    Predicted: {classes[pred_idx]} ({probs[pred_idx]:.1%})")
        for c, p in zip(classes, probs):
            print(f"      {c}: {p:.4f}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    start = time.time()
    print("Disease RAG Project - Local Runner")
    print("Base: " + BASE)
    print("Device: " + str(DEVICE))
    print("Started: " + time.strftime('%Y-%m-%d %H:%M:%S'))

    # Phase 1: EDA
    df = phase1_eda()

    # Phase 2: Clean & Split
    df = phase2_clean(df)

    # Phase 3: BiomedCLIP Embeddings — SKIPPED on CPU (too slow, not needed for training)
    print("\n[Skipping Phase 3: BiomedCLIP embeddings — run on Colab GPU for this step]")

    # Phase 4: Leakage check — SKIPPED (needs embeddings)
    print("[Skipping Phase 4: Leakage check — needs embeddings]")

    # Phase 5: Train all models
    phase5_training()

    # Phase 6: Evaluate
    phase6_evaluation()

    # Phase 7: Prediction test
    phase7_prediction()

    elapsed = time.time() - start
    print("\n" + "="*60)
    print("DONE - Total time: " + str(round(elapsed/60, 1)) + " minutes")
    print("="*60)
