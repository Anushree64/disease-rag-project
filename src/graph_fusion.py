"""
graph_fusion.py — PyTorch Graph Attention Network (GAT) fusion layer & k-NN ablation study.

Features:
- Constructs k-NN visual similarity graphs from deep CNN embeddings (ResNet18 / EfficientNet-B0).
- Pure PyTorch Graph Attention Layer (GAT) with multi-head attention mechanisms.
- Performs end-to-end node classification over disease image graphs.
- Conducts k-NN graph sparsity ablation studies (k=4, 8, 16).
- Saves results to results/<disease>_<backbone>_gat_k<k>_results.json.
"""

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from src.data_utils import get_dataloaders, load_disease_config, BASE_DIR
from src.models import build_model
from src.classical_baselines import extract_embeddings


class GATLayer(nn.Module):
    """
    Graph Attention Network (GAT) Layer implemented in PyTorch.
    Computes self-attention over Graph edges defined by edge_index.
    """
    def __init__(self, in_features: int, out_features: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.dropout = dropout

        self.W = nn.Linear(in_features, heads * out_features, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_features))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_features))

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        # Linear transformation
        h = self.W(x).view(N, self.heads, self.out_features)  # [N, heads, out_features]

        src_node, dst_node = edge_index[0], edge_index[1]

        # Compute attention logits
        alpha_src = (h * self.att_src).sum(dim=-1)  # [N, heads]
        alpha_dst = (h * self.att_dst).sum(dim=-1)  # [N, heads]

        edge_alpha = alpha_src[src_node] + alpha_dst[dst_node]  # [E, heads]
        edge_alpha = F.leaky_relu(edge_alpha, negative_slope=0.2)

        # Softmax over incoming edges per node
        # Scatter softmax approximation or dense matrix for graph size
        # Since graph size is N <= 3000, we can perform efficient edge-wise softmax
        max_node = N
        edge_alpha_exp = torch.exp(edge_alpha - edge_alpha.max())
        
        # Sum of exp by dst_node
        denom = torch.zeros(max_node, self.heads, device=x.device)
        denom.index_add_(0, dst_node, edge_alpha_exp)
        
        alpha = edge_alpha_exp / (denom[dst_node] + 1e-16)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Message passing: aggregate features weighted by alpha
        msg = h[src_node] * alpha.unsqueeze(-1)  # [E, heads, out_features]
        out = torch.zeros(N, self.heads, self.out_features, device=x.device)
        out.index_add_(0, dst_node, msg)

        # Average over heads
        out = out.mean(dim=1)  # [N, out_features]
        return out


class GATClassifier(nn.Module):
    """
    Two-layer GAT classifier for graph node embedding and classification.
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.gat1 = GATLayer(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATLayer(hidden_dim, hidden_dim, heads=1, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gat2(h, edge_index)
        h = F.elu(h)
        out = self.fc(h)
        return out


def build_knn_graph(X: np.ndarray, k: int = 8) -> torch.Tensor:
    """
    Build directed k-NN graph edge index [2, E] from node feature matrix X.
    """
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(X)), algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    src_list = []
    dst_list = []
    num_nodes = len(X)

    for i in range(num_nodes):
        for j in indices[i][1:]:  # Skip self loop
            src_list.append(j)
            dst_list.append(i)
            # Add reverse edge for undirected connectivity
            src_list.append(i)
            dst_list.append(j)

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    return torch.tensor(edge_index, dtype=torch.long)


def train_and_eval_gat(
    disease_name: str,
    backbone_name: str = 'resnet18',
    k: int = 8,
    epochs: int = 30,
    device: torch.device = None,
    cached_embeddings: Optional[Tuple] = None,
) -> Dict:
    """
    Train and evaluate GAT fusion model for a disease using embeddings from specified backbone.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_dir = BASE_DIR
    config = load_disease_config(disease_name)
    class_names = config['classes']
    num_classes = len(class_names)

    if cached_embeddings is not None:
        X_train, y_train, X_val, y_val, X_test, y_test = cached_embeddings
    else:
        # Load dataloaders and extract backbone embeddings with batch_size=64
        train_loader, val_loader, test_loader, _ = get_dataloaders(disease_name, batch_size=64)

        model = build_model(backbone_name=backbone_name, num_classes=num_classes, pretrained=False)
        ckpt_path = base_dir / "results" / f"{disease_name}_{backbone_name}_best.pt"
        if not ckpt_path.exists():
            print(f"  ❌ Checkpoint missing: {ckpt_path}")
            return {}

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)

        X_train, y_train = extract_embeddings(model, backbone_name, train_loader, device)
        X_val, y_val = extract_embeddings(model, backbone_name, val_loader, device)
        X_test, y_test = extract_embeddings(model, backbone_name, test_loader, device)

    # Combine all splits to form complete graph structure
    n_train = len(X_train)
    n_val = len(X_val)
    n_test = len(X_test)

    X_all = np.vstack([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])

    train_mask = torch.zeros(len(X_all), dtype=torch.bool)
    val_mask   = torch.zeros(len(X_all), dtype=torch.bool)
    test_mask  = torch.zeros(len(X_all), dtype=torch.bool)

    train_mask[:n_train] = True
    val_mask[n_train:n_train+n_val] = True
    test_mask[n_train+n_val:] = True

    # Construct k-NN graph
    edge_index = build_knn_graph(X_all, k=k).to(device)
    x_tensor = torch.tensor(X_all, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_all, dtype=torch.long).to(device)

    # Initialize GAT Classifier
    gat_model = GATClassifier(
        in_dim=X_all.shape[1],
        hidden_dim=128,
        num_classes=num_classes,
        heads=4,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_test_metrics = {}
    start_t = time.time()

    for epoch in range(1, epochs + 1):
        gat_model.train()
        optimizer.zero_grad()
        out = gat_model(x_tensor, edge_index)
        loss = criterion(out[train_mask], y_tensor[train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        gat_model.eval()
        with torch.no_grad():
            val_out = gat_model(x_tensor, edge_index)
            val_preds = val_out[val_mask].argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_all[val_mask.cpu().numpy()], val_preds)

            test_preds = val_out[test_mask].argmax(dim=1).cpu().numpy()
            test_probs = torch.softmax(val_out[test_mask], dim=1).cpu().numpy()
            y_test_true = y_all[test_mask.cpu().numpy()]

            if val_acc >= best_val_acc:
                best_val_acc = val_acc

                acc = accuracy_score(y_test_true, test_preds)
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_test_true, test_preds, average='weighted', zero_division=0,
                )
                cm = confusion_matrix(y_test_true, test_preds).tolist()
                report = classification_report(y_test_true, test_preds, target_names=class_names, zero_division=0)

                auroc = None
                try:
                    if num_classes == 2:
                        auroc = roc_auc_score(y_test_true, test_probs[:, 1])
                    else:
                        auroc = roc_auc_score(y_test_true, test_probs, multi_class='ovr', average='weighted')
                except Exception:
                    pass

                best_test_metrics = {
                    'accuracy': float(acc),
                    'precision': float(prec),
                    'recall': float(rec),
                    'f1': float(f1),
                    'auroc': float(auroc) if auroc is not None else None,
                    'confusion_matrix': cm,
                    'classification_report': report,
                    'class_names': class_names,
                    'num_test_samples': int(len(y_test_true)),
                    'best_val_acc': float(val_acc),
                    'k_neighbors': k,
                    'disease': disease_name,
                    'backbone': backbone_name,
                    'classifier': f'gat_k{k}',
                    'training_time_seconds': time.time() - start_t,
                }

    # Save to disk
    res_path = base_dir / "results" / f"{disease_name}_{backbone_name}_gat_k{k}_results.json"
    with open(res_path, 'w') as f:
        json.dump(best_test_metrics, f, indent=2)

    return best_test_metrics


def run_gat_ablation_study():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diseases = ['breast_cancer', 'cad', 'diabetes', 'ckd', 'nafld', 'parkinsons']
    backbones = ['resnet18', 'efficientnet_b0']
    k_values = [4, 8, 16]

    print("\n==================================================")
    print("  RUNNING GNN (GAT) FUSION LAYER & k-NN ABLATION")
    print("==================================================")

    all_gat_results = {}
    for d in diseases:
        for b in backbones:
            print(f"\n--- Extracting embeddings for {d.upper()} ({b}) ---")
            config = load_disease_config(d)
            num_classes = len(config['classes'])
            train_loader, val_loader, test_loader, _ = get_dataloaders(d, batch_size=64)

            model = build_model(backbone_name=b, num_classes=num_classes, pretrained=False)
            ckpt_path = BASE_DIR / "results" / f"{d}_{b}_best.pt"
            if not ckpt_path.exists():
                print(f"  ❌ Checkpoint missing: {ckpt_path}")
                continue

            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(state_dict)

            X_tr, y_tr = extract_embeddings(model, b, train_loader, device)
            X_va, y_va = extract_embeddings(model, b, val_loader, device)
            X_te, y_te = extract_embeddings(model, b, test_loader, device)
            cached_emb = (X_tr, y_tr, X_va, y_va, X_te, y_te)

            for k in k_values:
                key = f"{d}_{b}_gat_k{k}"
                print(f"  Training GAT for {d.upper()} | backbone: {b} | k={k}...")
                metrics = train_and_eval_gat(d, backbone_name=b, k=k, epochs=30, device=device, cached_embeddings=cached_emb)
                if metrics:
                    all_gat_results[key] = metrics
                    print(f"    ✓ Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, AUROC: {metrics.get('auroc') or 0:.4f}")

    # Save summary
    base_dir = BASE_DIR
    with open(base_dir / "results" / "gat_ablation_summary.json", 'w') as f:
        json.dump(all_gat_results, f, indent=2)

    print("\n  GAT Ablation study completed successfully!")


if __name__ == '__main__':
    run_gat_ablation_study()

