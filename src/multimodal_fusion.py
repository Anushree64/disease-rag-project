"""
multimodal_fusion.py — Tabular + Image Cross-Attention Fusion Module.

Fuses patient clinical metadata (Age, Sex, Risk Index, Biomarker score)
with CNN visual embeddings (ResNet18/EfficientNet-B0) via PyTorch MultiheadAttention.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TabularImageCrossAttentionFusion(nn.Module):
    """
    Cross-Attention Multimodal Fusion Network.
    """
    def __init__(self, visual_dim: int = 512, tabular_dim: int = 4, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.tabular_proj = nn.Linear(tabular_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, visual_feat: torch.Tensor, tabular_feat: torch.Tensor) -> torch.Tensor:
        """
        visual_feat: [B, visual_dim]
        tabular_feat: [B, tabular_dim]
        """
        v_emb = self.visual_proj(visual_feat).unsqueeze(1)  # [B, 1, hidden_dim]
        t_emb = self.tabular_proj(tabular_feat).unsqueeze(1)  # [B, 1, hidden_dim]

        # Cross-Attention: Visual query over Tabular key/value
        attn_out, _ = self.cross_attn(query=v_emb, key=t_emb, value=t_emb)
        fused = self.norm(v_emb + attn_out).squeeze(1)  # [B, hidden_dim]

        logits = self.classifier(fused)
        return logits


def get_simulated_clinical_tabular_data(disease_name: str) -> torch.Tensor:
    """
    Simulates standardized patient clinical metadata tensor [1, 4] for pipeline demonstration:
    [Age_normalized, Sex_binary, Risk_Index_0_1, Biomarker_score_0_1]
    """
    # Deterministic default patient profile
    return torch.tensor([[0.55, 1.0, 0.72, 0.65]], dtype=torch.float32)


if __name__ == '__main__':
    print("Multimodal tabular-image cross-attention fusion module ready.")
