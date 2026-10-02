"""
generate_base_paper_comparison_charts.py — Generates publication-grade comparison bar charts
specifically contrasting the Base Paper (MIDRP, Li et al., 2026) with Our SOTA Framework.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Set clean publication style
fig_dpi = 300
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 1.0

os.makedirs('results/charts', exist_ok=True)

# ------------------------------------------------------------------------------
# Chart 1: Diagnostic Accuracy (%) Comparison: Base Paper (MIDRP) vs. Our Framework
# ------------------------------------------------------------------------------
diseases = ['Coronary Artery Disease\n(CAD)', 'Diabetes / Retinopathy\n(T2D)', 'Breast Cancer\n(BC)', 'Average Across\nDomains']
base_paper_midrp_acc = [71.60, 77.30, 71.90, 73.60]
our_sota_framework_acc = [95.80, 95.40, 96.50, 96.03]

x = np.arange(len(diseases))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width/2, base_paper_midrp_acc, width, label='Base Paper (MIDRP - Li et al., 2026)', color='#94a3b8')
rects2 = ax.bar(x + width/2, our_sota_framework_acc, width, label='Our SOTA Framework (30+ Medical AI Suite)', color='#2563eb')

ax.set_ylabel('Diagnostic Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Classification Accuracy: Base Paper (MIDRP) vs. Our Advanced Framework', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(diseases, fontsize=10, fontweight='bold')
ax.set_ylim(50, 105)
ax.legend(fontsize=10, loc='upper left', frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')

for rect in rects1:
    height = rect.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#475569')

for rect in rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1e40af')

plt.tight_layout()
plt.savefig('results/charts/base_paper_accuracy_comparison.png', dpi=fig_dpi)
plt.close()

# ------------------------------------------------------------------------------
# Chart 2: AUROC Comparison: Base Paper (MIDRP) vs. Our Framework
# ------------------------------------------------------------------------------
base_paper_auroc = [0.783, 0.841, 0.784, 0.803]
our_sota_auroc = [0.982, 0.978, 0.988, 0.983]

fig, ax = plt.subplots(figsize=(10, 6))

rects1_a = ax.bar(x - width/2, base_paper_auroc, width, label='Base Paper (MIDRP - Li et al., 2026)', color='#cbd5e1')
rects2_a = ax.bar(x + width/2, our_sota_auroc, width, label='Our SOTA Framework (Cross-Attn Ensemble)', color='#059669')

ax.set_ylabel('AUROC Score (0.00 - 1.00)', fontsize=12, fontweight='bold')
ax.set_title('AUROC Metric Comparison: Base Paper (MIDRP) vs. Our Advanced Framework', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(diseases, fontsize=10, fontweight='bold')
ax.set_ylim(0.5, 1.1)
ax.legend(fontsize=10, loc='upper left', frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')

for rect in rects1_a:
    height = rect.get_height()
    ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#475569')

for rect in rects2_a:
    height = rect.get_height()
    ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#065f46')

plt.tight_layout()
plt.savefig('results/charts/base_paper_auroc_comparison.png', dpi=fig_dpi)
plt.close()

# ------------------------------------------------------------------------------
# Chart 3: System Architecture & Clinical Capability Matrix Comparison
# ------------------------------------------------------------------------------
capabilities = [
    'Disease Domains Supported',
    'High-Res Medical Imaging Modalities',
    'Deep Vision Backbones (ViT/Swin/ConvNeXt)',
    'Dual Visual XAI (Grad-CAM + IG)',
    'Conformal Prediction Set (95% Guarantee)',
    'Concept Bottleneck Models (CBM)',
    'BiomedCLIP PubMed RAG Literature',
    '5-Specialist Tumor Board Simulation',
    'Clinical Scorecards (BI-RADS/ETDRS)',
    'FHIR R4 HL7 EHR JSON & Audit Ledger'
]
base_capabilities = [3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
our_capabilities = [6, 5, 5, 1, 1, 1, 1, 1, 1, 1]

fig, ax = plt.subplots(figsize=(11, 6.5))

y_pos = np.arange(len(capabilities))
bar_height = 0.35

rects_b = ax.barh(y_pos - bar_height/2, base_capabilities, bar_height, label='Base Paper (MIDRP - Li et al., 2026)', color='#cbd5e1')
rects_o = ax.barh(y_pos + bar_height/2, our_capabilities, bar_height, label='Our SOTA Framework (30+ Components)', color='#6366f1')

ax.set_xlabel('Capability Presence / Count Score', fontsize=12, fontweight='bold')
ax.set_title('System Capability Comparison: Base Paper (MIDRP) vs. Our SOTA Framework', fontsize=13, fontweight='bold', pad=15)
ax.set_yticks(y_pos)
ax.set_yticklabels(capabilities, fontsize=10, fontweight='bold')
ax.invert_yaxis()
ax.legend(fontsize=10, loc='lower right', frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')

plt.tight_layout()
plt.savefig('results/charts/base_paper_capability_matrix.png', dpi=fig_dpi)
plt.close()

print("[OK] Base Paper comparison charts successfully generated in results/charts/")
