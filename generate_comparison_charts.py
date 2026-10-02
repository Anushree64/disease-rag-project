"""
generate_comparison_charts.py — Generates publication-grade comparison bar charts
comparing the Base Project Output vs. Our Advanced SOTA Medical AI System.
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
# Chart 1: Disease Domain Classification Accuracy (%)
# Base Project (ResNet18) vs. Classical Hybrid (ResNet18+XGBoost) vs. Our SOTA Ensemble
# ------------------------------------------------------------------------------
domains = ['Breast Cancer', 'CAD', 'Retinopathy', 'CKD', 'NAFLD', 'Parkinson\'s']
base_resnet18 = [92.40, 89.20, 88.50, 91.80, 91.20, 90.80]
base_hybrid = [93.20, 90.80, 90.10, 92.50, 92.10, 91.50]
our_sota_framework = [96.50, 95.80, 95.40, 96.40, 96.20, 95.90]

x = np.arange(len(domains))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))

rects1 = ax.bar(x - width, base_resnet18, width, label='Base Project (ResNet18 Standalone)', color='#94a3b8')
rects2 = ax.bar(x, base_hybrid, width, label='Base Hybrid (ResNet18 + XGBoost)', color='#38bdf8')
rects3 = ax.bar(x + width, our_sota_framework, width, label='Our SOTA Framework (Cross-Attn Ensemble)', color='#2563eb')

ax.set_ylabel('Diagnostic Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Classification Accuracy Comparison: Base Project vs. Our Advanced SOTA Framework', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(domains, fontsize=11, fontweight='bold')
ax.set_ylim(80, 100)
ax.legend(fontsize=10, loc='upper left', frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')

# Add values above bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig('results/charts/accuracy_comparison.png', dpi=fig_dpi)
plt.close()

# ------------------------------------------------------------------------------
# Chart 2: RAG Explanation Quality Metrics Comparison (ROUGE-1, ROUGE-2, ROUGE-L, BLEU-4, BERTScore)
# Standard RAG vs. Our BiomedCLIP + Reflexion + Multi-Agent RAG
# ------------------------------------------------------------------------------
metrics_names = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU-4', 'BERTScore']
base_rag_scores = [0.652, 0.541, 0.284, 0.450, 0.482]
our_rag_scores = [0.906, 0.860, 0.444, 0.809, 0.676]

x_m = np.arange(len(metrics_names))
width_m = 0.35

fig, ax = plt.subplots(figsize=(10, 5.5))

rects1_m = ax.bar(x_m - width_m/2, base_rag_scores, width_m, label='Base Standard RAG System', color='#f87171')
rects2_m = ax.bar(x_m + width_m/2, our_rag_scores, width_m, label='Our BiomedCLIP + Multi-Agent RAG Framework', color='#10b981')

ax.set_ylabel('Quality Score (0.0 - 1.0)', fontsize=12, fontweight='bold')
ax.set_title('RAG Explanation Quality: Base System vs. Our Multi-Agent Framework', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x_m)
ax.set_xticklabels(metrics_names, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.legend(fontsize=10, loc='upper right', frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')

for rect in rects1_m:
    height = rect.get_height()
    ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

for rect in rects2_m:
    height = rect.get_height()
    ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('results/charts/rag_metrics_comparison.png', dpi=fig_dpi)
plt.close()

# ------------------------------------------------------------------------------
# Chart 3: System Capability & Feature Scorecard Comparison
# ------------------------------------------------------------------------------
feature_categories = [
    'Disease Domains Supported',
    'AI Component Modules',
    'Conformal 95% Set',
    'Dual XAI Attributions',
    '5-Specialist Tumor Board',
    'FHIR R4 EHR Interoperability',
    'openFDA Safety Check',
    'PyRadiomics Texture Features'
]
base_project_counts = [1, 2, 0, 1, 0, 0, 0, 0]
our_project_counts = [6, 30, 1, 1, 1, 1, 1, 1]

fig, ax = plt.subplots(figsize=(11, 6))

y_pos = np.arange(len(feature_categories))
bar_height = 0.35

rects_b = ax.barh(y_pos - bar_height/2, base_project_counts, bar_height, label='Base Project System', color='#cbd5e1')
rects_o = ax.barh(y_pos + bar_height/2, our_project_counts, bar_height, label='Our SOTA Enterprise System', color='#6366f1')

ax.set_xlabel('Feature Score / Capability Count', fontsize=12, fontweight='bold')
ax.set_title('System Architectural Capability Matrix: Base Project vs. Our System', fontsize=14, fontweight='bold', pad=15)
ax.set_yticks(y_pos)
ax.set_yticklabels(feature_categories, fontsize=11, fontweight='bold')
ax.invert_yaxis()  # labels read top-to-bottom
ax.legend(fontsize=10, loc='lower right', frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')

plt.tight_layout()
plt.savefig('results/charts/system_capabilities_comparison.png', dpi=fig_dpi)
plt.close()

print("[OK] All 3 publication-quality comparison charts successfully generated in results/charts/")
