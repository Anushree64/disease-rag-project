# Disease RAG Project

A medical image classification pipeline that trains ResNet18 models on multiple disease datasets, with BiomedCLIP embeddings for data quality analysis and a keyword-based routing agent for inference.

## Diseases Supported

| Disease | Type | Classes | Images |
|---------|------|---------|--------|
| Breast Cancer | Histopathology (PNG) | benign, malignant | 7,632 |
| CAD | Coronary Angiography (JPG) | normal, abnormal | 205 |
| Diabetes (DR) | Retinal Scans (PNG) | Healthy, Mild/Moderate/Proliferate/Severe DR | 2,750 |
| CKD | Tabular (CSV) | - | - |
| NAFLD | Tabular (CSV) | - | - |
| Parkinsons | Tabular (CSV) | - | - |

> **Note:** CKD, NAFLD, and Parkinsons datasets are CSV tabular data (not images). The image classification pipeline runs on the first 3 diseases.

## Project Structure

```
disease-rag-project/
├── configs/
│   └── diseases.yaml          # Training hyperparameters per disease
├── src/
│   ├── data_utils.py           # Dataset loading & file gathering
│   └── models.py               # ResNet18 builder & transforms
├── data/                       # Dataset folders (not in git)
│   ├── breast_cancer/
│   ├── cad/
│   └── diabetes/
├── results/                    # Checkpoints & evaluation outputs
├── run_all.py                  # Main pipeline runner (7 phases)
├── disease_project_v2_fixed.ipynb  # Fixed Colab notebook
└── .gitignore
```

## Setup

### Dataset
Place your zip files and extract them into `data/`:
```
data/breast_cancer/benign/       (2,520 PNGs)
data/breast_cancer/malignant/    (5,112 PNGs)
data/cad/abnormal/               (78 JPGs)
data/cad/normal/                 (127 JPGs)
data/diabetes/Healthy/           (1,000 PNGs)
data/diabetes/Mild DR/           (370 PNGs)
data/diabetes/Moderate DR/       (900 PNGs)
data/diabetes/Proliferate DR/    (290 PNGs)
data/diabetes/Severe DR/         (190 PNGs)
```

### Dependencies
```bash
pip install torch torchvision scikit-learn Pillow pandas matplotlib pyyaml open_clip_torch
```

### Run Locally
```bash
python run_all.py
```

### Run on Google Colab
Upload `disease_project_v2_fixed.ipynb` to Colab with a T4 GPU runtime.

## Pipeline Phases

1. **EDA** — Directory listing, class breakdown, image sizes
2. **Data Cleaning** — MD5 deduplication, conflict removal, stratified group split
3. **BiomedCLIP Embeddings** — 512-dim embeddings for data quality analysis
4. **Leakage Check** — Test-to-train cosine similarity analysis
5. **Training** — ResNet18 with frozen backbone warmup → full fine-tuning
6. **Evaluation** — Classification report + confusion matrices
7. **Prediction Test** — Sample inference on each disease

## Original Notebook
The original Colab notebook (`disease_project_v2.ipynb`) had 4 truncated code cells. The fixed version (`disease_project_v2_fixed.ipynb`) completes all truncated cells.
