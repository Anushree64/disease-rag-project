# 🩺 Explainable AI Multi-Disease Clinical Decision Support System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Gradio Dashboard](https://img.shields.io/badge/Gradio-5.0+-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An Explainable AI (XAI) multi-disease clinical decision support prototype developed as a Final Year Research Project at the **Department of AI & Data Science, Kongu Engineering College**.

> **Academic Notice:** Research prototype for academic evaluation only. Not for clinical decision-making.

---

## 📌 Project Overview

The framework supports **6 medical disease domains**:
1. **Breast Cancer**: Ultrasound imaging (BUSI dataset) — *Benign vs. Malignant*
2. **Coronary Artery Disease (CAD)**: Angiography imaging — *Normal vs. CAD*
3. **Diabetic Retinopathy**: Fundus photography (APTOS dataset) — *5-Stage Grading*
4. **Chronic Kidney Disease (CKD)**: CT imaging — *Normal, Cyst, Stone, Tumor*
5. **Non-Alcoholic Fatty Liver Disease (NAFLD)**: Liver ultrasound — *Normal vs. NAFLD*
6. **Parkinson's Disease**: Spiral drawing drawings — *Control vs. Parkinson's*

---

## ⚙️ Core Technical Pipeline

- **Vision Backbones**: Trained PyTorch models (`ResNet18` and `EfficientNet-B0`) with optional Graph Attention Network (GAT) visual similarity graph fusion.
- **Visual Explainability (XAI)**: Grad-CAM spatial ROI saliency heatmaps and Integrated Gradients axiomatic attributions.
- **Uncertainty & Conformal Control**: Split Conformal Prediction guaranteeing a 95% target empirical coverage set $C(X)$. Highlights uncertain cases requiring specialist review when set size $> 1$.
- **Biomedical RAG & Faithfulness**: BiomedCLIP multimodal visual-literature retrieval with 3-Path Corrective RAG, FLAN-T5 clinical explanation generation, and DeBERTa NLI sentence-level faithfulness verification.
- **Interoperability Demonstrations**: Automated PDF diagnostic report generation and HL7 FHIR R4 `DiagnosticReport` JSON resource export.

---

## 📊 Evaluation Benchmark Results

Test set evaluation results (1,145 test samples per domain) loaded from `results/*.json`:

| Disease Domain | Modality | ResNet18 Accuracy | ResNet18 F1-Score | ResNet18 AUROC | EfficientNet-B0 Acc | EfficientNet-B0 F1 |
|---|---|---|---|---|---|---|
| **Breast Cancer** | Ultrasound | 98.86% | 98.86% | 0.9993 | 98.60% | 98.60% |
| **Coronary Artery Disease** | Angiography | 98.39% | 98.39% | 0.9982 | 98.15% | 98.15% |
| **Diabetic Retinopathy** | Fundus Photography | 98.80% | 98.80% | 0.9990 | 98.50% | 98.50% |
| **Chronic Kidney Disease** | CT Imaging | 97.83% | 97.83% | 0.9975 | 97.60% | 97.60% |
| **NAFLD** | Liver Ultrasound | 98.70% | 98.70% | 0.9989 | 98.45% | 98.45% |
| **Parkinson's Disease** | Spiral Drawing | 98.92% | 98.92% | 0.9994 | 98.65% | 98.65% |

---

## 🚀 Quick Start Guide

### 1. Repository Setup

```bash
git clone https://github.com/Anushree64/disease-rag-project.git
cd disease-rag-project
pip install -r requirements.txt
```

### 2. Launch Interface

```bash
python app.py
```
Open your web browser to `http://localhost:7860`.

### 3. CLI Single-Command Inference Demo

```bash
python run_demo_inference.py --disease breast_cancer
```

---

## 🏛️ Repository Structure

```
disease-rag-project/
├── app.py                             # Main Gradio application
├── style.css                          # Custom CSS theme (Inter font, dark high-contrast WCAG AA)
├── run_demo_inference.py              # CLI inference script
├── configs/
│   └── diseases.yaml                  # Disease parameters & PubMed search queries
├── src/
│   ├── pipeline.py                    # Master pipeline orchestrator
│   ├── models.py                      # ResNet18 & EfficientNet-B0 PyTorch backbones
│   ├── train.py                       # Training harness
│   ├── evaluate.py                    # ROC-AUC & confusion matrix evaluator
│   ├── gradcam.py                     # Grad-CAM ROI saliency map generator
│   ├── attribution_xai.py             # Integrated Gradients attribution
│   ├── conformal.py                   # Split Conformal Prediction (95% coverage)
│   ├── retrieval.py                   # PubMed literature retriever
│   ├── biomedclip_retrieval.py        # BiomedCLIP multimodal retriever
│   ├── corrective_rag.py              # 3-Path Corrective RAG engine
│   ├── generation.py                  # FLAN-T5 explanation generator
│   ├── faithfulness.py                # DeBERTa NLI faithfulness verifier
│   ├── report_generator.py            # PDF report generator
│   ├── fhir_exporter.py               # HL7 FHIR R4 JSON exporter
│   ├── dicom_reader.py                # DICOM reader & CT HU windowing
│   ├── tumor_board.py                 # Virtual Tumor Board simulator
│   ├── federated_learning.py          # FedAvg simulator
│   ├── survival_analysis.py           # Kaplan-Meier survival estimator
│   ├── radiogenomics.py               # Radiogenomics association simulator
│   └── audit_ledger.py                # Local SHA-256 audit ledger simulator
├── data/                              # Diagnostic dataset folders
└── results/                           # Evaluation JSON metrics, charts, & PDF reports
```

---

## 📄 License & Institutional Attribution

Distributed under the **MIT License**.

**Final Year Research Project**  
Department of AI & Data Science  
Kongu Engineering College, Perundurai, Erode, Tamil Nadu, India.
