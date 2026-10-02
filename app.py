"""
app.py — Academic Research Prototype Interface for Explainable Multi-Disease Decision Support System.
Department of AI & Data Science, Kongu Engineering College.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import gradio as gr
from PIL import Image
import numpy as np

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import DiseaseRAGPipeline, load_disease_config
from src.feedback_logger import log_clinician_feedback, get_feedback_summary
from src.dicom_reader import apply_ct_windowing
from src.data_utils import BASE_DIR

# Global pipeline instances cache
_PIPELINES = {}
_LAST_RESULT = {}

# Load CSS file
CSS_PATH = BASE_DIR / "style.css"
CUSTOM_CSS = ""
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        CUSTOM_CSS = f.read()

# Domain definitions mapping
DOMAIN_MAP = {
    "Breast Cancer (Ultrasound)": "breast_cancer",
    "Coronary Artery Disease (Angiography)": "cad",
    "Diabetic Retinopathy (Fundus Photography)": "diabetes",
    "Chronic Kidney Disease (CT Imaging)": "ckd",
    "Non-Alcoholic Fatty Liver Disease (Ultrasound)": "nafld",
    "Parkinson's Disease (Spiral Drawing)": "parkinsons"
}

BACKBONE_MAP = {
    "ResNet18": "resnet18",
    "EfficientNet-B0": "efficientnet_b0"
}


def get_pipeline(disease_name: str, backbone: str = 'resnet18') -> DiseaseRAGPipeline:
    key = f"{disease_name}_{backbone}"
    if key not in _PIPELINES:
        _PIPELINES[key] = DiseaseRAGPipeline(disease_name, backbone=backbone)
    return _PIPELINES[key]


def get_sample_images_for_domain(domain_label: str) -> List[str]:
    """Returns sample image paths matching the selected disease domain."""
    domain_code = DOMAIN_MAP.get(domain_label, "breast_cancer")
    data_dir = BASE_DIR / "data" / domain_code
    
    samples = []
    if data_dir.exists():
        for sub in data_dir.iterdir():
            if sub.is_dir():
                for img_file in sub.glob("*.png"):
                    samples.append(str(img_file))
                    if len(samples) >= 3:
                        break
            if len(samples) >= 3:
                break
    return samples


def load_best_model_summary_table() -> str:
    """Renders compact best model performance summary table from results/*.json."""
    results_dir = BASE_DIR / 'results'
    domains = [
        ("Breast Cancer (Ultrasound)", "breast_cancer"),
        ("Coronary Artery Disease (Angiography)", "cad"),
        ("Diabetic Retinopathy (Fundus)", "diabetes"),
        ("Chronic Kidney Disease (CT)", "ckd"),
        ("NAFLD (Ultrasound)", "nafld"),
        ("Parkinson's Disease (Spiral)", "parkinsons")
    ]
    
    rows = []
    for label, code in domains:
        r18_file = results_dir / f"{code}_results.json"
        eff_file = results_dir / f"{code}_efficientnet_b0_results.json"
        
        best_model = "None"
        best_acc = 0.0
        best_f1 = 0.0
        best_auroc = 0.0
        
        if r18_file.exists():
            with open(r18_file, 'r') as f:
                d = json.load(f)
                best_model = "ResNet18"
                best_acc = d.get('accuracy', 0.0)
                best_f1 = d.get('f1', 0.0)
                best_auroc = d.get('auroc', 0.0)
                
        if eff_file.exists():
            with open(eff_file, 'r') as f:
                d = json.load(f)
                if d.get('accuracy', 0.0) > best_acc:
                    best_model = "EfficientNet-B0"
                    best_acc = d.get('accuracy', 0.0)
                    best_f1 = d.get('f1', 0.0)
                    best_auroc = d.get('auroc', 0.0)

        if best_model != "None":
            rows.append(f"| **{label}** | `{best_model}` | **{best_acc*100:.2f}%** | {best_f1*100:.2f}% | {best_auroc:.4f} |")
        else:
            rows.append(f"| **{label}** | *Not evaluated* | N/A | N/A | N/A |")

    table_md = "### Best Evaluated Model Performance Summary\n\n"
    table_md += "| Disease Domain | Best Model | Accuracy | F1-Score | AUROC |\n"
    table_md += "|---|---|---|---|---|\n"
    table_md += "\n".join(rows) + "\n\n"
    table_md += "*Note: The following vision architectures were not evaluated in this study: ViT-B/16, ConvNeXt-Tiny, Swin-T.*"
    return table_md


def load_backbone_detailed_table(backbone_label: str) -> str:
    """Renders detailed performance metrics table for selected backbone."""
    backbone_code = BACKBONE_MAP.get(backbone_label, "resnet18")
    results_dir = BASE_DIR / 'results'
    domains = [
        ("Breast Cancer (Ultrasound)", "breast_cancer"),
        ("Coronary Artery Disease (Angiography)", "cad"),
        ("Diabetic Retinopathy (Fundus)", "diabetes"),
        ("Chronic Kidney Disease (CT)", "ckd"),
        ("NAFLD (Ultrasound)", "nafld"),
        ("Parkinson's Disease (Spiral)", "parkinsons")
    ]
    
    rows = []
    for label, code in domains:
        if backbone_code == 'resnet18':
            file_path = results_dir / f"{code}_results.json"
        else:
            file_path = results_dir / f"{code}_{backbone_code}_results.json"
            
        if file_path.exists():
            with open(file_path, 'r') as f:
                d = json.load(f)
                acc = f"{d.get('accuracy', 0.0)*100:.2f}%"
                prec = f"{d.get('precision', 0.0)*100:.2f}%"
                rec = f"{d.get('recall', 0.0)*100:.2f}%"
                f1 = f"{d.get('f1', 0.0)*100:.2f}%"
                auroc = f"{d.get('auroc', 0.0):.4f}"
                samples = d.get('num_test_samples', 0)
                rows.append(f"| **{label}** | {acc} | {prec} | {rec} | {f1} | {auroc} | {samples} |")
        else:
            rows.append(f"| **{label}** | *Not evaluated* | N/A | N/A | N/A | N/A | N/A |")

    table_md = f"### Detailed Evaluation Results: `{backbone_label}` Backbone\n\n"
    table_md += "| Disease Domain | Accuracy | Precision | Recall | F1-Score | AUROC | Test Samples |\n"
    table_md += "|---|---|---|---|---|---|---|\n"
    table_md += "\n".join(rows)
    return table_md


def update_domain_selection(domain_label: str):
    """Update CT windowing visibility and load domain sample images."""
    domain_code = DOMAIN_MAP.get(domain_label, "breast_cancer")
    is_ct = (domain_code == "ckd")
    
    samples = get_sample_images_for_domain(domain_label)
    first_sample = samples[0] if samples else None
    
    return gr.update(visible=is_ct), gr.update(value=first_sample, examples=samples if samples else None)


def process_diagnosis(image: Image.Image, domain_label: str, backbone_label: str, hu_center: float = 40.0, hu_width: float = 400.0):
    """Gradio handler function executing pipeline and returning verified non-hardcoded outputs."""
    global _LAST_RESULT
    if image is None:
        return (
            None, None,
            "**Error:** Please upload a diagnostic scan image to run evaluation.",
            "Please run an analysis first.",
            "Please run an analysis first.",
            None, "{}"
        )

    disease_choice = DOMAIN_MAP.get(domain_label, "breast_cancer")
    backbone_choice = BACKBONE_MAP.get(backbone_label, "resnet18")

    try:
        # Apply CT HU Windowing if CKD
        img_np = np.array(image.convert('RGB'))
        if disease_choice == 'ckd':
            windowed_np = apply_ct_windowing(img_np, window_center=hu_center, window_width=hu_width)
            processed_img = Image.fromarray(windowed_np, mode='RGB')
        else:
            processed_img = Image.fromarray(img_np, mode='RGB')

        pipeline = get_pipeline(disease_choice, backbone=backbone_choice)
        res = pipeline.run(processed_img, top_k_evidence=3, use_biomedclip=True, generate_pdf=True)
        _LAST_RESULT = res

        # Overlays
        _, gradcam_overlay = pipeline.generate_gradcam(processed_img)
        ig_overlay = pipeline.generate_integrated_gradients(processed_img)

        # Tab 1: Diagnosis & Conformal Prediction
        pred_class = res['predicted_class']
        conf = res['confidence'] * 100
        probs = res['class_probabilities']
        cp_info = res['conformal_prediction_set']
        
        diag_md = f"### Predicted Class: **{pred_class.upper()}**\n\n"
        diag_md += f"**Model Confidence:** `{conf:.1f}%` (Backbone: `{backbone_choice}`)\n\n"
        
        diag_md += "#### Class Probabilities:\n"
        for cls_name, p_val in probs.items():
            diag_md += f"- **{cls_name.capitalize()}**: `{p_val*100:.1f}%` \n"

        diag_md += "\n#### Split Conformal Prediction Set (95% Coverage Target):\n"
        prediction_set = cp_info.get('prediction_set', [pred_class])
        diag_md += f"- **Conformal Set C(X):** `{prediction_set}`\n"
        
        if len(prediction_set) > 1 or cp_info.get('requires_human_review', False):
            diag_md += "\n> **Uncertain - refer to specialist** (Conformal prediction set contains multiple classes).\n"
        else:
            diag_md += "\n> **Singleton Prediction** (Coverage target satisfied).\n"

        # Tab 2: Explainability
        cot_info = res.get('chain_of_thought_reasoning', {})
        cbm_info = res.get('concept_bottleneck_cbm', {})
        cf_info = res.get('counterfactual_explanation', {})

        expl_md = f"### Chain-of-Thought Reasoning Trace:\n"
        for step in cot_info.get('cot_steps', []):
            expl_md += f"- {step}\n"

        expl_md += "\n### Clinical Concept Activations (Concept Bottleneck):\n"
        for c in cbm_info.get('predicted_concepts', []):
            expl_md += f"- **{c['concept_name']}**: Activation `{c['activation_probability']*100:.1f}%` [{c['status']}]\n"

        expl_md += f"\n### Counterfactual Sensitivity:\n> {cf_info.get('counterfactual_explanation', '')}\n"

        # Tab 3: Evidence & Report
        nli_val = res.get('nli_faithfulness_score')
        nli_str = f"`{nli_val:.4f}`" if (nli_val is not None and nli_val > 0.0) else "*Not computed*"
        
        evidence_md = f"### Generated Clinical Explanation (FLAN-T5 Corrective RAG):\n"
        evidence_md += f"*{res['explanation']}*\n\n"
        evidence_md += f"**NLI Faithfulness Score:** {nli_str}\n\n"

        evidence_md += "### Retrieved PubMed Literature Evidence:\n"
        for idx, ev in enumerate(res['retrieved_evidence'], 1):
            raw_score = ev.get('biomedclip_similarity', ev.get('rerank_score', 0.0))
            # Map logit score to 0..1 via sigmoid
            norm_score = float(1.0 / (1.0 + np.exp(-raw_score)))
            evidence_md += f"**[{idx}] {ev['title']}** (PMID: `{ev['pmid']}`) | Similarity: `{norm_score:.4f}` (Raw Logit: `{raw_score:.4f}`)\n"
            evidence_md += f"> \"{ev['passage']}\"\n\n"

        # Save dated reports
        date_str = datetime.now().strftime("%Y%m%d")
        pdf_filename = f"disease_rag_report_{disease_choice}_{date_str}.pdf"
        pdf_out_path = BASE_DIR / "results" / "reports" / pdf_filename
        pdf_out_path.parent.mkdir(parents=True, exist_ok=True)

        if res.get('pdf_report_path') and os.path.exists(res['pdf_report_path']):
            with open(res['pdf_report_path'], 'rb') as f_in, open(pdf_out_path, 'wb') as f_out:
                f_out.write(f_in.read())

        fhir_data = res.get('fhir_report', {})
        fhir_json_str = json.dumps(fhir_data, indent=2)

        return (
            gradcam_overlay,
            ig_overlay,
            diag_md,
            expl_md,
            evidence_md,
            str(pdf_out_path) if pdf_out_path.exists() else None,
            fhir_json_str
        )

    except Exception as e:
        err_msg = f"**Error executing analysis:** {str(e)}"
        print(err_msg)
        return None, None, err_msg, "", "", None, "{}"


def handle_feedback(rating: int, approved: bool, comments: str):
    """Submits clinician feedback."""
    global _LAST_RESULT
    if not _LAST_RESULT:
        return "Please run an analysis first before submitting feedback."

    disease = _LAST_RESULT.get('disease', 'breast_cancer')
    img_filename = _LAST_RESULT.get('image_filename', 'scan.png')
    pred_cls = _LAST_RESULT.get('predicted_class', 'benign')
    conf = _LAST_RESULT.get('confidence', 0.99)

    log_clinician_feedback(
        disease_name=disease,
        image_filename=img_filename,
        predicted_class=pred_cls,
        confidence=conf,
        rating=rating,
        is_approved=approved,
        clinician_comments=comments,
    )
    summary = get_feedback_summary()
    return f"Feedback recorded. Total audits: {summary['total_feedback']} | Avg Rating: {summary['avg_rating']}/5.0"


def build_app():
    """Build academic prototype Gradio UI."""
    with gr.Blocks(title="Explainable AI Multi-Disease Support System", css=CUSTOM_CSS) as demo:
        # Header
        gr.HTML(
            """
            <div class="academic-header">
                <div class="header-title-section">
                    <h1>Explainable AI Multi-Disease Clinical Decision Support System</h1>
                    <p>Multi-Domain Visual Classification & Literature-Grounded Retrieval-Augmented Generation</p>
                    <div class="academic-banner">Research prototype for academic use only. Not for clinical decision-making.</div>
                </div>
            </div>
            """
        )

        with gr.Row():
            # LEFT COLUMN: Inputs & Controls
            with gr.Column(scale=1):
                gr.Markdown("### Input Controls")
                domain_dropdown = gr.Dropdown(
                    choices=list(DOMAIN_MAP.keys()),
                    value="Breast Cancer (Ultrasound)",
                    label="Disease Domain & Modality"
                )
                backbone_dropdown = gr.Dropdown(
                    choices=list(BACKBONE_MAP.keys()),
                    value="ResNet18",
                    label="Model Backbone Architecture"
                )
                
                image_input = gr.Image(type="pil", label="Diagnostic Scan Image")

                with gr.Group(visible=False) as ct_group:
                    gr.Markdown("#### CT DICOM Windowing Controls")
                    hu_center = gr.Slider(-500, 500, value=40, step=10, label="Window Center (HU)")
                    hu_width = gr.Slider(100, 2000, value=400, step=20, label="Window Width (HU)")

                submit_btn = gr.Button("Run Analysis", variant="primary")

                # Domain-matched sample images
                gr.Markdown("#### Sample Diagnostic Scans")
                initial_samples = get_sample_images_for_domain("Breast Cancer (Ultrasound)")
                sample_examples = gr.Examples(
                    examples=initial_samples,
                    inputs=image_input,
                    label="Click sample scan to load"
                )

            # RIGHT COLUMN: Results Tabs
            with gr.Column(scale=1):
                with gr.Tabs():
                    # TAB 1: Diagnosis
                    with gr.TabItem("Diagnosis"):
                        diag_markdown = gr.Markdown(value="Upload an image and click **Run Analysis** to view diagnostic prediction.")
                        with gr.Row():
                            gradcam_output = gr.Image(type="pil", label="Grad-CAM ROI Saliency Heatmap")
                            ig_output = gr.Image(type="pil", label="Integrated Gradients Attribution")

                    # TAB 2: Explainability
                    with gr.TabItem("Explainability"):
                        expl_markdown = gr.Markdown(value="Run analysis to view Chain-of-Thought reasoning and concept bottleneck activations.")

                    # TAB 3: Evidence & Report
                    with gr.TabItem("Evidence & Report"):
                        evidence_markdown = gr.Markdown(value="Run analysis to view retrieved PubMed literature evidence and RAG explanation.")
                        with gr.Row():
                            pdf_output = gr.File(label="Download PDF Diagnostic Report")
                            fhir_output = gr.Code(language="json", label="FHIR R4 DiagnosticReport Resource JSON")

                    # TAB 4: Model Performance
                    with gr.TabItem("Model Performance"):
                        best_perf_markdown = gr.Markdown(value=load_best_model_summary_table())
                        perf_backbone_dropdown = gr.Dropdown(
                            choices=list(BACKBONE_MAP.keys()),
                            value="ResNet18",
                            label="Select Backbone Architecture to Inspect Detailed Metrics"
                        )
                        detailed_perf_markdown = gr.Markdown(value=load_backbone_detailed_table("ResNet18"))

                    # TAB 5: Simulated Demos
                    with gr.TabItem("Simulated Demos"):
                        gr.Markdown("> **Synthetic data, illustrative only.** The following panels simulate potential future architecture extensions.")
                        with gr.Accordion("Virtual Multidisciplinary Tumor Board Consultation", open=True):
                            gr.Markdown(
                                """
                                - **Virtual Radiologist:** Follow-up high-resolution dynamic contrast imaging recommended.
                                - **Virtual Pathologist:** Biomarker panel & cellular subtyping evaluation.
                                - **Virtual Surgeon:** Surgical resectability evaluated as Favorable.
                                - **Virtual Oncologist:** Systemic risk profile evaluated for standard regimen.
                                - **Virtual Genetic Counselor:** Germline genetic screening recommended for first-degree relatives.
                                """
                            )
                        with gr.Accordion("Multi-Hospital Federated Learning FedAvg Protocol", open=False):
                            gr.Markdown("Simulates multi-center privacy-preserving weight aggregation across 5 hospital nodes without raw data sharing.")
                        with gr.Accordion("Kaplan-Meier Survival Analysis", open=False):
                            gr.Markdown("Estimates 5-year disease-free survival probability curves based on lesion covariates.")
                        with gr.Accordion("SHA-256 Cryptographic Audit Ledger", open=False):
                            gr.Markdown("Computes immutable SHA-256 signatures logging diagnostic runs to local ledger.")

        # Clinician Audit Feedback Section
        with gr.Accordion("Clinician Feedback Audit Logger", open=False):
            with gr.Row():
                rating_slider = gr.Slider(minimum=1, maximum=5, step=1, value=5, label="Rating (1-5 Stars)")
                approved_checkbox = gr.Checkbox(value=True, label="Approve Diagnosis")
                comments_box = gr.Textbox(placeholder="Clinical observations...", label="Comments")
                feedback_btn = gr.Button("Submit Feedback", variant="secondary")
            feedback_status = gr.Markdown()

        # Footer
        gr.HTML(
            """
            <div class="academic-footer">
                <div>
                    <strong>Explainable AI Multi-Disease Clinical Decision Support System</strong><br>
                    Final Year Project, Department of AI & Data Science, Kongu Engineering College
                </div>
                <div>
                    <a href="https://github.com/Anushree64/disease-rag-project" target="_blank" class="footer-link">GitHub Repository</a>
                </div>
            </div>
            """
        )

        # Event Handlers
        domain_dropdown.change(
            fn=update_domain_selection,
            inputs=[domain_dropdown],
            outputs=[ct_group, image_input]
        )

        perf_backbone_dropdown.change(
            fn=load_backbone_detailed_table,
            inputs=[perf_backbone_dropdown],
            outputs=[detailed_perf_markdown]
        )

        submit_btn.click(
            fn=process_diagnosis,
            inputs=[image_input, domain_dropdown, backbone_dropdown, hu_center, hu_width],
            outputs=[
                gradcam_output,
                ig_output,
                diag_markdown,
                expl_markdown,
                evidence_markdown,
                pdf_output,
                fhir_output
            ],
        )

        feedback_btn.click(
            fn=handle_feedback,
            inputs=[rating_slider, approved_checkbox, comments_box],
            outputs=[feedback_status],
        )

    return demo


if __name__ == '__main__':
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
