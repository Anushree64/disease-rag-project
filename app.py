"""
app.py — Academic Research Prototype Interface for Explainable Multi-Disease Decision Support System.
Department of AI & Data Science, Kongu Engineering College.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
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


def load_real_performance_metrics() -> str:
    """Reads actual evaluation JSON files from results/ directory."""
    results_dir = BASE_DIR / 'results'
    domains = [
        ("Breast Cancer", "breast_cancer"),
        ("Coronary Artery Disease", "cad"),
        ("Diabetic Retinopathy", "diabetes"),
        ("Chronic Kidney Disease", "ckd"),
        ("Non-Alcoholic Fatty Liver", "nafld"),
        ("Parkinson's Disease", "parkinsons")
    ]
    
    rows = []
    for label, code in domains:
        # ResNet18
        r18_file = results_dir / f"{code}_results.json"
        if r18_file.exists():
            with open(r18_file, 'r') as f:
                data = json.load(f)
                r18_acc = f"{data.get('accuracy', 0)*100:.2f}%"
                r18_prec = f"{data.get('precision', 0)*100:.2f}%"
                r18_rec = f"{data.get('recall', 0)*100:.2f}%"
                r18_f1 = f"{data.get('f1', 0)*100:.2f}%"
                r18_auroc = f"{data.get('auroc', 0):.4f}"
        else:
            r18_acc = r18_prec = r18_rec = r18_f1 = r18_auroc = "Not evaluated"

        # EfficientNet-B0
        eff_file = results_dir / f"{code}_efficientnet_b0_results.json"
        if eff_file.exists():
            with open(eff_file, 'r') as f:
                data = json.load(f)
                eff_acc = f"{data.get('accuracy', 0)*100:.2f}%"
                eff_f1 = f"{data.get('f1', 0)*100:.2f}%"
                eff_auroc = f"{data.get('auroc', 0):.4f}"
        else:
            eff_acc = eff_f1 = eff_auroc = "Not evaluated"

        rows.append(f"| **{label}** | {r18_acc} | {r18_prec} | {r18_rec} | {r18_f1} | {r18_auroc} | {eff_acc} | {eff_f1} | {eff_auroc} | Not evaluated | Not evaluated | Not evaluated |")

    table_md = """
### Test Set Evaluation Results (Loaded from `results/` JSON metrics)

| Disease Domain | ResNet18 Acc | ResNet18 Prec | ResNet18 Rec | ResNet18 F1 | ResNet18 AUROC | EfficientNet Acc | EfficientNet F1 | EfficientNet AUROC | ViT-B/16 | ConvNeXt-Tiny | Swin-T |
|---|---|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(rows)
    return table_md


def update_domain_ui(domain_label: str):
    """Toggle CT windowing sliders visibility based on domain selection."""
    domain_code = DOMAIN_MAP.get(domain_label, "breast_cancer")
    is_ct = (domain_code == "ckd")
    return gr.update(visible=is_ct)


def process_diagnosis(image: Image.Image, domain_label: str, backbone_label: str, hu_center: float = 40.0, hu_width: float = 400.0):
    """Gradio handler function executing pipeline and returning non-hardcoded outputs."""
    global _LAST_RESULT
    if image is None:
        return (
            None, None,
            "**Error:** Please upload a diagnostic image or select a sample scan.",
            "Please run an analysis first.",
            "Please run an analysis first.",
            None, None
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
            diag_md += "\n> **High-Confidence Singleton Prediction** (Coverage target satisfied).\n"

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
        nli_score = res.get('nli_faithfulness_score', 0.0)
        evidence_md = f"### Generated Clinical Explanation (FLAN-T5 Corrective RAG):\n"
        evidence_md += f"*{res['explanation']}*\n\n"
        evidence_md += f"**NLI Faithfulness Score:** `{nli_score:.4f}` (Entailment check against retrieved passages)\n\n"

        evidence_md += "### Retrieved PubMed Literature:\n"
        for idx, ev in enumerate(res['retrieved_evidence'], 1):
            score = ev.get('biomedclip_similarity', ev.get('rerank_score', 0.0))
            evidence_md += f"**[{idx}] {ev['title']}** (PMID: `{ev['pmid']}`) | Similarity: `{score:.4f}`\n"
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

        # Tab 5: Simulated Demos
        tb_info = res.get('tumor_board_consensus', {})
        fed_info = res.get('federated_learning_fedavg', {})
        surv_info = res.get('survival_analysis', {})
        audit_info = res.get('cryptographic_audit_block', {})

        sim_md = "> **Synthetic data, illustrative only.** The following panels represent simulated demonstrations of future architecture extensions.\n\n"
        sim_md += "### Multidisciplinary Tumor Board Simulation:\n"
        for spec in tb_info.get('specialist_opinions', []):
            sim_md += f"- **{spec['specialist_name']} ({spec['role']}):** {spec['recommendation']}\n"
        sim_md += f"\n**Consensus Directive:** {tb_info.get('consensus_directive', '')}\n\n"

        sim_md += f"### Federated Learning FedAvg Simulation:\n"
        sim_md += f"- **Global Aggregated Model Accuracy (Simulated):** `{fed_info.get('global_aggregated_accuracy_pct', 95.8)}%`\n"
        sim_md += f"- **Consortium Nodes:** `{fed_info.get('total_participating_nodes', 5)} Hospitals` ({fed_info.get('total_federated_samples', 15000)} Total Cohort Patients)\n\n"

        sim_md += f"### Kaplan-Meier Survival Analysis Simulation:\n"
        sim_md += f"- **Hazard Ratio (HR):** `{surv_info.get('hazard_ratio', 1.0)}` ({surv_info.get('risk_category', '')})\n"
        sim_md += f"- **5-Year Survival Probability:** `{surv_info.get('five_year_survival_rate', 'N/A')}`\n\n"

        sim_md += f"### SHA-256 Audit Trail Simulation:\n"
        sim_md += f"- **Block Index:** `#{audit_info.get('block_index', 1)}` | **SHA-256 Signature:** `{audit_info.get('sha256_signature', '')}`\n"

        return (
            gradcam_overlay,
            ig_overlay,
            diag_md,
            expl_md,
            evidence_md,
            str(pdf_out_path) if pdf_out_path.exists() else None,
            fhir_json_str,
            sim_md
        )

    except Exception as e:
        err_msg = f"**Error executing analysis:** {str(e)}"
        print(err_msg)
        return None, None, err_msg, "", "", None, "{}", ""


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

                # Sample images
                gr.Markdown("#### Sample Diagnostic Scans")
                sample_img_1 = BASE_DIR / "data" / "breast_cancer" / "benign" / "20586908.png"
                sample_img_2 = BASE_DIR / "data" / "breast_cancer" / "benign" / "20586960.png"

                sample_paths = []
                if sample_img_1.exists():
                    sample_paths.append(str(sample_img_1))
                if sample_img_2.exists():
                    sample_paths.append(str(sample_img_2))

                if sample_paths:
                    gr.Examples(examples=sample_paths, inputs=image_input, label="Click sample image to load")

            # RIGHT COLUMN: Results Tabs
            with gr.Column(scale=1.3):
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
                        perf_markdown = gr.Markdown(value=load_real_performance_metrics())

                    # TAB 5: Simulated Demos
                    with gr.TabItem("Simulated Demos"):
                        sim_markdown = gr.Markdown(value="> **Synthetic data, illustrative only.** Select a scan and click **Run Analysis** to view architecture extension simulations.")

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
            fn=update_domain_ui,
            inputs=[domain_dropdown],
            outputs=[ct_group]
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
                fhir_output,
                sim_markdown
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
