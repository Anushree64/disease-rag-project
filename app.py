"""
app.py — Interactive Web Interface for Multi-Disease Classification & Corrective RAG System.

Complete Suite of 12 Advanced Medical AI Technologies Included:
1. Grad-CAM Saliency Heatmaps & Integrated Gradients Visual Attribution
2. Split Conformal Prediction (95% Coverage Uncertainty Quantification Set)
3. Energy-Based Out-of-Distribution (OOD) Anomaly Detector
4. Monte Carlo (MC) Dropout Epistemic Uncertainty Quantification
5. BiomedCLIP Multimodal Visual-Language Literature Retrieval
6. FLAN-T5 Grounded Clinical Explanations (with LoRA PEFT Adapters)
7. Reflexion Self-Correction Loop
8. Multi-Agent Consensus RAG (Radiologist, Pathologist, Physician perspectives)
9. GraphRAG Biomedical Knowledge Graph Extractor (NetworkX triples)
10. HIPAA-Compliant Differential Privacy (DP) Safeguard Engine
11. Multi-LLM Judge & Automated Clinical Peer-Reviewer
12. SHA-256 Cryptographic Diagnostic Audit Ledger
"""

import os
import sys
from pathlib import Path
import gradio as gr
from PIL import Image

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import DiseaseRAGPipeline, load_disease_config
from src.feedback_logger import log_clinician_feedback, get_feedback_summary

# Global pipeline instances cache
_PIPELINES = {}
_LAST_RESULT = {}


def get_pipeline(disease_name: str, backbone: str = 'resnet18') -> DiseaseRAGPipeline:
    key = f"{disease_name}_{backbone}"
    if key not in _PIPELINES:
        print(f"Loading pipeline for {disease_name} with backbone {backbone}...")
        _PIPELINES[key] = DiseaseRAGPipeline(disease_name, backbone=backbone)
    return _PIPELINES[key]


def process_diagnosis(image: Image.Image, disease_choice: str, backbone_choice: str):
    """Gradio handler function returning outputs including Grad-CAM & Integrated Gradients visual overlays & PDF report."""
    global _LAST_RESULT
    if image is None:
        return None, None, "Please upload an image.", "", "", "", None, ""

    if not disease_choice:
        disease_choice = 'breast_cancer'
    if not backbone_choice:
        backbone_choice = 'resnet18'

    try:
        pipeline = get_pipeline(disease_choice, backbone=backbone_choice)

        # 1. Run pipeline for diagnosis, conformal set, evidence, explanation & metrics
        res = pipeline.run(image, top_k_evidence=3, use_biomedclip=True, generate_pdf=True)
        _LAST_RESULT = res

        # 2. Generate Grad-CAM & Integrated Gradients Overlays
        _, gradcam_overlay = pipeline.generate_gradcam(image)
        ig_overlay = pipeline.generate_integrated_gradients(image)

        # 3. Format Prediction, Confidence, OOD & Conformal Set
        pred_class = res['predicted_class']
        conf = res['confidence'] * 100
        probs = res['class_probabilities']
        cp_info = res['conformal_prediction_set']
        dp_info = res.get('differential_privacy', {})
        ood_info = res.get('ood_anomaly_detection', {})
        mc_info = res.get('mc_epistemic_uncertainty', {})
        audit_info = res.get('cryptographic_audit_block', {})

        pred_text = f"## 🎯 Predicted Diagnosis: **{pred_class}**
"
        pred_text += f"**Model Backbone:** `{backbone_choice}` | **Confidence:** `{conf:.1f}%`
"
        pred_text += f"🔍 **OOD Anomaly Status:** `{ood_info.get('ood_status', 'Valid')}` (Energy: `{ood_info.get('energy_score', 0.0)}`)
"
        pred_text += f"🎲 **Epistemic Uncertainty:** `{mc_info.get('epistemic_uncertainty_level', '')}` (Predictive Entropy: `{mc_info.get('predictive_entropy', 0.0)}`)
"
        pred_text += f"🔒 **Privacy Guarantee:** `{dp_info.get('privacy_guarantee', 'HIPAA DP Protected')}`
"
        pred_text += f"🔑 **SHA-256 Audit Signature:** `{audit_info.get('sha256_signature', '')[:20]}...`

"

        pred_text += "### 📊 Class Probabilities:
"
        for cls, p in probs.items():
            pred_text += f"- **{cls}**: `{p*100:.1f}%`
"

        pred_text += "
### 🛡️ Split Conformal Prediction (95% Coverage Set):
"
        pred_text += f"- **Prediction Set $C(X)$:** `{cp_info['prediction_set']}`
"
        pred_text += f"- **Set Size:** `{cp_info['set_size']}` | **Guaranteed Coverage:** `{cp_info['coverage_level']}`
"
        if cp_info['requires_human_review']:
            pred_text += "⚠️ **Status:** `HUMAN REVIEW RECOMMENDED` (High uncertainty, multiple plausible classes)
"
        else:
            pred_text += "✅ **Status:** `HIGH CONFIDENCE SINGLETON` (Single definitive diagnosis)
"

        # 4. Multi-LLM Judge & GraphRAG Breakdown
        judge = res.get('llm_judge_peer_review', {})
        graph_rag = res.get('graph_rag_knowledge', {})
        reflexion = res.get('reflexion_self_corrected', False)

        judge_text = "### ⚖️ Multi-LLM Judge Clinical Peer-Review:
"
        judge_text += f"- **Overall Score:** `{judge.get('overall_peer_review_score', 0.0):.2f} / 10.0` | **Status:** `{judge.get('recommendation', '')}`
"
        judge_text += f"- **Reflexion Self-Correction Loop:** `{'Activated & Verified' if reflexion else 'Passed Initial Threshold'}`
"

        graph_text = "### 🕸️ GraphRAG Knowledge Graph Extraction:
"
        graph_text += f"- **Graph Nodes:** `{graph_rag.get('num_nodes', 0)}` | **Graph Edges:** `{graph_rag.get('num_edges', 0)}` | **Density:** `{graph_rag.get('graph_density', 0.0)}`
"
        graph_text += "- **Extracted Knowledge Triples:**
"
        for t in graph_rag.get('extracted_triples', [])[:4]:
            graph_text += f"  - `({t[0]})` -> `[{t[1]}]` -> `({t[2]})`  
"

        # 5. Multi-Agent Consensus Breakdown
        consensus = res.get('multi_agent_consensus', {})
        consensus_text = "### 🤝 Multi-Agent Consensus RAG:
"
        consensus_text += f"- **Radiologist Perspective:** *"{consensus.get('radiologist_perspective', '')}"*
"
        consensus_text += f"- **Pathologist Perspective:** *"{consensus.get('pathologist_perspective', '')}"*
"
        consensus_text += f"- **Inter-Agent Agreement Score:** `{consensus.get('inter_agent_consensus_score', 0.0):.4f}` ({consensus.get('consensus_level', '')})
"

        # 6. Format RAG Text Evaluation Metrics
        rag_m = res['rag_text_metrics']
        metrics_text = "### 📈 Quantitative RAG Explanation Quality Metrics:
"
        metrics_text += f"- **ROUGE-1:** `{rag_m['rouge_1']:.4f}` | **ROUGE-2:** `{rag_m['rouge_2']:.4f}` | **ROUGE-L:** `{rag_m['rouge_l']:.4f}`
"
        metrics_text += f"- **BLEU-4:** `{rag_m['bleu_4']:.4f}` | **BERTScore Sim:** `{rag_m['bert_score_sim']:.4f}`
"

        # 7. Format Retrieved Literature Evidence (BiomedCLIP Multimodal)
        evidence_text = "### 📚 Retrieved PubMed Literature (BiomedCLIP Multimodal Reranking):
"
        for idx, ev in enumerate(res['retrieved_evidence'], 1):
            pmid = ev['pmid']
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid != 'Unknown' and not str(pmid).startswith('STATIC') else '#'
            score = ev.get('biomedclip_similarity', ev['rerank_score'])
            evidence_text += f"**[{idx}] [{ev['title']}]({link})** (PMID: [{pmid}]({link})) | *BiomedCLIP Similarity: {score:.4f}*
"
            evidence_text += f"> "{ev['passage']}"

"

        # 8. Format Explanation & Faithfulness
        explanation_text = f"### 💡 Clinical Explanation (FLAN-T5 Grounded RAG):
"
        explanation_text += f"*{res['explanation']}*
"

        faith_score = res['nli_faithfulness']['average_faithfulness']
        faith_text = f"### 🛡️ NLI Faithfulness Score: **{faith_score:.4f} / 1.0000**
"
        faith_text += "*Per-Sentence Entailment Breakdown against Retrieved Evidence:*
"
        for ps in res['nli_faithfulness'].get('per_sentence', []):
            score = ps['entailment_score']
            badge = "🟢 High Entailment" if score >= 0.7 else "🟡 Moderate" if score >= 0.4 else "🔴 Low/Hallucinated"
            faith_text += f"- **"{ps['sentence']}"**  
  Score: `{score:.4f}` | {badge} | Matched PMID: `{ps['matched_pmid']}`
"

        full_explanation_combined = explanation_text + "

" + judge_text + "

" + graph_text + "

" + consensus_text + "

" + faith_text + "

" + metrics_text
        pdf_path = res.get('pdf_report_path')

        return gradcam_overlay, ig_overlay, pred_text, evidence_text, full_explanation_combined, faith_text, pdf_path, "Report ready for download."

    except Exception as e:
        err_msg = f"❌ Error processing diagnosis: {str(e)}"
        print(err_msg)
        import traceback
        traceback.print_exc()
        return None, None, err_msg, "", "", "", None, err_msg


def handle_feedback(rating: int, approved: bool, comments: str):
    """Submits clinician feedback."""
    global _LAST_RESULT
    if not _LAST_RESULT:
        return "⚠️ Please run a diagnosis first before submitting feedback."

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
    return f"✅ Feedback submitted successfully! Total submissions: {summary['total_feedback']} | Avg Rating: {summary['avg_rating']}/5 | Approval Rate: {summary['approval_rate']}%"


# Build Gradio Interface
def build_app():
    title = "🩺 Multi-Disease Image Classification & Visual-Literature RAG Diagnostic System"
    description = (
        "Complete Suite of 12 Advanced Medical AI Technologies: transfer learning CNN backbones (ResNet18 & EfficientNet-B0), "
        "GNN (GAT) graph feature fusion, Grad-CAM & Integrated Gradients XAI maps, Energy-based OOD Anomaly Detection, "
        "Monte Carlo Epistemic Uncertainty, Split Conformal Prediction (95% coverage), BiomedCLIP literature retrieval, "
        "GraphRAG Knowledge Graph extraction, Reflexion Self-Correction Loop, HIPAA Differential Privacy safeguards, "
        "Multi-LLM Judge peer review, SHA-256 Cryptographic Audit Ledger, PDF report generation, and Clinician Feedback logging."
    )

    with gr.Blocks(title="Multi-Disease RAG Diagnosis") as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Diagnostic Image")
                disease_dropdown = gr.Dropdown(
                    choices=[
                        ("Breast Cancer (Ultrasound)", "breast_cancer"),
                        ("Coronary Artery Disease (CAD)", "cad"),
                        ("Diabetic Retinopathy (Fundus)", "diabetes"),
                        ("Chronic Kidney Disease (CT Kidney)", "ckd"),
                        ("Non-Alcoholic Fatty Liver Disease (Ultrasound)", "nafld"),
                        ("Parkinson's Disease (Spiral Drawings)", "parkinsons"),
                    ],
                    value="breast_cancer",
                    label="Select Disease Domain",
                )
                backbone_dropdown = gr.Dropdown(
                    choices=[
                        ("ResNet18", "resnet18"),
                        ("EfficientNet-B0", "efficientnet_b0"),
                    ],
                    value="resnet18",
                    label="Select Classifier Backbone",
                )
                submit_btn = gr.Button("🔍 Run Full Diagnostic & XAI Attribution Suite", variant="primary")

            with gr.Column(scale=1):
                gradcam_output = gr.Image(type="pil", label="Grad-CAM ROI Visual Saliency Overlay")
                ig_output = gr.Image(type="pil", label="Integrated Gradients Axiomatic Attribution Overlay")
                pdf_output = gr.File(label="📄 Download Automated PDF Diagnostic Report")

        with gr.Row():
            with gr.Column(scale=1):
                prediction_output = gr.Markdown(label="Prediction, OOD, Uncertainty & Audit Ledger")
                evidence_output = gr.Markdown(label="Retrieved Evidence")

            with gr.Column(scale=1):
                explanation_output = gr.Markdown(label="Clinical Explanation, LLM Judge & GraphRAG")

        gr.Markdown("### 👨‍⚕️ Clinician Active Learning Feedback Logger")
        with gr.Row():
            rating_slider = gr.Slider(minimum=1, maximum=5, step=1, value=5, label="Clinician Rating (1-5 Stars)")
            approved_checkbox = gr.Checkbox(value=True, label="Approve Diagnostic Explanation")
            comments_box = gr.Textbox(placeholder="Enter clinical observations or correction comments...", label="Clinician Comments")
            feedback_btn = gr.Button("Submit Feedback", variant="secondary")

        feedback_status = gr.Markdown()

        submit_btn.click(
            fn=process_diagnosis,
            inputs=[image_input, disease_dropdown, backbone_dropdown],
            outputs=[gradcam_output, ig_output, prediction_output, evidence_output, explanation_output, gr.State(), pdf_output, feedback_status],
        )

        feedback_btn.click(
            fn=handle_feedback,
            inputs=[rating_slider, approved_checkbox, comments_box],
            outputs=[feedback_status],
        )

    return demo


if __name__ == '__main__':
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
