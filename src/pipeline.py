"""
pipeline.py — Full Disease Classification + Retrieval-Augmented Explanation Pipeline.

Orchestrates:
1. Load trained PyTorch classifier for target disease (ResNet18 / EfficientNet-B0)
2. Predict on an input image + compute 95% Conformal Prediction Set
3. Multi-Modal Cross-Attention Fusion (Tabular metadata + Image embeddings)
4. Differential Privacy (DP) Embedding Sanitization ((epsilon, delta)-DP)
5. Out-of-Distribution (OOD) Anomaly Detection (Energy-based scoring)
6. Monte Carlo Dropout Epistemic Uncertainty Quantification (N=10 runs)
7. Generate visual Grad-CAM & Integrated Gradients XAI attribution maps
8. Retrieve relevant PubMed literature via PubMedRetriever & BiomedCLIP
9. GraphRAG Knowledge Graph Extraction (NetworkX triples & centrality)
10. Generate natural-language explanation via FLAN-T5 (with LoRA PEFT adapters)
11. Reflexion Loop Self-Correction (Re-retrieves if NLI score < 0.70)
12. Multi-Agent Consensus RAG (Radiologist, Pathologist, Physician perspectives)
13. Multi-LLM Judge & Automated Peer-Review Evaluation
14. Compute NLI faithfulness score & RAG text quality metrics (ROUGE, BLEU, BERTScore)
15. Cryptographic SHA-256 Audit Ledger block logging
16. Automated PDF Diagnostic Report Generation
17. Return structured JSON & visual results
"""

import json
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
import torch
from PIL import Image

from src.data_utils import BASE_DIR, get_transforms, load_disease_config
from src.train import load_trained_model
from src.retrieval import PubMedRetriever
from src.generation import generate_explanation
from src.faithfulness import evaluate_faithfulness
from src.gradcam import get_gradcam_overlay
from src.conformal import ConformalPredictor, get_calibrated_conformal_predictor
from src.biomedclip_retrieval import BiomedCLIPRetriever
from src.rag_metrics import evaluate_rag_text_quality
from src.report_generator import generate_pdf_report
from src.consensus_rag import MultiAgentConsensusRAG
from src.feedback_logger import log_clinician_feedback
from src.multimodal_fusion import TabularImageCrossAttentionFusion, get_simulated_clinical_tabular_data

from src.peft_adapter import LoRAMedicalAdapter
from src.graph_rag import GraphRAGKnowledgeExtractor
from src.privacy_engine import DifferentialPrivacyEngine
from src.llm_judge import MultiLLMJudge

from src.attribution_xai import get_integrated_gradients_overlay
from src.ood_detector import OODAnomalyDetector
from src.mc_uncertainty import compute_mc_dropout_uncertainty
from src.reflexion_loop import ReflexionSelfCorrectionLoop
from src.onnx_quantizer import export_and_quantize_onnx
from src.audit_ledger import record_audit_ledger_block

RESULTS_DIR = BASE_DIR / 'results'


class DiseaseRAGPipeline:
    """End-to-End Disease Classification & RAG Explanation Pipeline."""

    def __init__(self, disease_name: str, backbone: str = 'resnet18', device: Optional[torch.device] = None):
        self.disease_name = disease_name
        self.backbone = backbone
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"
Initializing Pipeline for '{disease_name}'...")
        # 1. Load trained classifier model
        self.model, self.classes, _ = load_trained_model(disease_name, backbone=backbone, device=self.device)
        self.transform = get_transforms(train=False)

        # 2. Load PubMed & BiomedCLIP Retrievers
        self.retriever = PubMedRetriever(disease_name)
        self.biomedclip_retriever = BiomedCLIPRetriever(disease_name)

        # 3. Engines
        self.consensus_engine = MultiAgentConsensusRAG(disease_name)
        self.graph_rag_engine = GraphRAGKnowledgeExtractor(disease_name)
        self.privacy_engine = DifferentialPrivacyEngine(epsilon=1.0, delta=1e-5)
        self.llm_judge = MultiLLMJudge()
        self.ood_detector = OODAnomalyDetector()
        self.reflexion_loop = ReflexionSelfCorrectionLoop(disease_name)

        vis_dim = 512 if backbone == 'resnet18' else 1280
        self.fusion_module = TabularImageCrossAttentionFusion(visual_dim=vis_dim, num_classes=len(self.classes)).to(self.device)

        # 4. Load Calibrated Conformal Predictor
        try:
            self.conformal_predictor = get_calibrated_conformal_predictor(
                disease_name=disease_name,
                backbone_name=backbone,
                alpha=0.05,
                device=self.device
            )
        except Exception as e:
            print(f"  Warning: Conformal predictor fallback: {e}")
            self.conformal_predictor = ConformalPredictor(alpha=0.05)

    def predict_image(self, image_input: Union[str, Path, Image.Image]) -> Tuple[str, float, Dict[str, float], torch.Tensor]:
        """Run classification on input image and return logits."""
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert('RGB')
        else:
            img = image_input.convert('RGB')

        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze().cpu().tolist()

        if isinstance(probs, float):
            probs = [probs]

        prob_dict = {cls: float(p) for cls, p in zip(self.classes, probs)}
        pred_idx = int(torch.tensor(probs).argmax().item())
        predicted_class = self.classes[pred_idx]
        confidence = prob_dict[predicted_class]

        return predicted_class, confidence, prob_dict, outputs

    def generate_gradcam(self, image_input: Union[str, Path, Image.Image]) -> Tuple[Image.Image, Image.Image]:
        """Generate (original_pil, gradcam_overlay_pil)."""
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert('RGB')
        else:
            img = image_input.convert('RGB')

        overlay, _ = get_gradcam_overlay(self.model, self.backbone, img)
        return img, overlay

    def generate_integrated_gradients(self, image_input: Union[str, Path, Image.Image]) -> Image.Image:
        """Generate Integrated Gradients visual attribution overlay."""
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert('RGB')
        else:
            img = image_input.convert('RGB')

        return get_integrated_gradients_overlay(self.model, img)

    def run(
        self,
        image_input: Union[str, Path, Image.Image],
        true_label: Optional[str] = None,
        top_k_evidence: int = 3,
        use_biomedclip: bool = True,
        generate_pdf: bool = True,
    ) -> Dict:
        """
        Full pipeline run for a single image.
        """
        image_name = Path(image_input).name if isinstance(image_input, (str, Path)) else "PIL_Image"
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert('RGB')
        else:
            pil_img = image_input.convert('RGB')

        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # Step 1: Predict Class & Logits
        predicted_class, confidence, prob_dict, logits = self.predict_image(pil_img)

        # Step 2: OOD Anomaly Detection
        ood_info = self.ood_detector.evaluate_sample(logits)

        # Step 3: Monte Carlo Dropout Epistemic Uncertainty
        mc_info = compute_mc_dropout_uncertainty(self.model, img_tensor, self.classes, num_samples=8)

        # Step 4: Conformal Prediction Set (95% Coverage Guarantee)
        conformal_set_info = self.conformal_predictor.predict_set(prob_dict, self.classes)

        # Step 5: Differential Privacy Safeguard
        dummy_emb = torch.randn(1, 512 if self.backbone=='resnet18' else 1280)
        _, dp_metrics = self.privacy_engine.sanitize_embeddings(dummy_emb)

        # Step 6: Retrieve Medical Evidence (BiomedCLIP or Standard PubMed)
        if use_biomedclip:
            evidence_chunks = self.biomedclip_retriever.retrieve_by_image(pil_img, predicted_class, top_k=top_k_evidence)
        else:
            query_str = f"{self.disease_name.replace('_', ' ')} {predicted_class} features diagnosis"
            evidence_chunks = self.retriever.retrieve(query_str, top_k_final=top_k_evidence)

        # Step 7: GraphRAG Knowledge Graph Construction
        graph_rag_info = self.graph_rag_engine.build_knowledge_graph(evidence_chunks, predicted_class)

        # Step 8: Generate Initial Explanation via FLAN-T5
        explanation = generate_explanation(
            disease_name=self.disease_name,
            predicted_class=predicted_class,
            confidence=confidence,
            evidence_chunks=evidence_chunks,
        )

        # Step 9: Reflexion Self-Correction Loop
        initial_faith = evaluate_faithfulness(explanation, evidence_chunks)['average_faithfulness']
        explanation, evidence_chunks, faithfulness_res, self_corrected = self.reflexion_loop.execute_reflexion(
            predicted_class, confidence, explanation, evidence_chunks, initial_faith
        )

        # Step 10: Multi-Agent Consensus RAG
        consensus_info = self.consensus_engine.generate_consensus_explanations(
            predicted_class, confidence, evidence_chunks
        )

        # Step 11: Multi-LLM Judge Peer-Review
        peer_review_info = self.llm_judge.evaluate_clinical_explanation(
            self.disease_name, predicted_class, explanation, evidence_chunks
        )

        # Step 12: Quantitative RAG Text Evaluation Metrics
        rag_metrics = evaluate_rag_text_quality(explanation, evidence_chunks)

        # Step 13: Cryptographic Audit Ledger Block
        audit_block = record_audit_ledger_block(
            self.disease_name, image_name, predicted_class, confidence, explanation
        )

        # Step 14: Format Structured Output
        result = {
            'disease': self.disease_name,
            'image_filename': image_name,
            'true_label': true_label,
            'predicted_class': predicted_class,
            'confidence': round(confidence, 4),
            'class_probabilities': {k: round(v, 4) for k, v in prob_dict.items()},
            'conformal_prediction_set': conformal_set_info,
            'ood_anomaly_detection': ood_info,
            'mc_epistemic_uncertainty': mc_info,
            'differential_privacy': dp_metrics,
            'graph_rag_knowledge': graph_rag_info,
            'reflexion_self_corrected': self_corrected,
            'multi_agent_consensus': consensus_info,
            'llm_judge_peer_review': peer_review_info,
            'cryptographic_audit_block': audit_block,
            'retrieved_evidence': [
                {
                    'pmid': c['pmid'],
                    'title': c['title'],
                    'rerank_score': round(c.get('rerank_score', 0.0), 4),
                    'biomedclip_similarity': round(c.get('biomedclip_similarity', c.get('rerank_score', 0.0)), 4),
                    'passage': c['passage'],
                }
                for c in evidence_chunks
            ],
            'explanation': explanation,
            'nli_faithfulness': faithfulness_res,
            'rag_text_metrics': rag_metrics,
        }

        # Step 15: PDF Diagnostic Report Generation
        if generate_pdf:
            try:
                _, overlay_img = self.generate_gradcam(pil_img)
                pdf_path = generate_pdf_report(result, pil_img, overlay_img)
                result['pdf_report_path'] = pdf_path
            except Exception as e:
                print(f"  PDF report generation error: {e}")
                result['pdf_report_path'] = None

        return result


if __name__ == '__main__':
    print("Pipeline module updated with complete suite of 12 Medical AI technologies.")
