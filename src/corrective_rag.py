"""
corrective_rag.py — Implements 3-Path Corrective RAG Policy.

Paths:
1. Path 1 (Score >= 0.70): High confidence direct retrieval pass-through.
2. Path 2 (0.30 <= Score < 0.70): Medium confidence query expansion & re-retrieval.
3. Path 3 (Score < 0.30): Low confidence fallback to static medical domain knowledge base.
"""

from typing import Dict, List, Tuple, Optional
from src.retrieval import PubMedRetriever, fetch_pubmed_abstracts
from src.data_utils import load_disease_config


STATIC_KNOWLEDGE_BASE = {
    'breast_cancer': [
        {
            'pmid': 'STATIC_KB_BC_01',
            'title': 'Clinical Guidelines for Mammographic and Ultrasound Diagnosis of Breast Carcinoma',
            'passage': 'Malignant breast lesions typically present as irregular, microlobulated, or spiculated hypoechoic masses with posterior acoustic shadowing and microcalcifications. Benign lesions are usually oval, circumscribed, and wider-than-tall.',
            'rerank_score': 0.85,
        }
    ],
    'cad': [
        {
            'pmid': 'STATIC_KB_CAD_01',
            'title': 'Echocardiographic and Angiographic Features of Coronary Artery Disease',
            'passage': 'Coronary Artery Disease (CAD) leads to myocardial ischemia causing regional wall motion abnormalities on echocardiography and luminal stenosis >50% on coronary angiograms.',
            'rerank_score': 0.85,
        }
    ],
    'diabetes': [
        {
            'pmid': 'STATIC_KB_DM_01',
            'title': 'Diabetic Retinopathy Classification Guidelines',
            'passage': 'Diabetic retinopathy features include microaneurysms, hemorrhages, hard exudates, cotton wool spots, and neovascularization. Severity ranges from mild non-proliferative to proliferative retinopathy.',
            'rerank_score': 0.85,
        }
    ],
    'ckd': [
        {
            'pmid': 'STATIC_KB_CKD_01',
            'title': 'Diagnostic Imaging of Chronic Kidney Disease and Renal Masses',
            'passage': 'CT Kidney imaging differentiates normal renal parenchyma from simple fluid-filled cysts, dense renal stones with acoustic shadowing, and solid enhancing renal cell carcinoma tumors.',
            'rerank_score': 0.85,
        }
    ],
    'nafld': [
        {
            'pmid': 'STATIC_KB_NAFLD_01',
            'title': 'Ultrasound Assessment of Hepatic Steatosis (NAFLD)',
            'passage': 'Hepatic steatosis (fatty liver) on ultrasound exhibits diffuse hyperechoic liver parenchyma (bright liver), increased contrast between liver and kidney, and vascular blurring.',
            'rerank_score': 0.85,
        }
    ],
    'parkinsons': [
        {
            'pmid': 'STATIC_KB_PD_01',
            'title': 'Kinematic and Spiral Drawing Analysis in Parkinson Disease',
            'passage': 'Parkinson disease patient drawings exhibit severe micrographia, tremor oscillations, irregular line width, and fluency disruption compared to smooth control spiral drawings.',
            'rerank_score': 0.85,
        }
    ],
}


class CorrectiveRAGRetriever:
    """3-Path Corrective Retrieval Engine."""

    def __init__(self, disease_name: str):
        self.disease_name = disease_name
        self.retriever = PubMedRetriever(disease_name)
        self.config = load_disease_config(disease_name)

    def retrieve_with_corrective_policy(
        self,
        query: str,
        top_k: int = 3,
        high_threshold: float = 0.70,
        low_threshold: float = 0.30,
    ) -> Tuple[List[Dict], str, Dict]:
        """
        Executes 3-path corrective policy:
        Returns (passages, policy_status, metadata).
        """
        initial_passages = self.retriever.retrieve(query, top_k_final=top_k)
        max_score = initial_passages[0]['rerank_score'] if initial_passages else 0.0

        metadata = {
            'initial_max_score': float(max_score),
            'initial_query': query,
            'path_taken': '',
        }

        # Path 1: High Confidence
        if max_score >= high_threshold:
            metadata['path_taken'] = 'Path 1: High Confidence Direct Retrieval'
            return initial_passages, 'HIGH_CONFIDENCE_DIRECT', metadata

        # Path 2: Medium/Ambiguous Confidence -> Query Expansion
        elif max_score >= low_threshold:
            expanded_query = f"{query} clinical features pathology diagnosis guidelines"
            metadata['path_taken'] = 'Path 2: Medium Confidence Query Expansion'
            metadata['expanded_query'] = expanded_query

            expanded_passages = self.retriever.retrieve(expanded_query, top_k_final=top_k)
            # Combine & deduplicate by pmid/chunk_id
            combined = {p.get('chunk_id', p['pmid']): p for p in (initial_passages + expanded_passages)}
            sorted_combined = sorted(combined.values(), key=lambda x: x['rerank_score'], reverse=True)
            return sorted_combined[:top_k], 'MEDIUM_CONFIDENCE_EXPANDED', metadata

        # Path 3: Low/Irrelevant -> Static Domain Knowledge Fallback
        else:
            metadata['path_taken'] = 'Path 3: Static Domain Knowledge Fallback'
            static_passages = STATIC_KNOWLEDGE_BASE.get(
                self.disease_name,
                [{'pmid': 'STATIC_0', 'title': 'Medical Reference', 'passage': query, 'rerank_score': 0.80}]
            )
            return static_passages[:top_k], 'LOW_CONFIDENCE_STATIC_KNOWLEDGE', metadata
