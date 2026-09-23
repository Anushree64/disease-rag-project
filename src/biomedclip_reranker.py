"""
biomedclip_reranker.py — Zero-Shot Cross-Modal Image-to-Literature BiomedCLIP Reranker.

Reranks retrieved PubMed literature by aligning visual image embeddings directly
with multimodal biomedical text/figure representations using BiomedCLIP cross-modal scoring.
"""

import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.biomedclip_retrieval import BiomedCLIPRetriever


class CrossModalBiomedCLIPReranker:
    """Reranks retrieved evidence passages using visual-language BiomedCLIP embedding similarity."""

    def __init__(self, disease_name: str):
        self.disease_name = disease_name
        self.retriever = BiomedCLIPRetriever(disease_name)

    def rerank_evidence_with_image(
        self,
        image_input,
        evidence_list: List[Dict],
        top_k: int = 3
    ) -> List[Dict]:
        """
        Rerank evidence articles using BiomedCLIP visual-text similarity.
        """
        if not evidence_list:
            return []

        passages = [ev.get('passage', ev.get('title', '')) for ev in evidence_list]
        reranked = self.retriever.rerank_documents_with_image(image_input, passages, top_k=top_k)

        scored_evidence = []
        for item in reranked:
            idx = item['index']
            if idx < len(evidence_list):
                ev = dict(evidence_list[idx])
                ev['biomedclip_crossmodal_score'] = item['biomedclip_similarity_score']
                scored_evidence.append(ev)

        return scored_evidence


if __name__ == '__main__':
    reranker = CrossModalBiomedCLIPReranker('breast_cancer')
    print("  [OK] Cross-Modal BiomedCLIP Reranker ready.")
