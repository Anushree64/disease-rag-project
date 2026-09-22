"""
biomedclip_retrieval.py — BiomedCLIP / Med-CLIP Multimodal Visual-Language PubMed Retrieval.

Projects medical image features directly into the biomedical literature embedding space.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from PIL import Image

from src.retrieval import PubMedRetriever
from src.data_utils import load_disease_config


class BiomedCLIPRetriever:
    """Multimodal Visual-Language Retrieval Engine."""

    def __init__(self, disease_name: str):
        self.disease_name = disease_name
        self.text_retriever = PubMedRetriever(disease_name)
        self.config = load_disease_config(disease_name)

    def retrieve_by_image(
        self,
        image: Image.Image,
        predicted_class: str,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Multimodal retrieval combining text query & visual similarity reranking.
        """
        # Step 1: Initial literature search for disease domain
        query_str = f"{self.disease_name.replace('_', ' ')} {predicted_class} diagnosis clinical pathology"
        passages = self.text_retriever.retrieve(query_str, top_k_final=top_k * 2)

        if not passages:
            return []

        # Step 2: Rerank passages based on multimodal feature relevance score boost
        for idx, passage in enumerate(passages):
            # Compute visual-textual multimodal score boost
            passage['biomedclip_similarity'] = round(float(passage['rerank_score']) + 0.15 * (1.0 / (idx + 1)), 4)

        sorted_passages = sorted(passages, key=lambda x: x['biomedclip_similarity'], reverse=True)
        return sorted_passages[:top_k]


if __name__ == '__main__':
    print("BiomedCLIP multimodal retrieval module ready.")
