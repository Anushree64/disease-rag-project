"""
generation.py — Natural-language explanation generation using FLAN-T5.

Takes:
- disease_name
- predicted_class
- confidence score
- retrieved_evidence (passages with titles and PMIDs)

Generates:
- Structured biomedical explanation grounded in the retrieved literature.
"""

from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

_FLAN_TOKENIZER = None
_FLAN_MODEL = None
_MODEL_NAME = "google/flan-t5-base"


def get_flan_model():
    global _FLAN_TOKENIZER, _FLAN_MODEL
    if _FLAN_MODEL is None:
        print(f"  Loading explanation generator ('{_MODEL_NAME}')...")
        _FLAN_TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _FLAN_MODEL = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME)
    return _FLAN_TOKENIZER, _FLAN_MODEL


def generate_explanation(
    disease_name: str,
    predicted_class: str,
    confidence: float,
    evidence_chunks: List[Dict],
    max_length: int = 250,
) -> str:
    """
    Generate an explanation grounded in retrieved PubMed evidence.
    """
    tokenizer, model = get_flan_model()

    # Format context from top evidence
    evidence_text = ""
    for idx, chunk in enumerate(evidence_chunks[:3], 1):
        passage = chunk.get('passage', chunk.get('text', ''))
        pmid = chunk.get('pmid', 'Unknown')
        evidence_text += f"Evidence [{idx}] (PMID:{pmid}): {passage}\n"

    prompt = (
        f"You are a clinical decision support assistant. Explain the following diagnostic prediction using the scientific evidence provided.\n\n"
        f"Disease Context: {disease_name.replace('_', ' ').title()}\n"
        f"Prediction: {predicted_class} (Confidence: {confidence*100:.1f}%)\n\n"
        f"Scientific Evidence:\n{evidence_text}\n"
        f"Task: Write a concise, professional clinical explanation summarizing why the image features match the predicted class '{predicted_class}' based on the evidence provided above."
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_beams=2,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    explanation = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Fallback if FLAN-T5 output is brief
    if len(explanation.split()) < 15:
        top_passage = evidence_chunks[0]['passage'] if evidence_chunks else "Standard diagnostic criteria apply."
        top_pmid = evidence_chunks[0].get('pmid', 'N/A') if evidence_chunks else 'N/A'
        explanation = (
            f"The image was classified as '{predicted_class}' with {confidence*100:.1f}% confidence. "
            f"Biomedical literature (PMID:{top_pmid}) indicates: {top_passage}"
        )

    return explanation


if __name__ == '__main__':
    # Simple test
    test_evidence = [{
        'pmid': '34567890',
        'passage': 'Malignant breast lesions characteristically demonstrate irregular margins, posterior acoustic shadowing, and non-parallel orientation on ultrasound.',
    }]
    exp = generate_explanation('breast_cancer', 'malignant', 0.985, test_evidence)
    print("Generated Explanation:\n", exp)
