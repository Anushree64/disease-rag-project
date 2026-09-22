"""
faithfulness.py — NLI-based Faithfulness & Hallucination Scoring.

Uses 'cross-encoder/nli-deberta-v3-base' (or 'cross-encoder/nli-distilroberta-base') to score whether generated explanation sentences are entailed by the retrieved evidence.

Outputs:
- per_sentence_scores: List of dicts with sentence, entailment_prob, neutral_prob, contradiction_prob
- average_faithfulness: float (mean entailment score across sentences)
"""

import re
from typing import Dict, List, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_NLI_TOKENIZER = None
_NLI_MODEL = None
_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"


def get_nli_model():
    global _NLI_TOKENIZER, _NLI_MODEL
    if _NLI_MODEL is None:
        print(f"  Loading NLI faithfulness scorer ('{_MODEL_NAME}')...")
        try:
            _NLI_TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_NAME)
            _NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        except Exception as e:
            print(f"  ⚠️  Failed to load {_MODEL_NAME}: {e}. Falling back to 'cross-encoder/nli-distilroberta-base'...")
            fallback = "cross-encoder/nli-distilroberta-base"
            _NLI_TOKENIZER = AutoTokenizer.from_pretrained(fallback)
            _NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(fallback)

    return _NLI_TOKENIZER, _NLI_MODEL


def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def evaluate_faithfulness(
    explanation: str,
    evidence_chunks: List[Dict],
) -> Dict:
    """
    Check NLI entailment of each sentence in explanation against premises in evidence_chunks.

    Returns:
    {
      'average_faithfulness': float,  # Mean entailment score (0.0 - 1.0)
      'per_sentence': [
         {
           'sentence': str,
           'best_entailment_prob': float,
           'best_evidence_pmid': str,
         }
      ]
    }
    """
    tokenizer, model = get_nli_model()

    sentences = split_sentences(explanation)
    if not sentences or not evidence_chunks:
        return {
            'average_faithfulness': 1.0,
            'per_sentence': [],
        }

    # Prepare evidence premise texts
    premises = [
        chunk.get('passage', chunk.get('text', ''))
        for chunk in evidence_chunks
    ]
    pmids = [chunk.get('pmid', 'Unknown') for chunk in evidence_chunks]

    per_sentence_results = []
    entailment_scores = []

    # Map model label IDs to entailment/neutral/contradiction
    # DeBERTa-v3 NLI mapping: 0=contradiction, 1=neutral, 2=entailment
    id2label = model.config.id2label

    for sent in sentences:
        best_entailment = 0.0
        best_pmid = pmids[0] if pmids else "N/A"

        for premise, pmid in zip(premises, pmids):
            inputs = tokenizer(premise, sent, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

            # Find index of entailment label
            entail_idx = None
            for idx, label in id2label.items():
                if 'entail' in label.lower():
                    entail_idx = idx
                    break

            if entail_idx is None:
                entail_idx = 2 if len(probs) == 3 else 1

            entail_prob = float(probs[entail_idx])

            if entail_prob > best_entailment:
                best_entailment = entail_prob
                best_pmid = pmid

        entailment_scores.append(best_entailment)
        per_sentence_results.append({
            'sentence': sent,
            'entailment_score': round(best_entailment, 4),
            'matched_pmid': best_pmid,
        })

    avg_faithfulness = float(np.mean(entailment_scores)) if entailment_scores else 1.0

    return {
        'average_faithfulness': round(avg_faithfulness, 4),
        'per_sentence': per_sentence_results,
    }


if __name__ == '__main__':
    test_evidence = [{
        'pmid': '12345678',
        'passage': 'Ultrasound features of malignant breast masses include microcalcifications, ill-defined margins, and vertical orientation.'
    }]
    test_exp = "Malignant breast masses display microcalcifications and ill-defined margins on ultrasound examination."

    res = evaluate_faithfulness(test_exp, test_evidence)
    print("Faithfulness Evaluation Results:")
    print("Avg Faithfulness:", res['average_faithfulness'])
    for s in res['per_sentence']:
        print("  Sentence:", s['sentence'])
        print("  Entailment Score:", s['entailment_score'], "PMID:", s['matched_pmid'])
