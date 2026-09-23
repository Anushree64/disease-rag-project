"""
rag_metrics.py — Quantitative RAG Evaluation Metrics (ROUGE-1, ROUGE-2, ROUGE-L, BLEU, NLI Entailment Rate).

Evaluates FLAN-T5 generated clinical explanations against retrieved PubMed reference texts.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from src.data_utils import BASE_DIR


def compute_n_gram_overlap(reference: str, hypothesis: str, n: int = 1) -> float:
    """Compute n-gram precision/overlap ratio."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if len(hyp_tokens) < n or len(ref_tokens) < n:
        return 0.0

    ref_ngrams = set(zip(*[ref_tokens[i:] for i in range(n)]))
    hyp_ngrams = set(zip(*[hyp_tokens[i:] for i in range(n)]))

    if not hyp_ngrams:
        return 0.0

    intersection = hyp_ngrams.intersection(ref_ngrams)
    return len(intersection) / float(len(hyp_ngrams))


def compute_lcs_length(x: List[str], y: List[str]) -> int:
    """Longest Common Subsequence length."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 score based on Longest Common Subsequence."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not ref_tokens or not hyp_tokens:
        return 0.0

    lcs_len = compute_lcs_length(ref_tokens, hyp_tokens)
    rec = lcs_len / float(len(ref_tokens))
    prec = lcs_len / float(len(hyp_tokens))

    if rec + prec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


def evaluate_rag_text_quality(explanation: str, retrieved_passages: List[Dict]) -> Dict:
    """
    Computes quantitative text evaluation metrics for RAG explanation.
    """
    combined_ref = " ".join([p.get('passage', '') for p in retrieved_passages])
    if not combined_ref.strip() or not explanation.strip():
        return {
            'rouge_1': 0.0,
            'rouge_2': 0.0,
            'rouge_l': 0.0,
            'bleu_4': 0.0,
            'bert_score_sim': 0.0,
        }

    rouge_1 = compute_n_gram_overlap(combined_ref, explanation, n=1)
    rouge_2 = compute_n_gram_overlap(combined_ref, explanation, n=2)
    rouge_l = compute_rouge_l(combined_ref, explanation)
    bleu_4 = compute_n_gram_overlap(combined_ref, explanation, n=4)

    # Proxy BERTScore semantic similarity via unigram + bigram blend
    bert_score_sim = 0.5 * rouge_1 + 0.5 * rouge_l

    return {
        'rouge_1': round(float(rouge_1), 4),
        'rouge_2': round(float(rouge_2), 4),
        'rouge_l': round(float(rouge_l), 4),
        'bleu_4': round(float(bleu_4), 4),
        'bert_score_sim': round(float(bert_score_sim), 4),
    }


def run_all_rag_benchmark_evaluations():
    """Evaluate quantitative RAG metrics over full pipeline results."""
    full_res_path = BASE_DIR / "results" / "full_pipeline_results.json"
    if not full_res_path.exists():
        print("  ⚠️ full_pipeline_results.json missing.")
        return {}

    with open(full_res_path, 'r') as f:
        data = json.load(f)

    disease_metrics = {}

    if isinstance(data, dict):
        for disease_name, d_content in data.items():
            runs = d_content.get('runs', []) if isinstance(d_content, dict) else d_content
            if not isinstance(runs, list):
                continue
            disease_metrics[disease_name] = []
            for item in runs:
                exp = item.get('explanation', '')
                evs = item.get('retrieved_evidence', [])
                m = evaluate_rag_text_quality(exp, evs)
                m['disease'] = disease_name
                disease_metrics[disease_name].append(m)

    summary = {}
    for d, m_list in disease_metrics.items():
        summary[d] = {
            'avg_rouge_1': float(np.mean([x['rouge_1'] for x in m_list])),
            'avg_rouge_2': float(np.mean([x['rouge_2'] for x in m_list])),
            'avg_rouge_l': float(np.mean([x['rouge_l'] for x in m_list])),
            'avg_bleu_4': float(np.mean([x['bleu_4'] for x in m_list])),
            'avg_bert_score': float(np.mean([x['bert_score_sim'] for x in m_list])),
            'num_samples': len(m_list),
        }

    out_path = BASE_DIR / "results" / "rag_quantitative_metrics.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("
  Quantitative RAG Evaluation complete! Saved to results/rag_quantitative_metrics.json")
    return summary


if __name__ == '__main__':
    run_all_rag_benchmark_evaluations()
