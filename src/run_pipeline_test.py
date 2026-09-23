"""
run_pipeline_test.py — Step 9 & Step 10 execution.

1. Evaluates the full end-to-end pipeline on at least 10 test images per class per disease.
2. Saves structured results to results/full_pipeline_results.json.
3. Prints a summary table: per-disease accuracy, average confidence, average NLI faithfulness.
4. Generates results_summary.txt formatted with section headers, citations, explanations, and scores.
"""

import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

from src.data_utils import BASE_DIR, load_disease_config, find_class_dir
from src.pipeline import DiseaseRAGPipeline, RESULTS_DIR


def select_test_images(disease_name: str, images_per_class: int = 10) -> List[Dict]:
    """
    Select up to `images_per_class` test images for each class of a disease.
    Uses deterministically sorted paths with fixed random seed.
    """
    config = load_disease_config(disease_name)
    root_path = BASE_DIR / config['root_path']
    classes = config['classes']

    random.seed(42)
    selected_samples = []

    for cls_name in classes:
        cls_dir = find_class_dir(root_path, cls_name)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        all_imgs = sorted([
            p for p in cls_dir.rglob('*')
            if p.is_file() and p.suffix.lower() in image_extensions
        ])

        # Take last 15% (simulating test set partition) to ensure unseen test images
        split_start = int(len(all_imgs) * 0.85)
        test_candidates = all_imgs[split_start:] if split_start < len(all_imgs) else all_imgs

        # Pick images_per_class samples
        chosen = random.sample(test_candidates, min(images_per_class, len(test_candidates)))

        for img_path in chosen:
            selected_samples.append({
                'disease': disease_name,
                'image_path': str(img_path),
                'true_label': cls_name,
            })

    return selected_samples


def run_full_pipeline_evaluation(diseases: List[str] = None, samples_per_class: int = 10) -> Dict:
    """Run full pipeline evaluation across all diseases."""
    diseases = diseases or ['breast_cancer', 'cad', 'diabetes']

    all_results = {}
    summary_metrics = {}

    for disease in diseases:
        print(f"\n{'='*70}")
        print(f"  RUNNING FULL PIPELINE EVALUATION: {disease.upper()}")
        print(f"{'='*70}")

        try:
            pipeline = DiseaseRAGPipeline(disease)
            samples = select_test_images(disease, images_per_class=samples_per_class)
            print(f"  Selected {len(samples)} test images across {len(pipeline.classes)} classes.")

            disease_runs = []
            correct_count = 0
            confidences = []
            faithfulness_scores = []

            for idx, sample in enumerate(samples, 1):
                img_path = sample['image_path']
                true_label = sample['true_label']

                print(f"  [{idx}/{len(samples)}] Processing: {Path(img_path).name} (True: {true_label})...")
                res = pipeline.run(img_path, true_label=true_label, top_k_evidence=3)
                disease_runs.append(res)

                if res['predicted_class'] == true_label:
                    correct_count += 1

                confidences.append(res['confidence'])
                faithfulness_scores.append(res['nli_faithfulness']['average_faithfulness'])

            accuracy = correct_count / len(samples) if samples else 0.0
            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            avg_faith = float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0

            summary_metrics[disease] = {
                'total_samples': len(samples),
                'correct_predictions': correct_count,
                'accuracy': round(accuracy, 4),
                'avg_confidence': round(avg_conf, 4),
                'avg_faithfulness': round(avg_faith, 4),
            }

            all_results[disease] = {
                'summary': summary_metrics[disease],
                'runs': disease_runs,
            }

            print(f"\n  [OK] {disease.upper()} Evaluation Summary:")
            print(f"     Accuracy:         {accuracy*100:.2f}%")
            print(f"     Avg Confidence:   {avg_conf*100:.2f}%")
            print(f"     Avg Faithfulness: {avg_faith:.4f}")

        except Exception as e:
            print(f"  [ERROR] Failed evaluation for {disease}: {e}")
            import traceback
            traceback.print_exc()

    # Save structured JSON
    json_path = RESULTS_DIR / 'full_pipeline_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Full pipeline results saved to: {json_path}")

    # Generate summary text report
    generate_summary_text_report(all_results)

    # Print overall summary table
    print_summary_table(summary_metrics)

    return all_results


def print_summary_table(summary_metrics: Dict):
    """Print ASCII summary table."""
    table_data = []
    for disease, metrics in summary_metrics.items():
        table_data.append([
            disease.replace('_', ' ').title(),
            metrics['total_samples'],
            f"{metrics['accuracy']*100:.2f}%",
            f"{metrics['avg_confidence']*100:.2f}%",
            f"{metrics['avg_faithfulness']:.4f}",
        ])

    headers = ["Disease", "Test Samples", "Accuracy", "Avg Confidence", "Avg NLI Faithfulness"]
    print("\n" + "="*70)
    print("  OVERALL PIPELINE PERFORMANCE SUMMARY TABLE")
    print("="*70)
    if HAS_TABULATE:
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        print(f"{'Disease':<20} | {'Samples':<8} | {'Accuracy':<10} | {'Avg Conf':<10} | {'Avg Faith':<10}")
        print("-" * 65)
        for row in table_data:
            print(f"{row[0]:<20} | {row[1]:<8} | {row[2]:<10} | {row[3]:<10} | {row[4]:<10}")


def generate_summary_text_report(all_results: Dict):
    """Generate results_summary.txt with human-readable formatting."""
    report_path = BASE_DIR / 'results_summary.txt'

    lines = []
    lines.append("================================================================================")
    lines.append("      MULTI-DISEASE CLASSIFICATION + RAG EXPLANATION PIPELINE RESULTS SUMMARY")
    lines.append("================================================================================\n")

    for disease, data in all_results.items():
        summary = data['summary']
        lines.append(f"########################################################################")
        lines.append(f" DISEASE: {disease.replace('_', ' ').upper()}")
        lines.append(f" Total Test Images Evaluated: {summary['total_samples']}")
        lines.append(f" Classification Accuracy:      {summary['accuracy']*100:.2f}% ({summary['correct_predictions']}/{summary['total_samples']})")
        lines.append(f" Average Confidence Score:    {summary['avg_confidence']*100:.2f}%")
        lines.append(f" Average NLI Faithfulness:    {summary['avg_faithfulness']:.4f}")
        lines.append(f"########################################################################\n")

        lines.append("SAMPLE PREDICTIONS & GENERATED EXPLANATIONS:\n")

        for idx, run in enumerate(data['runs'], 1):
            lines.append(f"--- Sample [{idx}/{len(data['runs'])}] ---")
            lines.append(f"Image File:        {run['image_filename']}")
            lines.append(f"True Label:        {run['true_label']}")
            lines.append(f"Predicted Class:   {run['predicted_class']} (Confidence: {run['confidence']*100:.2f}%)")
            lines.append(f"Class Probs:       {run['class_probabilities']}")
            lines.append(f"NLI Faithfulness:  {run['nli_faithfulness']['average_faithfulness']:.4f}")
            lines.append("\nRetrieved Biomedical Evidence (PubMed Abstracts):")

            for ev in run['retrieved_evidence']:
                lines.append(f"  • [PMID {ev['pmid']}] {ev['title']}")
                lines.append(f"    Rerank Score: {ev['rerank_score']:.4f}")
                lines.append(f"    Excerpt: {ev['passage'][:180]}...")

            lines.append("\nGenerated Clinical Explanation:")
            lines.append(f"  \"{run['explanation']}\"")
            lines.append("\nPer-Sentence Entailment Scores:")
            for ps in run['nli_faithfulness'].get('per_sentence', []):
                lines.append(f"  - Sentence: \"{ps['sentence']}\"")
                lines.append(f"    Entailment: {ps['entailment_score']:.4f} (Matched PMID: {ps['matched_pmid']})")

            lines.append("\n" + "-"*60 + "\n")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"  Human-readable summary report saved to: {report_path}")


if __name__ == '__main__':
    run_full_pipeline_evaluation()
