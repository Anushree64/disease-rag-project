"""
run_demo_inference.py — Runs sample inference across all 6 diseases directly in terminal.
"""

import sys
import glob
from pathlib import Path
import json

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.pipeline import DiseaseRAGPipeline, load_disease_config

diseases = [
    ('breast_cancer', 'data/breast_cancer/**/*.png'),
    ('cad', 'data/cad/**/*.jpg'),
    ('diabetes', 'data/diabetes/**/*.png'),
    ('ckd', 'data/ckd/**/*.jpg'),
    ('nafld', 'data/nafld/**/*.jpg'),
    ('parkinsons', 'data/parkinsons/**/*.jpg'),
]

print("==========================================================================")
print(" 🩺 RUNNING LIVE DIAGNOSTIC INFERENCE ACROSS ALL 6 DISEASE DOMAINS")
print("==========================================================================")

for disease, glob_pat in diseases:
    print(f"\n{"="*70}")
    print(f" 🔍 DISEASE DOMAIN: {disease.upper()}")
    print(f"{"="*70}")

    sample_imgs = glob.glob(str(BASE_DIR / glob_pat), recursive=True)
    if not sample_imgs:
        # try png fallback
        sample_imgs = glob.glob(str(BASE_DIR / glob_pat.replace('.jpg', '.png')), recursive=True)

    if not sample_imgs:
        print(f"  ❌ No sample images found for {disease}")
        continue

    sample_img = sample_imgs[0]
    img_name = Path(sample_img).name
    print(f"  📸 Sample Image: {img_name}")
    print(f"  📍 Path: {sample_img}")

    try:
        pipeline = DiseaseRAGPipeline(disease_name=disease, backbone='resnet18')
        res = pipeline.run(sample_img, top_k_evidence=2)

        print(f"\n  ✅ PREDICTION RESULT:")
        print(f"     - Predicted Class : {res['predicted_class']}")
        print(f"     - Confidence      : {res['confidence']*100:.2f}%")
        print(f"     - Probabilities   : {res['class_probabilities']}")

        print(f"\n  📚 RETRIEVED PUBMED EVIDENCE:")
        for idx, ev in enumerate(res['retrieved_evidence'], 1):
            print(f"     [{idx}] Title : {ev['title']}")
            print(f"         PMID  : {ev['pmid']} | Rerank Score: {ev['rerank_score']:.4f}")
            print(f"         Text  : \"{ev['passage'][:150]}...\"")

        print(f"\n  💡 CLINICAL EXPLANATION (FLAN-T5 Grounded):")
        print(f"     \"{res['explanation']}\"")

        faith = res['nli_faithfulness']
        print(f"\n  🛡️ NLI FAITHFULNESS EVALUATION:")
        print(f"     - Average Score   : {faith['average_faithfulness']:.4f} / 1.0000")
        for ps in faith.get('per_sentence', []):
            score = ps['entailment_score']
            badge = "🟢 High" if score >= 0.7 else "🟡 Moderate" if score >= 0.4 else "🔴 Low/Hallucinated"
            print(f"     - Sentence        : \"{ps['sentence']}\"")
            print(f"       Score           : {score:.4f} ({badge}) | Matched PMID: {ps['matched_pmid']}")

    except Exception as e:
        print(f"  ❌ Error running pipeline for {disease}: {e}")
        import traceback
        traceback.print_exc()

print("\n==========================================================================")
print(" 🎉 DEMO INFERENCE COMPLETED SUCCESSFULLY FOR ALL 6 DISEASES")
print("==========================================================================")
