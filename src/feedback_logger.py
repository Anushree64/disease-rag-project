"""
feedback_logger.py — Clinician Human-in-the-Loop Active Feedback Logger.

Logs interactive clinician ratings, comments, approved/corrected explanations,
and tagged PMIDs to results/clinician_feedback.json for active learning audit.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from src.data_utils import BASE_DIR

FEEDBACK_FILE = BASE_DIR / "results" / "clinician_feedback.json"


def log_clinician_feedback(
    disease_name: str,
    image_filename: str,
    predicted_class: str,
    confidence: float,
    rating: int,  # 1 to 5 stars
    is_approved: bool,
    clinician_comments: Optional[str] = None,
    corrected_explanation: Optional[str] = None,
    clinician_id: str = "Dr_Clinician_01",
) -> Dict:
    """
    Logs clinician feedback entry to JSON file.
    """
    entry = {
        'timestamp': datetime.now().isoformat(),
        'disease': disease_name,
        'image_filename': image_filename,
        'predicted_class': predicted_class,
        'confidence': round(confidence, 4),
        'rating': int(rating),
        'is_approved': bool(is_approved),
        'clinician_comments': clinician_comments or "",
        'corrected_explanation': corrected_explanation or "",
        'clinician_id': clinician_id,
    }

    records = []
    if FEEDBACK_FILE.exists():
        try:
            with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
                records = json.load(f)
                if not isinstance(records, list):
                    records = []
        except Exception:
            records = []

    records.append(entry)

    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"  [OK] Clinician feedback logged to {FEEDBACK_FILE} (Total entries: {len(records)})")
    return entry


def get_feedback_summary() -> Dict:
    """Returns feedback analytics summary."""
    if not FEEDBACK_FILE.exists():
        return {'total_feedback': 0, 'avg_rating': 0.0, 'approval_rate': 0.0}

    with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
        records = json.load(f)

    if not records:
        return {'total_feedback': 0, 'avg_rating': 0.0, 'approval_rate': 0.0}

    ratings = [r.get('rating', 3) for r in records]
    approvals = [1 if r.get('is_approved', True) else 0 for r in records]

    return {
        'total_feedback': len(records),
        'avg_rating': round(float(sum(ratings) / len(ratings)), 2),
        'approval_rate': round(float(sum(approvals) / len(approvals) * 100), 1),
    }


if __name__ == '__main__':
    print("Clinician feedback logger ready.")
