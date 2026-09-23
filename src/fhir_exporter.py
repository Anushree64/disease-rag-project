"""
fhir_exporter.py — FHIR R4 HL7 Electronic Health Record (EHR) Standard Export Engine.

Converts AI diagnostic pipeline outputs into compliant FHIR R4 DiagnosticReport
and Observation JSON resources for hospital EHR integration (Epic, Cerner, Allscripts).
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_utils import BASE_DIR


class FHIREHRExporter:
    """Export pipeline results into FHIR R4 compliant JSON resources."""

    def __init__(self, system_uri: str = "http://ai-diagnostic-framework.org/fhir"):
        self.system_uri = system_uri

    def export_fhir_report(self, pipeline_result: Dict, patient_id: str = "PATIENT-99482") -> Dict:
        """
        Build FHIR R4 DiagnosticReport resource JSON.
        """
        now = datetime.utcnow().isoformat() + "Z"
        disease = pipeline_result.get('disease', 'breast_cancer')
        pred_cls = pipeline_result.get('predicted_class', 'Unknown')
        conf = float(pipeline_result.get('confidence', 0.0))
        cp_info = pipeline_result.get('conformal_prediction_set', {})

        report_id = f"diag-rep-{disease}-{patient_id}"

        fhir_resource = {
            "resourceType": "DiagnosticReport",
            "id": report_id,
            "meta": {
                "versionId": "1",
                "lastUpdated": now,
                "profile": ["http://hl7.org/fhir/StructureDefinition/DiagnosticReport"]
            },
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                            "code": "RAD",
                            "display": "Radiology"
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "47528-5",
                        "display": f"{disease.replace('_', ' ').title()} Diagnostic Imaging AI Report"
                    }
                ],
                "text": f"AI Diagnostic Report for {disease}"
            },
            "subject": {
                "reference": f"Patient/{patient_id}",
                "display": f"Patient ID: {patient_id}"
            },
            "effectiveDateTime": now,
            "issued": now,
            "conclusion": f"Predicted Diagnosis: {pred_cls} (Confidence: {conf*100:.1f}%). Conformal Set: {cp_info.get('prediction_set', [])}.",
            "result": [
                {
                    "reference": f"Observation/obs-prediction-{patient_id}",
                    "display": "Primary AI Class Probability Observation"
                },
                {
                    "reference": f"Observation/obs-conformal-{patient_id}",
                    "display": "Split Conformal 95% Coverage Uncertainty Observation"
                }
            ],
            "contained": [
                {
                    "resourceType": "Observation",
                    "id": f"obs-prediction-{patient_id}",
                    "status": "final",
                    "code": {
                        "text": "Primary AI Class Prediction Probability"
                    },
                    "valueQuantity": {
                        "value": round(conf * 100, 2),
                        "unit": "%",
                        "system": "http://unitsofmeasure.org",
                        "code": "%"
                    }
                },
                {
                    "resourceType": "Observation",
                    "id": f"obs-conformal-{patient_id}",
                    "status": "final",
                    "code": {
                        "text": "Conformal Set Size"
                    },
                    "valueInteger": cp_info.get('set_size', 1)
                }
            ],
            "extension": [
                {
                    "url": f"{self.system_uri}/StructureDefinition/nli-faithfulness",
                    "valueDecimal": round(pipeline_result.get('nli_faithfulness', {}).get('average_faithfulness', 0.0), 4)
                },
                {
                    "url": f"{self.system_uri}/StructureDefinition/audit-signature",
                    "valueString": pipeline_result.get('audit_ledger_block', {}).get('sha256_signature', 'N/A')
                }
            ]
        }

        output_path = BASE_DIR / "results" / f"fhir_report_{disease}_{patient_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(fhir_resource, f, indent=2)

        print(f"  [OK] Exported FHIR R4 DiagnosticReport to: {output_path}")
        return fhir_resource


if __name__ == '__main__':
    exporter = FHIREHRExporter()
    sample_res = {
        'disease': 'breast_cancer',
        'predicted_class': 'malignant',
        'confidence': 0.985,
        'conformal_prediction_set': {'prediction_set': ['malignant'], 'set_size': 1},
        'nli_faithfulness': {'average_faithfulness': 0.924},
        'audit_ledger_block': {'sha256_signature': 'abc123xyz456'}
    }
    exporter.export_fhir_report(sample_res)
