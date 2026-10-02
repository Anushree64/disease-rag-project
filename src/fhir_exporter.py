"""
fhir_exporter.py — FHIR R4 HL7 Electronic Health Record (EHR) Standard Export Engine.

Converts AI diagnostic pipeline outputs into compliant FHIR R4 DiagnosticReport
and Observation JSON resources for hospital EHR integration.
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

    def export_fhir_report(self, pipeline_result: Dict, patient_id: Optional[str] = None) -> Dict:
        """
        Build FHIR R4 DiagnosticReport resource JSON.
        """
        now = datetime.utcnow().isoformat() + "Z"
        disease = pipeline_result.get('disease', 'breast_cancer')
        pred_cls = pipeline_result.get('predicted_class', 'Unknown')
        conf = float(pipeline_result.get('confidence', 0.0))
        cp_info = pipeline_result.get('conformal_prediction_set', {})

        if not patient_id:
            patient_id = f"anonymous-case-{datetime.now().strftime('%Y%m%d%H%M%S')}"

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
                        "display": f"{disease.replace('_', ' ').title()} Diagnostic Report"
                    }
                ]
            },
            "subject": {
                "reference": f"Patient/{patient_id}"
            },
            "effectiveDateTime": now,
            "issued": now,
            "conclusion": f"Diagnostic prediction: {pred_cls} ({conf*100:.1f}% confidence). Conformal prediction set: {cp_info.get('prediction_set', [])}.",
            "conclusionCode": [
                {
                    "coding": [
                        {
                            "system": self.system_uri,
                            "code": pred_cls,
                            "display": pred_cls.title()
                        }
                    ]
                }
            ]
        }

        return fhir_resource
