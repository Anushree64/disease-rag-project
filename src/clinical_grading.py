"""
Clinical Diagnostic Grading Standards Engine
Computes BI-RADS, ETDRS, LI-RADS, KDIGO, NYHA, and Hoehn & Yahr clinical scorecards.
"""

from typing import Dict, Any

class ClinicalGradingEngine:
    def compute_scorecard(self, disease_type: str, prediction: str, confidence: float, lesion_area_mm2: float) -> Dict[str, Any]:
        """
        Maps disease prediction and features to standard hospital grading scales.
        """
        d_lower = disease_type.lower()
        is_positive = "malignant" in prediction.lower() or "positive" in prediction.lower() or "severe" in prediction.lower() or "cad" in d_lower
        
        if "breast" in d_lower:
            if not is_positive:
                grade = "BI-RADS 2"
                desc = "Benign Finding - Non-cancerous lesion or simple cyst."
            elif confidence > 0.85 or lesion_area_mm2 > 200:
                grade = "BI-RADS 4C"
                desc = "High Suspicion for Malignancy (50%-94% probability). Biopsy recommended."
            else:
                grade = "BI-RADS 4A"
                desc = "Low Suspicion for Malignancy (2%-10% probability)."
            scale_name = "BI-RADS (Breast Imaging Reporting and Data System)"
            
        elif "diabetes" in d_lower:
            if not is_positive:
                grade = "ETDRS Level 10"
                desc = "No Diabetic Retinopathy detected."
            elif confidence > 0.80:
                grade = "ETDRS Level 61"
                desc = "Moderate Proliferative Diabetic Retinopathy with Macular Edema."
            else:
                grade = "ETDRS Level 35"
                desc = "Mild Non-Proliferative Diabetic Retinopathy."
            scale_name = "ETDRS (Early Treatment Diabetic Retinopathy Study)"

        elif "nafld" in d_lower:
            if not is_positive:
                grade = "LI-RADS 1"
                desc = "Definitely Benign Hepatic Parenchyma."
            elif confidence > 0.80:
                grade = "LI-RADS 4"
                desc = "Probably Malignant / Severe Steatohepatitis with Fibrosis."
            else:
                grade = "LI-RADS 3"
                desc = "Intermediate Probability of Steatohepatitis."
            scale_name = "LI-RADS (Liver Imaging Reporting and Data System)"

        elif "ckd" in d_lower:
            if not is_positive:
                grade = "KDIGO Stage G1 / A1"
                desc = "Normal kidney function (eGFR > 90 mL/min/1.73m²)."
            else:
                grade = "KDIGO Stage G3b / A2"
                desc = "Moderate to severe reduction in kidney function."
            scale_name = "KDIGO (Kidney Disease: Improving Global Outcomes)"

        elif "cad" in d_lower:
            if not is_positive:
                grade = "NYHA Class I"
                desc = "No limitation of physical activity."
            else:
                grade = "NYHA Class III"
                desc = "Marked limitation of physical activity; comfortable at rest."
            scale_name = "NYHA Functional Classification"

        elif "parkinson" in d_lower:
            if not is_positive:
                grade = "Hoehn & Yahr Stage 0"
                desc = "No signs of disease."
            else:
                grade = "Hoehn & Yahr Stage 2.5"
                desc = "Mild bilateral disease with recovery on pull test."
            scale_name = "Hoehn & Yahr Parkinson's Staging"

        else:
            scale_name = "General Clinical Grade"
            grade = "Grade II" if is_positive else "Grade I"
            desc = "Standard clinical severity classification."

        return {
            "scale_name": scale_name,
            "grade_code": grade,
            "clinical_description": desc,
            "biomarker_severity_index": round(min(10.0, (confidence * 7.5) + (lesion_area_mm2 / 100.0)), 2)
        }
