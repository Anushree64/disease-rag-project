"""
icd_snomed_mapper.py — Automated ICD-10-CM & SNOMED CT Medical Coding Mapper Engine.

Maps predicted clinical disease diagnoses and sub-classifications to official
ICD-10-CM diagnostic billing codes and SNOMED CT clinical terminology concepts.
"""

import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Comprehensive ICD-10 & SNOMED CT Medical Code Database
MEDICAL_CODE_DATABASE = {
    'breast_cancer': {
        'malignant': {
            'icd10_code': 'C50.911',
            'icd10_title': 'Malignant neoplasm of unspecified site of right female breast',
            'snomed_ct_id': '254837009',
            'snomed_ct_term': 'Malignant neoplasm of breast (disorder)',
        },
        'benign': {
            'icd10_code': 'D24.1',
            'icd10_title': 'Benign neoplasm of right breast',
            'snomed_ct_id': '254843006',
            'snomed_ct_term': 'Benign neoplasm of breast (disorder)',
        }
    },
    'cad': {
        'abnormal': {
            'icd10_code': 'I25.10',
            'icd10_title': 'Atherosclerotic heart disease of native coronary artery without angina pectoris',
            'snomed_ct_id': '53741008',
            'snomed_ct_term': 'Coronary arteriosclerosis (disorder)',
        },
        'normal': {
            'icd10_code': 'Z01.810',
            'icd10_title': 'Encounter for preprocedural cardiovascular examination',
            'snomed_ct_id': '17621005',
            'snomed_ct_term': 'Normal cardiovascular system (finding)',
        }
    },
    'diabetes': {
        'Moderate DR': {
            'icd10_code': 'E11.3399',
            'icd10_title': 'Type 2 diabetes mellitus with moderate nonproliferative diabetic retinopathy',
            'snomed_ct_id': '1551000119106',
            'snomed_ct_term': 'Moderate nonproliferative diabetic retinopathy due to type 2 diabetes mellitus',
        },
        'Severe DR': {
            'icd10_code': 'E11.3499',
            'icd10_title': 'Type 2 diabetes mellitus with severe nonproliferative diabetic retinopathy',
            'snomed_ct_id': '1561000119108',
            'snomed_ct_term': 'Severe nonproliferative diabetic retinopathy due to type 2 diabetes mellitus',
        },
        'Healthy': {
            'icd10_code': 'Z00.00',
            'icd10_title': 'Encounter for general adult medical examination without abnormal findings',
            'snomed_ct_id': '108329005',
            'snomed_ct_term': 'Normal fundus of eye (finding)',
        }
    },
    'ckd': {
        'Tumor': {
            'icd10_code': 'C64.1',
            'icd10_title': 'Malignant neoplasm of right kidney, except renal pelvis',
            'snomed_ct_id': '254932002',
            'snomed_ct_term': 'Malignant neoplasm of kidney (disorder)',
        },
        'Stone': {
            'icd10_code': 'N20.0',
            'icd10_title': 'Calculus of kidney (Nephrolithiasis)',
            'snomed_ct_id': '95570007',
            'snomed_ct_term': 'Calculus of kidney (disorder)',
        },
        'Cyst': {
            'icd10_code': 'N28.1',
            'icd10_title': 'Cyst of kidney, acquired',
            'snomed_ct_id': '236439005',
            'snomed_ct_term': 'Renal cyst (disorder)',
        },
        'Normal': {
            'icd10_code': 'Z01.89',
            'icd10_title': 'Encounter for other specified special examinations',
            'snomed_ct_id': '1761000119102',
            'snomed_ct_term': 'Normal renal structure (finding)',
        }
    },
    'nafld': {
        'fatty_liver': {
            'icd10_code': 'K76.0',
            'icd10_title': 'Fatty (change of) liver, not elsewhere classified (NAFLD)',
            'snomed_ct_id': '197321007',
            'snomed_ct_term': 'Nonalcoholic fatty liver (disorder)',
        },
        'normal': {
            'icd10_code': 'Z00.00',
            'icd10_title': 'Encounter for general medical examination',
            'snomed_ct_id': '176150005',
            'snomed_ct_term': 'Normal liver structure (finding)',
        }
    },
    'parkinsons': {
        'parkinson': {
            'icd10_code': 'G20.A1',
            'icd10_title': 'Parkinson disease without dyskinesia, without motor fluctuations',
            'snomed_ct_id': '49049000',
            'snomed_ct_term': 'Parkinson\'s disease (disorder)',
        },
        'healthy': {
            'icd10_code': 'Z00.00',
            'icd10_title': 'Encounter for general medical examination',
            'snomed_ct_id': '225606002',
            'snomed_ct_term': 'Normal motor function (finding)',
        }
    }
}


class ICDSNOMEDMapper:
    """Map disease predictions to official ICD-10 and SNOMED CT medical codes."""

    def get_medical_codes(self, disease_name: str, predicted_class: str) -> Dict:
        """Retrieve ICD-10 and SNOMED CT codes."""
        domain_db = MEDICAL_CODE_DATABASE.get(disease_name.lower(), {})
        code_info = domain_db.get(predicted_class, None)

        if not code_info:
            # Default fallback
            code_info = {
                'icd10_code': 'R69',
                'icd10_title': f"Illness, unspecified associated with {disease_name}",
                'snomed_ct_id': '404684003',
                'snomed_ct_term': f"Clinical finding related to {disease_name}",
            }

        return {
            'disease': disease_name,
            'predicted_class': predicted_class,
            'icd10_code': code_info['icd10_code'],
            'icd10_title': code_info['icd10_title'],
            'snomed_ct_id': code_info['snomed_ct_id'],
            'snomed_ct_term': code_info['snomed_ct_term'],
            'billing_reimbursement_eligible': True,
        }


if __name__ == '__main__':
    mapper = ICDSNOMEDMapper()
    res = mapper.get_medical_codes('breast_cancer', 'malignant')
    print("  [OK] ICD-10 & SNOMED CT Codes:", res)
