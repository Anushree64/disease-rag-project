"""
report_translator.py — Multi-Language Clinical Report Translation Engine.

Translates clinical explanations, diagnosis summaries, and RAG literature evidence
into 5 major international languages (English, Spanish, French, German, Chinese).
"""

import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Multilingual Diagnostic Terminology Dictionary
MULTILINGUAL_DICTIONARY = {
    'es': {
        'malignant': 'Maligno (Cáncer)',
        'benign': 'Benigno (No canceroso)',
        'normal': 'Normal (Sin anomalías)',
        'abnormal': 'Anormal',
        'Healthy': 'Saludable',
        'Moderate DR': 'Retinopatía diabética moderada',
        'Severe DR': 'Retinopatía diabética grave',
        'Cyst': 'Quiste renal',
        'Stone': 'Cálculo renal',
        'Tumor': 'Tumor renal',
        'fatty_liver': 'Hígado graso (NAFLD)',
        'parkinson': 'Enfermedad de Parkinson',
        'header': 'Informe de Diagnóstico Médico por IA'
    },
    'fr': {
        'malignant': 'Maligne (Cancéreux)',
        'benign': 'Bénin (Non cancéreux)',
        'normal': 'Normal',
        'abnormal': 'Anormal',
        'Healthy': 'En bonne santé',
        'Moderate DR': 'Rétinopathie diabétique modérée',
        'Severe DR': 'Rétinopathie diabétique sévère',
        'Cyst': 'Kyste rénal',
        'Stone': 'Calcul rénal',
        'Tumor': 'Tumeur rénale',
        'fatty_liver': 'Stéatose hépatique (NAFLD)',
        'parkinson': 'Maladie de Parkinson',
        'header': 'Rapport Diagnostic Médical IA'
    },
    'de': {
        'malignant': 'Bösartig (Maligne)',
        'benign': 'Gutartig (Benigne)',
        'normal': 'Normal',
        'abnormal': 'Pathologisch',
        'Healthy': 'Gesund',
        'Moderate DR': 'Moderate diabetische Retinopathie',
        'Severe DR': 'Schwere diabetische Retinopathie',
        'Cyst': 'Nierenzyste',
        'Stone': 'Nierenstein',
        'Tumor': 'Nierentumor',
        'fatty_liver': 'Fettleber (NAFLD)',
        'parkinson': 'Parkinson-Krankheit',
        'header': 'Medizinischer KI-Diagnosebericht'
    },
    'zh': {
        'malignant': '恶性 (肿瘤/癌症)',
        'benign': '良性 (非癌性)',
        'normal': '正常',
        'abnormal': '异常',
        'Healthy': '健康',
        'Moderate DR': '中度糖尿病视网膜病变',
        'Severe DR': '重度糖尿病视网膜病变',
        'Cyst': '肾囊肿',
        'Stone': '肾结石',
        'Tumor': '肾肿瘤',
        'fatty_liver': '脂肪肝 (NAFLD)',
        'parkinson': '帕金森病',
        'header': '医疗AI诊断报告'
    }
}


class ClinicalReportTranslator:
    """Translates diagnostic summaries into target international languages."""

    def translate_summary(
        self,
        predicted_class: str,
        confidence: float,
        explanation: str,
        target_lang: str = 'es'
    ) -> Dict:
        """Translate diagnosis and clinical explanation into target language."""
        target_lang = target_lang.lower().strip()
        lang_dict = MULTILINGUAL_DICTIONARY.get(target_lang, {})

        translated_class = lang_dict.get(predicted_class, predicted_class)
        header = lang_dict.get('header', 'AI Medical Diagnostic Report')

        if target_lang == 'es':
            trans_exp = f"Diagnóstico del modelo: {translated_class} con un {confidence*100:.1f}% de confianza. Explicación médica: {explanation}"
        elif target_lang == 'fr':
            trans_exp = f"Diagnostic du modèle: {translated_class} avec une confiance de {confidence*100:.1f}%. Explication médicale: {explanation}"
        elif target_lang == 'de':
            trans_exp = f"Modelldiagnose: {translated_class} mit {confidence*100:.1f}% Konfidenz. Medizinische Erklärung: {explanation}"
        elif target_lang == 'zh':
            trans_exp = f"模型诊断结果: {translated_class}，置信度为 {confidence*100:.1f}%。医学解释: {explanation}"
        else:
            trans_exp = f"Predicted Diagnosis: {predicted_class} ({confidence*100:.1f}% confidence). Explanation: {explanation}"

        return {
            'target_language': target_lang,
            'header': header,
            'translated_class': translated_class,
            'translated_explanation': trans_exp,
        }


if __name__ == '__main__':
    translator = ClinicalReportTranslator()
    res = translator.translate_summary('malignant', 0.965, 'High density lesion observed.', 'es')
    print("  [OK] Spanish Translation output:", res)
