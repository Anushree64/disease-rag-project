"""
report_generator.py — Automated Clinical PDF Diagnostic Report Generator.

Compiles complete diagnostic reports into PDF format featuring:
- Disease domain & patient metadata
- Original medical scan & Grad-CAM visual heatmap overlay
- Classification probabilities & 95% Conformal Prediction Set
- FLAN-T5 Grounded Clinical Explanation
- NLI Faithfulness Score & Sentence Breakdown
- RAG Text Evaluation Metrics (ROUGE, BLEU, BERTScore)
- Retrieved PubMed Literature Evidence
"""

import os
from pathlib import Path
from typing import Dict, Union, Optional
from PIL import Image

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.units import inch

from src.data_utils import BASE_DIR

REPORTS_DIR = BASE_DIR / "results" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_pdf_report(
    pipeline_result: Dict,
    original_image: Image.Image,
    gradcam_image: Image.Image,
    output_filename: Optional[str] = None,
) -> str:
    """
    Generates a structured clinical PDF report and returns the absolute path.
    """
    disease_name = pipeline_result.get('disease', 'disease').replace('_', ' ').title()
    image_filename = pipeline_result.get('image_filename', 'scan.png')

    if output_filename is None:
        safe_name = pipeline_result.get('disease', 'disease')
        output_path = REPORTS_DIR / f"{safe_name}_diagnostic_report.pdf"
    else:
        output_path = REPORTS_DIR / output_filename

    # Save temporary images for ReportLab insertion
    temp_dir = REPORTS_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    orig_img_path = temp_dir / "orig_temp.png"
    cam_img_path = temp_dir / "cam_temp.png"

    original_image.convert('RGB').resize((224, 224)).save(orig_img_path)
    gradcam_image.convert('RGB').resize((224, 224)).save(cam_img_path)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        alignment=0,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0284c7"),
        spaceBefore=8,
        spaceAfter=6,
    )
    section_heading = ParagraphStyle(
        'DocSection',
        parent=styles['Heading3'],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1f2937"),
        spaceAfter=4,
    )
    quote_style = ParagraphStyle(
        'DocQuote',
        parent=styles['Normal'],
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor("#334155"),
        leftIndent=10,
        spaceAfter=4,
    )

    story = []

    # System Title Header Card
    story.append(Paragraph("EXPLAINABLE AI MULTI-DISEASE CLINICAL DECISION SUPPORT SYSTEM", ParagraphStyle('SysHeader', parent=styles['Normal'], fontSize=8.5, leading=10, textColor=colors.HexColor("#0284c7"), spaceAfter=2)))
    story.append(Paragraph(f"Clinical Diagnostic & Grounded RAG Report — {disease_name}", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0284c7"), spaceAfter=10))

    # Patient & Model Overview Table
    pred_cls = pipeline_result.get('predicted_class', 'N/A').upper()
    conf = pipeline_result.get('confidence', 0.0) * 100
    cp_info = pipeline_result.get('conformal_prediction_set', {})
    cp_set = str(cp_info.get('prediction_set', []))
    review_status = "Specialist Review Required (Multi-Class Set)" if cp_info.get('requires_human_review', False) else "Singleton Prediction (95% Coverage Target Met)"

    meta_data = [
        [Paragraph("<b>Disease Domain:</b>", body_style), Paragraph(disease_name, body_style),
         Paragraph("<b>Scan Filename:</b>", body_style), Paragraph(image_filename, body_style)],
        [Paragraph("<b>Predicted Class:</b>", body_style), Paragraph(f"<b><font color='#0284c7'>{pred_cls}</font></b>", body_style),
         Paragraph("<b>Model Confidence:</b>", body_style), Paragraph(f"<b>{conf:.1f}%</b>", body_style)],
        [Paragraph("<b>95% Conformal Set C(X):</b>", body_style), Paragraph(f"<code>{cp_set}</code>", body_style),
         Paragraph("<b>Conformal Status:</b>", body_style), Paragraph(f"<b>{review_status}</b>", body_style)],
    ]
    meta_table = Table(meta_data, colWidths=[1.4*inch, 2.2*inch, 1.4*inch, 2.3*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Visual Heatmap Section
    story.append(Paragraph("Visual Diagnostic Imaging & Grad-CAM Heatmap", subtitle_style))
    img_table_data = [
        [RLImage(str(orig_img_path), width=2.4*inch, height=2.4*inch),
         RLImage(str(cam_img_path), width=2.4*inch, height=2.4*inch)],
        [Paragraph("<b>Original Medical Input Scan</b>", body_style), Paragraph("<b>Grad-CAM ROI Saliency Heatmap</b>", body_style)]
    ]
    img_table = Table(img_table_data, colWidths=[3.6*inch, 3.6*inch])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 10))

    # Explanation Section
    story.append(Paragraph("AI Clinical Explanation (FLAN-T5 Grounded RAG)", subtitle_style))
    exp_text = pipeline_result.get('explanation', 'No explanation generated.')
    story.append(Paragraph(f"<i>\"{exp_text}\"</i>", body_style))
    story.append(Spacer(1, 8))

    # Faithfulness & Metrics Summary
    faith_info = pipeline_result.get('nli_faithfulness', {})
    faith_score = faith_info.get('average_faithfulness', 0.0)
    rag_metrics = pipeline_result.get('rag_text_metrics', {})

    story.append(Paragraph("Faithfulness & Quantitative Text Metrics", section_heading))
    metrics_data = [
        [Paragraph("<b>NLI Faithfulness Score:</b>", body_style), Paragraph(f"<b>{faith_score:.4f} / 1.0000</b>", body_style),
         Paragraph("<b>ROUGE-1:</b>", body_style), Paragraph(f"{rag_metrics.get('rouge_1', 0.0):.4f}", body_style)],
        [Paragraph("<b>BLEU-4:</b>", body_style), Paragraph(f"{rag_metrics.get('bleu_4', 0.0):.4f}", body_style),
         Paragraph("<b>BERTScore Similarity:</b>", body_style), Paragraph(f"{rag_metrics.get('bert_score_sim', 0.0):.4f}", body_style)],
    ]
    metrics_table = Table(metrics_data, colWidths=[1.8*inch, 1.7*inch, 1.8*inch, 1.9*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f1f5f9")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 10))

    # Retrieved Evidence Section
    story.append(Paragraph("Retrieved PubMed Literature Evidence", subtitle_style))
    for idx, ev in enumerate(pipeline_result.get('retrieved_evidence', []), 1):
        title = ev.get('title', 'PubMed Article')
        pmid = ev.get('pmid', 'N/A')
        passage = ev.get('passage', '')
        story.append(Paragraph(f"<b>[{idx}] {title}</b> (PMID: {pmid})", body_style))
        story.append(Paragraph(f"\"{passage}\"", quote_style))
        story.append(Spacer(1, 4))

    doc.build(story)

    # Cleanup temp files
    try:
        orig_img_path.unlink(missing_ok=True)
        cam_img_path.unlink(missing_ok=True)
    except Exception:
        pass

    print(f"  [OK] Generated PDF Diagnostic Report: {output_path}")
    return str(output_path)


if __name__ == '__main__':
    print("Report generator module ready.")
