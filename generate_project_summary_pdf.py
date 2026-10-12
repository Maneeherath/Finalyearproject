# ============================================================
#  PDF PROJECT SUMMARY GENERATOR (UPDATED)
#  Generates a high-quality summary PDF for Progress Review II
#  Including the full 14-model training comparison results
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import os, time
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, 'Project_Progress_Review_II_Summary.pdf')

def build_pdf():
    doc = SimpleDocTemplate(OUTPUT_PATH, pagesize=letter,
                            rightMargin=45, leftMargin=45,
                            topMargin=45, bottomMargin=45)
    story = []
    styles = getSampleStyleSheet()
    
    # Palette definition
    c_primary = colors.HexColor("#1A365D")   # Deep navy
    c_secondary = colors.HexColor("#2C7A7B") # Muted teal
    c_neutral = colors.HexColor("#2D3748")   # Charcoal body
    c_light = colors.HexColor("#F7FAFC")     # Warm white background
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'],
        fontName='Helvetica-Bold', fontSize=18, leading=22,
        textColor=c_primary, spaceAfter=6, alignment=1
    )
    
    subtitle_style = ParagraphStyle(
        'DocSub', parent=styles['Normal'],
        fontName='Helvetica', fontSize=9, leading=12,
        textColor=colors.HexColor("#4A5568"), spaceAfter=10, alignment=1
    )
    
    h1_style = ParagraphStyle(
        'DocH1', parent=styles['Heading2'],
        fontName='Helvetica-Bold', fontSize=12, leading=16,
        textColor=c_secondary, spaceBefore=10, spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'DocBody', parent=styles['Normal'],
        fontName='Helvetica', fontSize=9, leading=13,
        textColor=c_neutral, spaceAfter=4
    )
    
    bold_body_style = ParagraphStyle(
        'DocBodyBold', parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    bullet_style = ParagraphStyle(
        'DocBullet', parent=body_style,
        leftIndent=15, firstLineIndent=-10, spaceAfter=3
    )

    # ── HEADER & TITLE BLOCK ──────────────────────────────────────────
    story.append(Paragraph("KDU FACULTY OF COMPUTING · INDIVIDUAL RESEARCH PROJECT (CS4999)", subtitle_style))
    story.append(Paragraph("Computational Modeling of Cognitive Performance & Gaming Behavior", title_style))
    story.append(Paragraph("Progress Review II Technical Deliverables & Evaluation Summary", subtitle_style))
    
    # Student / Supervisor Details Table
    details_data = [
        [Paragraph("<b>Student Name:</b> P R M K Herath (D/BCS/23/0009)", body_style),
         Paragraph("<b>Degree Program:</b> BSc (Hons) Computer Science", body_style)],
        [Paragraph("<b>Supervisor:</b> Dr. BVKI Vidanage", body_style),
         Paragraph("<b>Co-Supervisor:</b> Mr. Akshan Bandara", body_style)],
        [Paragraph("<b>Submission Date:</b> April 2026", body_style),
         Paragraph("<b>Current Status:</b> All Deliverables Deployed Successfully", body_style)]
    ]
    t_details = Table(details_data, colWidths=[260, 260])
    t_details.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('LINEBELOW', (0,2), (1,2), 1, c_primary),
    ]))
    story.append(t_details)
    story.append(Spacer(1, 10))
    
    # ── SECTION 1: SYSTEM OVERVIEW ────────────────────────────────────
    story.append(Paragraph("1. Project Synopsis & Objectives", h1_style))
    story.append(Paragraph(
        "Gaming disorder is an emerging psychological threat in student populations. This project develops a "
        "<b>unified, multi-modal machine learning system</b> integrating clinical self-reports, cognitive biometrics, "
        "and physical fatigue metrics to accurately identify addiction risk and trace cognitive performance levels. "
        "To fulfill the requirements of Progress Review II, we have delivered a fully integrated interactive dashboard "
        "utilizing state-of-the-art ensemble models, deep CNN models, and automated reporting.",
        body_style
    ))
    
    # ── SECTION 2: MODEL PERFORMANCE ──────────────────────────────────
    story.append(Paragraph("2. Technical Machine Learning & Evaluation Leaderboard", h1_style))
    story.append(Paragraph(
        "A rigorous classifier exploration was completed by training, tuning, and comparing 14 pipeline configurations. "
        "The models are evaluated below using Stratified 5-Fold Cross-Validation to ensure generalization:",
        body_style
    ))
    
    # Model comparison table
    model_table_data = [
        [Paragraph("<b>Rank</b>", bold_body_style), Paragraph("<b>Algorithm / Configuration</b>", bold_body_style), Paragraph("<b>Test Acc</b>", bold_body_style), Paragraph("<b>CV Acc (5-Fold)</b>", bold_body_style), Paragraph("<b>CV Std</b>", bold_body_style), Paragraph("<b>Selection Status</b>", bold_body_style)],
        [Paragraph("1", body_style), Paragraph("<b>Stacking Ensemble</b>", body_style), Paragraph("85.00%", body_style), Paragraph("83.00%", body_style), Paragraph("±4.40%", body_style), Paragraph("🏆 <b>Selected Best</b>", body_style)],
        [Paragraph("2", body_style), Paragraph("Random Forest (Base)", body_style), Paragraph("80.00%", body_style), Paragraph("82.00%", body_style), Paragraph("±5.52%", body_style), Paragraph("Superseded", body_style)],
        [Paragraph("3", body_style), Paragraph("Random Forest (Tuned)", body_style), Paragraph("81.67%", body_style), Paragraph("81.67%", body_style), Paragraph("±3.16%", body_style), Paragraph("Candidate", body_style)],
        [Paragraph("4", body_style), Paragraph("Extra Trees", body_style), Paragraph("76.67%", body_style), Paragraph("81.33%", body_style), Paragraph("±3.86%", body_style), Paragraph("Candidate", body_style)],
        [Paragraph("5", body_style), Paragraph("XGBoost (Tuned)", body_style), Paragraph("78.33%", body_style), Paragraph("81.00%", body_style), Paragraph("±5.01%", body_style), Paragraph("Candidate", body_style)],
        [Paragraph("6", body_style), Paragraph("Voting Ensemble", body_style), Paragraph("80.00%", body_style), Paragraph("81.00%", body_style), Paragraph("±4.03%", body_style), Paragraph("Candidate", body_style)],
        [Paragraph("7", body_style), Paragraph("LightGBM", body_style), Paragraph("76.67%", body_style), Paragraph("78.33%", body_style), Paragraph("±2.58%", body_style), Paragraph("Candidate", body_style)],
        [Paragraph("8", body_style), Paragraph("Gradient Boosting", body_style), Paragraph("83.33%", body_style), Paragraph("78.00%", body_style), Paragraph("±5.31%", body_style), Paragraph("Candidate", body_style)],
        [Paragraph("9", body_style), Paragraph("AdaBoost", body_style), Paragraph("73.33%", body_style), Paragraph("74.67%", body_style), Paragraph("±5.31%", body_style), Paragraph("Baseline", body_style)],
        [Paragraph("10", body_style), Paragraph("SVM (RBF)", body_style), Paragraph("76.67%", body_style), Paragraph("68.67%", body_style), Paragraph("±2.87%", body_style), Paragraph("Baseline", body_style)],
        [Paragraph("11", body_style), Paragraph("Decision Tree", body_style), Paragraph("66.67%", body_style), Paragraph("67.00%", body_style), Paragraph("±3.71%", body_style), Paragraph("Baseline", body_style)],
        [Paragraph("12", body_style), Paragraph("MLP Neural Net", body_style), Paragraph("48.33%", body_style), Paragraph("60.67%", body_style), Paragraph("±10.78%", body_style), Paragraph("Baseline", body_style)]
    ]
    t_model = Table(model_table_data, colWidths=[35, 175, 75, 95, 60, 80])
    t_model.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('BACKGROUND', (0,0), (-1,0), c_light),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor("#EBF8FF")), # Highlight Row 1
        ('PADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_model)
    story.append(Spacer(1, 8))
    
    # ── SECTION 3: BIOMETRIC AND REGRESSION IMPLEMENTATION ───────────
    story.append(Paragraph("3. Multi-Modal Regressions & Biometric Performance", h1_style))
    story.append(Paragraph("• <b>Deep ResNet50 Eye Fatigue Classifier:</b> Trained on 84,898 eye images (MRL dataset). Reached <b>66.33% accuracy</b> with an <b>AUC of 0.89</b>. Demonstrates a high fatigue detection recall rate of 93.00% to safeguard screening parameters.", bullet_style))
    story.append(Paragraph("• <b>Attention score Regressor:</b> Leverages Random Forest regression models to predict student focus. Accomplished an improved R² score of <b>0.417</b> (RMSE: 0.099).", bullet_style))
    story.append(Paragraph("• <b>Memory score Regressor:</b> Predicts short-term recall capacities. Accomplished an R² score of <b>0.176</b> (RMSE: 0.135).", bullet_style))
    story.append(Paragraph("• <b>Late Fusion Integration:</b> Aggregates classifiers, biometrics, and K-Means predictions using a confidence-weighted late fusion rule, achieving a 3.00% performance boost over baseline architectures.", bullet_style))
    
    # ── SECTION 4: NOVEL MODULES ──────────────────────────────────────
    story.append(Paragraph("4. Implemented Novelty Features (App v5 Research Edition)", h1_style))
    story.append(Paragraph(
        "To maximize research contribution and multi-modal integration, the Streamlit app was upgraded to Version 5 (Research Edition) featuring 6 core integrated modules:",
        body_style
    ))
    story.append(Paragraph("1. <b>Multimodal Late Fusion Engine:</b> Combines Tabular Stacking Ensemble, Cognitive ANN, and CNN Eye Fatigue prior using confidence-weighted log-linear opinion pooling.", bullet_style))
    story.append(Paragraph("2. <b>Live Webcam & Computer Vision Eye Scanner:</b> Captures live eye photos via OpenCV, extracting Aperture Openness, Scleral Redness, Brightness, and Contrast.", bullet_style))
    story.append(Paragraph("3. <b>Interactive 'What-If' Sensitivity Simulator:</b> Real-time habit lab showing how lifestyle modifications affect Addiction Risk %, Health Score, and Recovery Days.", bullet_style))
    story.append(Paragraph("4. <b>6-Axis Gamer Persona Radar Chart:</b> Spider-web visualization comparing individual behavioral & cognitive metrics against gamer population benchmarks.", bullet_style))
    story.append(Paragraph("5. <b>20-20-20 Eye Strain & Detox Planner:</b> Clinical ocular fatigue prevention routines, recovery timeline estimators, and daily detox schedules.", bullet_style))
    story.append(Paragraph("6. <b>Upgraded ReportLab PDF Exporter:</b> Enables users to download a signed, structured verification PDF summary report directly from the web dashboard.", bullet_style))
    
    # ── SECTION 5: HEALTH ──────────────────────────────────────────────
    story.append(Paragraph("5. Codebase Cleanup & System Architecture", h1_style))
    story.append(Paragraph(
        "A full directory refactoring was completed. Obsolete scripts and temporary stubs were safely cleaned. "
        "The project is consolidated under structured directories (<code>data/</code>, <code>models/</code>, <code>notebooks/</code>, "
        "and <code>results/</code>). The main application is deployed via <code>app_v5.py</code> using a virtualized <code>.venv</code>.",
        body_style
    ))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Report Certified By:</b> Final Year Research Project Team | Intake 40", bold_body_style))
    
    doc.build(story)
    print(f"[SUCCESS] PDF generated at: {OUTPUT_PATH}")

if __name__ == '__main__':
    build_pdf()
