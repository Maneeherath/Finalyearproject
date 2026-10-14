# ============================================================
#  PROGRESS REVIEW II — PRESENTATION GENERATOR (UPDATED)
#  Student: P R M K Herath | D/BCS/23/0009
#  Generates a .pptx file with project summary slides
# ============================================================

import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
OUTPUT_PATH = os.path.join(BASE_DIR, 'Progress_Review_II_Presentation.pptx')

# ── Color Palette ────────────────────────────────────────────
BG_DARK      = RGBColor(0x0F, 0x17, 0x2A)
BG_CARD      = RGBColor(0x1E, 0x29, 0x3B)
ACCENT_BLUE  = RGBColor(0x38, 0xBD, 0xF8)
ACCENT_GREEN = RGBColor(0x4A, 0xDE, 0x80)
ACCENT_AMBER = RGBColor(0xFB, 0xBF, 0x24)
ACCENT_RED   = RGBColor(0xF8, 0x71, 0x71)
ACCENT_PURPLE= RGBColor(0xA7, 0x8B, 0xFA)
WHITE        = RGBColor(0xFF, 0xFF, 0xFF)
GRAY_LIGHT   = RGBColor(0xCB, 0xD5, 0xE1)
GRAY_MED     = RGBColor(0x94, 0xA3, 0xB8)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# ── Helper Functions ─────────────────────────────────────────
def add_dark_background(slide):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = BG_DARK

def add_textbox(slide, left, top, width, height, text, font_size=18, color=WHITE, bold=False, alignment=PP_ALIGN.LEFT, font_name='Calibri'):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return tf

def add_paragraph(tf, text, font_size=16, color=WHITE, bold=False, alignment=PP_ALIGN.LEFT, font_name='Calibri', space_before=Pt(4), space_after=Pt(4)):
    p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    p.space_before = space_before
    p.space_after = space_after
    return p

def add_rounded_rect(slide, left, top, width, height, fill_color=BG_CARD):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape

def add_accent_bar(slide, left, top, width=0.08, height=0.8, color=ACCENT_BLUE):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


# ═══════════════════════════════════════════════════════════════
#  SLIDE 1: TITLE SLIDE
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_dark_background(slide)
add_accent_bar(slide, 0, 0, width=13.333, height=0.06, color=ACCENT_BLUE)

add_textbox(slide, 1.0, 1.8, 11.3, 1.2,
    "Gaming Behavior & Cognitive Performance Predictor",
    font_size=40, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

add_textbox(slide, 1.0, 3.0, 11.3, 0.8,
    "A Multimodal Late Fusion Approach with Clinical Neuropsychological Assessment",
    font_size=22, color=ACCENT_BLUE, bold=False, alignment=PP_ALIGN.CENTER)

add_textbox(slide, 1.0, 4.2, 11.3, 0.5,
    "Progress Review II",
    font_size=20, color=ACCENT_AMBER, bold=True, alignment=PP_ALIGN.CENTER)

tf = add_textbox(slide, 1.0, 5.0, 11.3, 0.4,
    "P R M K Herath  |  D/BCS/23/0009",
    font_size=18, color=GRAY_LIGHT, bold=False, alignment=PP_ALIGN.CENTER)
add_paragraph(tf, "BSc (Hons) Computer Science  |  General Sir John Kotelawala Defence University",
    font_size=14, color=GRAY_MED, alignment=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════
#  SLIDE 2: AGENDA / OUTLINE
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_dark_background(slide)
add_accent_bar(slide, 0, 0, width=13.333, height=0.06, color=ACCENT_BLUE)

add_textbox(slide, 0.8, 0.4, 5, 0.6, "Presentation Outline", font_size=32, color=WHITE, bold=True)

items = [
    ("01", "Recap of Progress Review I", ACCENT_BLUE),
    ("02", "System Architecture & Methodology", ACCENT_PURPLE),
    ("03", "Machine Learning Pipeline", ACCENT_AMBER),
    ("04", "Novelty: Multimodal Late Fusion Engine", ACCENT_BLUE),
    ("05", "Novelty: Clinical Neuropsychological Tests", ACCENT_GREEN),
    ("06", "Novelty: Live Webcam Eye Fatigue Scanner", ACCENT_RED),
    ("07", "Results & Evaluation Metrics", ACCENT_AMBER),
    ("08", "Web Application Demo (Streamlit v5)", ACCENT_PURPLE),
    ("09", "Conclusion & Future Work", ACCENT_BLUE),
]

for i, (num, title, color) in enumerate(items):
    y = 1.3 + i * 0.6
    add_accent_bar(slide, 1.0, y, width=0.06, height=0.45, color=color)
    add_textbox(slide, 1.3, y, 1, 0.45, num, font_size=18, color=color, bold=True)
    add_textbox(slide, 2.0, y, 9, 0.45, title, font_size=18, color=WHITE)


# ═══════════════════════════════════════════════════════════════
#  SLIDE 3: RECAP OF PROGRESS REVIEW I
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_dark_background(slide)
add_accent_bar(slide, 0, 0, width=13.333, height=0.06, color=ACCENT_BLUE)

add_textbox(slide, 0.8, 0.4, 8, 0.6, "01  Recap of Progress Review I", font_size=32, color=WHITE, bold=True)

# Left Card: Achievements in Review I
add_rounded_rect(slide, 0.8, 1.3, 5.8, 5.5)
add_accent_bar(slide, 0.8, 1.3, width=0.06, height=5.5, color=ACCENT_GREEN)

tf = add_textbox(slide, 1.2, 1.5, 5.2, 0.5, "Completed in Review I", font_size=22, color=ACCENT_GREEN, bold=True)
add_paragraph(tf, "   Problem Identification & Scope:", font_size=16, color=WHITE, bold=True)
add_paragraph(tf, "     • Defined gaming addiction impact on KDU undergrads", font_size=14, color=GRAY_LIGHT)
add_paragraph(tf, "   Data Collection & Tabular Model Baseline:", font_size=16, color=WHITE, bold=True)
add_paragraph(tf, "     • Primary survey of 300 student records", font_size=14, color=GRAY_LIGHT)
add_paragraph(tf, "     • Benchmarked 14 ML models -> RF baseline established (83.0% CV Acc)", font_size=14, color=GRAY_LIGHT)
add_paragraph(tf, "   Initial Prototype:", font_size=16, color=WHITE, bold=True)
add_paragraph(tf, "     • basic Streamlit UI with static survey risk calculation", font_size=14, color=GRAY_LIGHT)

# Right Card: Panel Feedback & Progress Made for Review II
add_rounded_rect(slide, 7.0, 1.3, 5.5, 5.5)
add_accent_bar(slide, 7.0, 1.3, width=0.06, height=5.5, color=ACCENT_PURPLE)

tf = add_textbox(slide, 7.4, 1.5, 4.9, 0.5, "Feedback & Expansion (Review II)", font_size=22, color=ACCENT_PURPLE, bold=True)
add_paragraph(tf, "   Panel Recommendations Received:", font_size=16, color=WHITE, bold=True)
add_paragraph(tf, "     1. Move beyond single-modality survey models", font_size=14, color=GRAY_LIGHT)
add_paragraph(tf, "     2. Add objective cognitive & visual measurement", font_size=14, color=GRAY_LIGHT)
add_paragraph(tf, "     3. Implement formal multimodal data fusion", font_size=14, color=GRAY_LIGHT)
add_paragraph(tf, "", font_size=6, color=GRAY_LIGHT)
add_paragraph(tf, "   Major Progress Delivered for Review II:", font_size=16, color=ACCENT_AMBER, bold=True)
add_paragraph(tf, "     • Log-Linear Late Fusion Engine (+3% accuracy gain)", font_size=14, color=WHITE)
add_paragraph(tf, "     • 4 Clinical Neuropsych Tests (Stroop, Flanker, DSST, Grid)", font_size=14, color=WHITE)
add_paragraph(tf, "     • Live Webcam Eye Fatigue Detector (CNN 93.4% Acc)", font_size=14, color=WHITE)
add_paragraph(tf, "     • Advanced Streamlit v5 Web Application & PDF Engine", font_size=14, color=WHITE)


# ═══════════════════════════════════════════════════════════════
#  SLIDE 4: SYSTEM ARCHITECTURE
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_dark_background(slide)
add_accent_bar(slide, 0, 0, width=13.333, height=0.06, color=ACCENT_PURPLE)

add_textbox(slide, 0.8, 0.4, 10, 0.6, "02  System Architecture - Multimodal Late Fusion", font_size=32, color=WHITE, bold=True)

labels_in = [
    ("Tabular Data\n(Survey & Demographics)", "Random Forest\nClassifier", ACCENT_BLUE),
    ("Cognitive Tests\n(Stroop, Flanker, DSST)", "Keras ANN\nMulti-Output Regressor", ACCENT_GREEN),
    ("Visual Data\n(Webcam / Eye Image)", "CNN (ResNet50)\nEye Fatigue Prior", ACCENT_AMBER),
]

for i, (inp, model, color) in enumerate(labels_in):
    x = 0.8 + i * 4.2
    add_rounded_rect(slide, x, 1.3, 3.8, 1.2)
    add_accent_bar(slide, x, 1.3, width=0.06, height=1.2, color=color)
    add_textbox(slide, x + 0.2, 1.35, 3.4, 1.1, inp, font_size=14, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    add_textbox(slide, x + 1.4, 2.55, 1, 0.4, "v", font_size=24, color=color, alignment=PP_ALIGN.CENTER)
    add_rounded_rect(slide, x, 3.0, 3.8, 1.0)
    add_textbox(slide, x + 0.2, 3.05, 3.4, 0.9, model, font_size=14, color=color, bold=True, alignment=PP_ALIGN.CENTER)

for i in range(3):
    x = 0.8 + i * 4.2
    add_textbox(slide, x + 1.4, 4.05, 1, 0.4, "v", font_size=24, color=ACCENT_PURPLE, alignment=PP_ALIGN.CENTER)

add_rounded_rect(slide, 2.5, 4.5, 8.3, 1.2, fill_color=RGBColor(0x2D, 0x1B, 0x4E))
add_accent_bar(slide, 2.5, 4.5, width=8.3, height=0.06, color=ACCENT_PURPLE)
add_textbox(slide, 2.7, 4.6, 7.9, 0.5, "Multimodal Late Fusion Engine", font_size=22, color=ACCENT_PURPLE, bold=True, alignment=PP_ALIGN.CENTER)
add_textbox(slide, 2.7, 5.1, 7.9, 0.5, "Confidence-Weighted Log-Linear Opinion Pooling  |  Dynamic Model Trust Allocation", font_size=14, color=GRAY_LIGHT, alignment=PP_ALIGN.CENTER)

add_textbox(slide, 6.2, 5.75, 1, 0.4, "v", font_size=24, color=ACCENT_GREEN, alignment=PP_ALIGN.CENTER)

add_rounded_rect(slide, 3.5, 6.2, 6.3, 1.0, fill_color=RGBColor(0x14, 0x2E, 0x1A))
add_accent_bar(slide, 3.5, 6.2, width=6.3, height=0.06, color=ACCENT_GREEN)
add_textbox(slide, 3.7, 6.3, 5.9, 0.8, "Final Risk: Low / Medium / High  |  Health Score  |  Gamer Archetype  |  PDF Report", font_size=15, color=ACCENT_GREEN, bold=False, alignment=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════
#  SLIDE 5: ML PIPELINE
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_dark_background(slide)
add_accent_bar(slide, 0, 0, width=13.333, height=0.06, color=ACCENT_BLUE)

add_textbox(slide, 0.8, 0.4, 10, 0.6, "03  Machine Learning Pipeline - Tabular Model", font_size=32, color=WHITE, bold=True)

add_rounded_rect(slide, 0.8, 1.3, 5.8, 2.8)
add_accent_bar(slide, 0.8, 1.3, width=0.06, height=2.8, color=ACCENT_BLUE)
tf = add_textbox(slide, 1.2, 1.4, 5.2, 0.4, "Dataset Overview", font_size=20, color=ACCENT_BLUE, bold=True)
add_paragraph(tf, "   Source: Gaming Behavior Survey (KDU Students)", font_size=15, color=GRAY_LIGHT)
add_paragraph(tf, "   Samples: 300 student records", font_size=15, color=GRAY_LIGHT)
add_paragraph(tf, "   Features: 17 (demographics, gaming habits, sleep, GPA)", font_size=15, color=GRAY_LIGHT)
add_paragraph(tf, "   Target: addiction_level (Low / Medium / High)", font_size=15, color=GRAY_LIGHT)
add_paragraph(tf, "   Preprocessing: Label Encoding + One-Hot + MinMaxScaler", font_size=15, color=GRAY_LIGHT)

add_rounded_rect(slide, 7.0, 1.3, 5.5, 2.8)
add_accent_bar(slide, 7.0, 1.3, width=0.06, height=2.8, color=ACCENT_GREEN)
tf = add_textbox(slide, 7.4, 1.4, 4.9, 0.4, "Model Selection (14 Compared)", font_size=20, color=ACCENT_GREEN, bold=True)
add_paragraph(tf, "   Compared 14 classifiers including SVM, KNN, LR, DT, GB, XGBoost, RF, etc.", font_size=15, color=GRAY_LIGHT)
add_paragraph(tf, "   Best Model: Stacking Ensemble (Meta-Learner)", font_size=15, color=ACCENT_GREEN, bold=True)
add_paragraph(tf, "   5-Fold CV Accuracy: 83.00% (+/-4.4%)", font_size=15, color=WHITE, bold=True)
add_paragraph(tf, "   Feature Importance: total_hours_week, sleep_hours, GPA dominate", font_size=15, color=GRAY_LIGHT)

chart_files = [
    ('chartA_cross_validation.png', 'Cross-Validation Results'),
    ('chartB_feature_importance.png', 'Feature Importance'),
    ('chartC_cv_boxplot.png', 'CV Accuracy Boxplot')
]

for i, (fname, title) in enumerate(chart_files):
    x = 0.8 + i * 4.2
    path = os.path.join(RESULTS_DIR, fname)
    if os.path.exists(path):
        slide.shapes.add_picture(path, Inches(x), Inches(4.4), width=Inches(3.8), height=Inches(2.8))
    add_textbox(slide, x, 4.15, 3.8, 0.3, title, font_size=12, color=GRAY_MED, alignment=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════
#  SLIDE 6: NOVELTY — LATE FUSION ENGINE
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_dark_background(slide)
add_accent_bar(slide, 0, 0, width=13.333, height=0.06, color=ACCENT_PURPLE)

add_textbox(slide, 0.8, 0.4, 10, 0.6, "04  Novelty - Multimodal Late Fusion Engine", font_size=32, color=WHITE, bold=True)

add_rounded_rect(slide, 0.8, 1.3, 6.0, 5.8)
add_accent_bar(slide, 0.8, 1.3, width=0.06, height=5.8, color=ACCENT_PURPLE)

tf = add_textbox(slide, 1.2, 1.4, 5.4, 0.5, "Why Late Fusion?", font_size=22, color=ACCENT_PURPLE, bold=True)
add_paragraph(tf, "Unlike early fusion (concatenating raw features), late fusion preserves the independence of each modality's model.", font_size=15, color=GRAY_LIGHT)
add_paragraph(tf, "", font_size=8, color=GRAY_LIGHT)
add_paragraph(tf, "Algorithm: Log-Linear Opinion Pooling", font_size=20, color=ACCENT_AMBER, bold=True)
add_paragraph(tf, "P_fused(y)  =  Product of P_k(y)^(w_k)", font_size=18, color=WHITE, bold=True)
add_paragraph(tf, "Where w_k = confidence weight of model k", font_size=14, color=GRAY_MED)
add_paragraph(tf, "", font_size=8, color=GRAY_LIGHT)
add_paragraph(tf, "Key Properties:", font_size=18, color=ACCENT_BLUE, bold=True)
add_paragraph(tf, "   Confident models get higher weights automatically", font_size=15, color=GRAY_LIGHT)
add_paragraph(tf, "   Uncertain modalities are gracefully suppressed", font_size=15, color=GRAY_LIGHT)
add_paragraph(tf, "   Missing modalities default to uniform priors", font_size=15, color=GRAY_LIGHT)
add_paragraph(tf, "   Mathematically grounded in Bayesian probability", font_size=15, color=GRAY_LIGHT)

add_rounded_rect(slide, 7.2, 1.3, 5.3, 2.5)
add_accent_bar(slide, 7.2, 1.3, width=0.06, height=2.5, color=ACCENT_GREEN)
tf = add_textbox(slide, 7.6, 1.4, 4.7, 0.5, "Fusion Evaluation Results", font_size=20, color=ACCENT_GREEN, bold=True)
add_paragraph(tf, "RF-Only Mean Accuracy:          76.3%", font_size=16, color=GRAY_LIGHT)
add_paragraph(tf, "Fused Realistic Accuracy:         79.3%", font_size=16, color=WHITE, bold=True)
add_paragraph(tf, "Fused Best-Case Accuracy:       79.0%", font_size=16, color=GRAY_LIGHT)
add_paragraph(tf, "Improvement:                              +3.0%", font_size=18, color=ACCENT_GREEN, bold=True)

add_rounded_rect(slide, 7.2, 4.1, 5.3, 3.0)
add_accent_bar(slide, 7.2, 4.1, width=0.06, height=3.0, color=ACCENT_BLUE)
tf = add_textbox(slide, 7.6, 4.2, 4.7, 0.5, "Three Voting Modalities", font_size=20, color=ACCENT_BLUE, bold=True)
add_paragraph(tf, "1. RF Vote (Tabular)", font_size=16, color=ACCENT_BLUE, bold=True)
add_paragraph(tf, "   predict_proba -> 3-class probability vector", font_size=13, color=GRAY_LIGHT)
add_paragraph(tf, "2. Cognitive ANN Vote", font_size=16, color=ACCENT_GREEN, bold=True)
add_paragraph(tf, "   Attention + Memory -> Softmax evidence classifier", font_size=13, color=GRAY_LIGHT)
add_paragraph(tf, "3. Eye Fatigue CNN Prior", font_size=16, color=ACCENT_AMBER, bold=True)
add_paragraph(tf, "   Fatigue score -> Heuristic probability mapping", font_size=13, color=GRAY_LIGHT)


# ═══════════════════════════════════════════════════════════════
#  SLIDE 7: NOVELTY — CLINICAL COGNITIVE TESTS
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_dark_background(slide)
add_accent_bar(slide, 0, 0, width=13.333, height=0.06, color=ACCENT_GREEN)

add_textbox(slide, 0.8, 0.4, 10, 0.6, "05  Novelty - Clinical Neuropsychological Tests", font_size=32, color=WHITE, bold=True)

tests = [
    ("Stroop Interference Test", "Inhibitory Control & Selective Attention", "Identify FONT COLOR while ignoring the word text. Measures frontal lobe cognitive fatigue - the gold standard in clinical neuropsychology.", ACCENT_BLUE),
    ("Visuospatial Grid Memory", "Working Memory Capacity", "Memorize 5 highlighted tiles in a 4x4 grid, then reproduce the pattern. Evaluates gaming-critical spatial working memory.", ACCENT_GREEN),
    ("Eriksen Flanker Task", "Spatial Selective Attention & Reaction Speed", "Identify the direction of the CENTER arrow amid distracting flankers (e.g., <<><<). Measures the brain's ability to filter peripheral noise.", ACCENT_AMBER),
    ("Symbol-Digit Substitution (DSST)", "Information Processing Speed", "Decode a symbol to number mapping as fast as possible. A sensitive clinical probe for sustained attention, visual scanning, and motor coordination.", ACCENT_PURPLE),
]

for i, (name, domain, desc, color) in enumerate(tests):
    y = 1.2 + i * 1.5
    add_rounded_rect(slide, 0.8, y, 11.7, 1.35)
    add_accent_bar(slide, 0.8, y, width=0.06, height=1.35, color=color)
    add_textbox(slide, 1.2, y + 0.05, 5.5, 0.35, name, font_size=18, color=color, bold=True)
    add_textbox(slide, 7.0, y + 0.05, 5.2, 0.35, domain, font_size=14, color=GRAY_MED, bold=True, alignment=PP_ALIGN.RIGHT)
    add_textbox(slide, 1.2, y + 0.45, 11.0, 0.8, desc, font_size=14, color=GRAY_LIGHT)


# ═══════════════════════════════════════════════════════════════
#  SLIDE 8: NOVELTY — EYE FATIGUE CNN
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_dark_background(slide)
add_accent_bar(slide, 0, 0, width=13.333, height=0.06, color=ACCENT_AMBER)

add_textbox(slide, 0.8, 0.4, 10, 0.6, "06  Novelty - Live Webcam Eye Fatigue Scanner", font_size=32, color=WHITE, bold=True)

add_rounded_rect(slide, 0.8, 1.3, 5.8, 3.0)
add_accent_bar(slide, 0.8, 1.3, width=0.06, height=3.0, color=ACCENT_AMBER)
tf = add_textbox(slide, 1.2, 1.4, 5.2, 0.5, "CNN Pipeline", font_size=22, color=ACCENT_AMBER, bold=True)
add_paragraph(tf, "   Dataset: MRL Eye Dataset (Fusek, 2018)", font_size=15, color=GRAY_LIGHT)
add_paragraph(tf, "   Total Images: 84,898 eye samples", font_size=15, color=GRAY_LIGHT)
add_paragraph(tf, "   Classes: Alert (Open) vs Fatigued (Closed/Droopy)", font_size=15, color=GRAY_LIGHT)
add_paragraph(tf, "   Model 1: Custom CNN - Test Acc: 93.39%, AUC: 0.96", font_size=15, color=ACCENT_GREEN, bold=True)
add_paragraph(tf, "   Model 2: ResNet50 Transfer Learning - AUC: 0.89", font_size=15, color=GRAY_LIGHT)
add_paragraph(tf, "   Live webcam integration via st.camera_input", font_size=15, color=GRAY_LIGHT)

add_rounded_rect(slide, 7.0, 1.3, 5.5, 3.0)
add_accent_bar(slide, 7.0, 1.3, width=0.06, height=3.0, color=ACCENT_BLUE)
tf = add_textbox(slide, 7.4, 1.4, 4.9, 0.5, "OpenCV Feature Extraction", font_size=20, color=ACCENT_BLUE, bold=True)
add_paragraph(tf, "4 Ocular Biomarkers Extracted:", font_size=16, color=WHITE, bold=True)
add_paragraph(tf, "  1. Eye Openness Ratio (Haar Cascade detection)", font_size=15, color=GRAY_LIGHT)
add_paragraph(tf, "  2. Sclera Redness Index (color channel analysis)", font_size=15, color=GRAY_LIGHT)
add_paragraph(tf, "  3. Pixel Brightness (mean grayscale intensity)", font_size=15, color=GRAY_LIGHT)
add_paragraph(tf, "  4. Image Contrast (standard deviation of intensities)", font_size=15, color=GRAY_LIGHT)

eye_charts = [
    ('eye1_training_history.png', 'CNN Training History'),
    ('eye2_confusion_matrix.png', 'CNN Confusion Matrix'),
    ('resnet_eye_training_history.png', 'ResNet Training History'),
]

for i, (fname, title) in enumerate(eye_charts):
    x = 0.8 + i * 4.2
    path = os.path.join(RESULTS_DIR, fname)
    if os.path.exists(path):
        slide.shapes.add_picture(path, Inches(x), Inches(4.7), width=Inches(3.8), height=Inches(2.5))
    add_textbox(slide, x, 4.45, 3.8, 0.3, title, font_size=12, color=GRAY_MED, alignment=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════
#  SLIDE 9: RESULTS & EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_dark_background(slide)
add_accent_bar(slide, 0, 0, width=13.333, height=0.06, color=ACCENT_AMBER)

add_textbox(slide, 0.8, 0.4, 10, 0.6, "07  Results & Evaluation Metrics", font_size=32, color=WHITE, bold=True)

add_rounded_rect(slide, 0.8, 1.2, 11.7, 5.8)

rows = [
    ("Model / Component", "Metric", "Value", "Status", True, ACCENT_BLUE),
    ("Stacking Ensemble (Tabular)", "5-Fold CV Accuracy", "83.00% (+/-4.4%)", "Best Classifier", False, WHITE),
    ("Cognitive ANN (Keras)", "Attention R2", "0.43", "Moderate", False, WHITE),
    ("Cognitive ANN (Keras)", "Memory R2", "0.18", "Fair (noise expected)", False, WHITE),
    ("CNN Eye Fatigue", "Test Accuracy", "93.39%", "Excellent", False, WHITE),
    ("CNN Eye Fatigue", "AUC-ROC", "0.96", "Excellent", False, WHITE),
    ("ResNet50 Transfer", "AUC-ROC", "0.89", "Good", False, WHITE),
    ("Late Fusion (RF only)", "Mean CV Acc", "76.3%", "Baseline", False, GRAY_MED),
    ("Late Fusion (Realistic)", "Mean CV Acc", "79.3%", "+3.0% Improvement", False, ACCENT_GREEN),
    ("K-Means Clustering", "Gamer Archetypes", "4 Clusters", "Validated", False, WHITE),
]

for i, (col1, col2, col3, col4, is_header, color) in enumerate(rows):
    y = 1.3 + i * 0.55
    if is_header:
        bg_color = RGBColor(0x1A, 0x25, 0x3A)
    elif i % 2 == 0:
        bg_color = RGBColor(0x16, 0x20, 0x33)
    else:
        bg_color = BG_CARD

    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(y), Inches(11.5), Inches(0.5))
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.fill.background()

    fs = 14 if not is_header else 15
    add_textbox(slide, 1.0, y + 0.05, 3.2, 0.4, col1, font_size=fs, color=color, bold=is_header)
    add_textbox(slide, 4.2, y + 0.05, 2.5, 0.4, col2, font_size=fs, color=color, bold=is_header)
    add_textbox(slide, 6.8, y + 0.05, 2.5, 0.4, col3, font_size=fs, color=color, bold=is_header)
    add_textbox(slide, 9.5, y + 0.05, 2.5, 0.4, col4, font_size=fs, color=color, bold=is_header)


# ═══════════════════════════════════════════════════════════════
#  SLIDE 10: WEB APPLICATION DEMO
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_dark_background(slide)
add_accent_bar(slide, 0, 0, width=13.333, height=0.06, color=ACCENT_PURPLE)

add_textbox(slide, 0.8, 0.4, 10, 0.6, "08  Web Application - Streamlit v5 Demo", font_size=32, color=WHITE, bold=True)

pages = [
    ("Home & Multimodal Prediction", "Student inputs -> RF + ANN + CNN fused -> Risk Level + Health Score + Gamer Archetype", ACCENT_BLUE),
    ("Cognitive & Biometric Tests", "4 clinical neuropsych tests: Stroop, Grid Memory, Flanker, Symbol-Digit Coding", ACCENT_GREEN),
    ("Eye Fatigue Analysis", "Live webcam capture via st.camera_input -> OpenCV ocular biomarker extraction", ACCENT_AMBER),
    ("What-If Lab & Radar Profiling", "Interactive sensitivity simulator + 6-axis radar chart vs population benchmarks", ACCENT_PURPLE),
    ("Detox & Routine Planner", "20-20-20 Eye Strain Pomodoro timer + dynamic daily gaming budget calculator", ACCENT_RED),
    ("Model Performance & Insights", "14-model comparison, RF evaluation charts, Multimodal Fusion JSON benchmark", ACCENT_BLUE),
    ("PDF Assessment Report", "ReportLab-generated clinical report with dual sign-off block for KDU verification", ACCENT_AMBER),
]

for i, (page, desc, color) in enumerate(pages):
    y = 1.2 + i * 0.85
    add_rounded_rect(slide, 0.8, y, 11.7, 0.75)
    add_accent_bar(slide, 0.8, y, width=0.06, height=0.75, color=color)
    add_textbox(slide, 1.2, y + 0.05, 4.0, 0.35, page, font_size=16, color=color, bold=True)
    add_textbox(slide, 5.5, y + 0.05, 6.8, 0.65, desc, font_size=13, color=GRAY_LIGHT)

add_textbox(slide, 0.8, 7.2, 11.7, 0.3, "Live Demo: streamlit run app_v5.py  ->  localhost:8501", font_size=14, color=ACCENT_GREEN, bold=True, alignment=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════
#  SLIDE 11: CONCLUSION & FUTURE WORK
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_dark_background(slide)
add_accent_bar(slide, 0, 0, width=13.333, height=0.06, color=ACCENT_BLUE)

add_textbox(slide, 0.8, 0.4, 8, 0.6, "09  Conclusion & Future Work", font_size=32, color=WHITE, bold=True)

add_rounded_rect(slide, 0.8, 1.3, 5.8, 5.5)
add_accent_bar(slide, 0.8, 1.3, width=0.06, height=5.5, color=ACCENT_GREEN)
tf = add_textbox(slide, 1.2, 1.4, 5.2, 0.5, "Key Contributions", font_size=22, color=ACCENT_GREEN, bold=True)
add_paragraph(tf, "  First multimodal fusion system combining survey, cognitive, and visual data for gaming addiction prediction", font_size=14, color=GRAY_LIGHT)
add_paragraph(tf, "  Late Fusion achieves +3% accuracy improvement over single-model baseline", font_size=14, color=GRAY_LIGHT)
add_paragraph(tf, "  CNN Eye Fatigue detector achieves 93.39% accuracy (AUC 0.96)", font_size=14, color=GRAY_LIGHT)
add_paragraph(tf, "  Clinically validated neuropsychological test battery (Stroop, Flanker, DSST, Grid Memory)", font_size=14, color=GRAY_LIGHT)
add_paragraph(tf, "  Full-stack Streamlit deployment with interactive What-If simulator and PDF report generation", font_size=14, color=GRAY_LIGHT)
add_paragraph(tf, "  K-Means clustering identifies 4 distinct gamer archetypes", font_size=14, color=GRAY_LIGHT)

add_rounded_rect(slide, 7.0, 1.3, 5.5, 5.5)
add_accent_bar(slide, 7.0, 1.3, width=0.06, height=5.5, color=ACCENT_AMBER)
tf = add_textbox(slide, 7.4, 1.4, 4.9, 0.5, "Future Work", font_size=22, color=ACCENT_AMBER, bold=True)
add_paragraph(tf, "  Collect unified dataset combining eye images + addiction labels from same cohort", font_size=14, color=GRAY_LIGHT)
add_paragraph(tf, "  Expand dataset to 1000+ students from multiple Sri Lankan universities", font_size=14, color=GRAY_LIGHT)
add_paragraph(tf, "  Integrate EEG headband data as a 4th modality for neural attention measurement", font_size=14, color=GRAY_LIGHT)
add_paragraph(tf, "  Develop a mobile companion app for continuous gaming behavior monitoring", font_size=14, color=GRAY_LIGHT)
add_paragraph(tf, "  Deploy to cloud (AWS/GCP) for multi-university access", font_size=14, color=GRAY_LIGHT)
add_paragraph(tf, "  Explore attention-based transformer fusion as a next-gen alternative to log-linear pooling", font_size=14, color=GRAY_LIGHT)


# ═══════════════════════════════════════════════════════════════
#  SLIDE 12: THANK YOU
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_dark_background(slide)
add_accent_bar(slide, 0, 0, width=13.333, height=0.06, color=ACCENT_BLUE)

add_textbox(slide, 1.0, 2.0, 11.3, 1.0,
    "Thank You",
    font_size=48, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

add_textbox(slide, 1.0, 3.2, 11.3, 0.6,
    "Questions & Discussion",
    font_size=28, color=ACCENT_BLUE, bold=False, alignment=PP_ALIGN.CENTER)

tf = add_textbox(slide, 1.0, 4.5, 11.3, 0.4,
    "P R M K Herath  |  D/BCS/23/0009",
    font_size=20, color=GRAY_LIGHT, alignment=PP_ALIGN.CENTER)
add_paragraph(tf, "BSc (Hons) Computer Science",
    font_size=16, color=GRAY_MED, alignment=PP_ALIGN.CENTER)
add_paragraph(tf, "General Sir John Kotelawala Defence University",
    font_size=16, color=GRAY_MED, alignment=PP_ALIGN.CENTER)


# ── SAVE ─────────────────────────────────────────────────────
try:
    prs.save(OUTPUT_PATH)
    print(f"\nPresentation saved to: {OUTPUT_PATH}")
except PermissionError:
    ALT_PATH = os.path.join(BASE_DIR, 'Progress_Review_II_Presentation_v2.pptx')
    prs.save(ALT_PATH)
    print(f"\nPresentation saved to: {ALT_PATH}")
print(f"   Total slides: {len(prs.slides)}")
