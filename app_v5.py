# ============================================================
#  FINAL WEB APP v5
#  Behaviour model trained on REAL data: 322 Sri Lankan students with the
#  IGDS9-SF (Manchanayake, 2021; notebooks/07_train_real_igd_model.py)
#  Guided patient workflow:
#    1. Student Details      - gaming, sleep and wellbeing questionnaire + mood (PHQ-4) and loneliness (UCLA-3)
#    2. Cognitive Tests      - 2-Back, Task-Switching, Mental Rotation, Go/No-Go
#    3. Eye Check            - face/eye localisation + eye-state CNN (optional)
#    4. Risk Result          - confidence-weighted late fusion + PDF report
#    5. Solution & Plan      - personal habit changes + CBT-based reduction programme
#  Researcher tools: model performance, continuous eye monitor
#  Student: P R M K Herath | D/BCS/23/0009
#  Degree: BSc (Hons) Computer Science | KDU
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib, os, cv2, time, random
from PIL import Image
import io

# ReportLab imports for PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Import Late Fusion Engine module
import fusion_engine as fe
# Eye localisation + eye-state CNN for the Eye Check
import eye_analysis
# Shared feature encoding for the real-data IGD model
import igd_features as igdf
# Research-based cognitive battery (2-Back, Task-Switching, Mental Rotation, Go/No-Go)
import research_cognitive_tests as rct
# Continuous eye-session logs (researcher tool)
import eye_sessions
# Consented study: webcam eye measures vs gaming hours / IGDS9-SF (researcher tool)
import eye_gaming_study
# Evidence-based self-help programme for reducing problem gaming (CBT components)
import reduction_program

st.set_page_config(
    page_title="Gaming Behavior & Cognitive Predictor",
    page_icon="🎮",
    layout="wide"
)

BASE = os.path.dirname(os.path.abspath(__file__))

# Phone layout: smaller page titles so content starts on the first screen
st.markdown("""<style>
@media (max-width: 640px) {
  h1 { font-size: 1.7rem !important; line-height: 1.25 !important; }
  h2 { font-size: 1.35rem !important; }
  h3 { font-size: 1.15rem !important; }
}
</style>""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    # Behaviour model and behaviour clusters are trained on REAL data
    # (notebooks/07_train_real_igd_model.py, Manchanayake 2021, n = 322)
    igd = joblib.load(os.path.join(BASE, 'models', 'igd_risk_model.pkl'))
    clusters = joblib.load(os.path.join(BASE, 'models', 'igd_clusters.pkl'))
    # Eye check: YuNet eye localisation + eye-state CNN trained on the MRL Eye Dataset
    # (notebooks/08_train_eye_state_model.py, evaluated on people unseen in training)
    analyzer = eye_analysis.EyeAnalyzer()
    return igd['model'], igd['meta'], clusters, analyzer


try:
    igd_model, igd_meta, igd_clusters, eye_analyzer = load_all_models()
except FileNotFoundError as e:
    st.error(f"Model file missing ({e.filename}). Run `python notebooks/07_train_real_igd_model.py` first.")
    st.stop()

model_name = igd_meta['model_name']


def predict_igds(X):
    """Predicted IGDS9-SF score, kept inside the scale's 9-45 range."""
    return float(np.clip(igd_model.predict(X)[0], 9, 45))


# Answers that are compared against the training data's typical (median) answer
SENSITIVITY_GROUPS = {
    'Weekday gaming': ['weekday_hours'], 'Weekend gaming': ['weekend_hours'], 'Sleep hours': ['sleep_hours'],
    'Sleep quality': ['sleep_quality'], 'Age started gaming': ['start_age'], 'Gender': ['is_male'],
    'Game mode': ['mode_online_multi', 'mode_online_single'],
    'Genre': ['genre_br', 'genre_shooter', 'genre_action', 'genre_sports'],
    'Device': ['device_mobile', 'device_pc'], 'Self-esteem': ['self_esteem'], 'Few friends': ['few_friends'],
    'Sports': ['sports_level'],
}


def answer_sensitivity(X_user, pred, top=8):
    medians = igd_meta['feature_medians']
    deltas = {}
    for label, cols in SENSITIVITY_GROUPS.items():
        X_cf = X_user.copy()
        for c in cols:
            X_cf[c] = medians[c]
        if 'weekday_hours' in cols or 'weekend_hours' in cols:
            X_cf['weekly_hours'] = 5 * X_cf['weekday_hours'] + 2 * X_cf['weekend_hours']
        deltas[label] = pred - predict_igds(X_cf)
    ranked = sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top]
    return dict(reversed(ranked))


def habit_changes(assessment):
    """Changeable habits ranked by how much the real-data model's predicted IGDS9-SF score would drop."""
    base = pd.DataFrame([assessment['features']])[igdf.FEATURES]
    f = base.iloc[0]
    base_score = predict_igds(base)
    candidates = []
    if f['weekend_hours'] > 1:
        new = max(0.5, round(f['weekend_hours']) / 2)
        candidates.append((f"Halve weekend gaming ({f['weekend_hours']:g} → {new:g} h/day)", {'weekend_hours': new}))
    if f['weekday_hours'] > 1:
        new = max(0.5, round(f['weekday_hours']) / 2)
        candidates.append((f"Halve weekday gaming ({f['weekday_hours']:g} → {new:g} h/day)", {'weekday_hours': new}))
    if f['sleep_hours'] < 7:
        candidates.append((f"Sleep 7.5 h a night (now {f['sleep_hours']:g} h)", {'sleep_hours': 7.5}))
    if f['sleep_quality'] < 8:
        new = min(10, f['sleep_quality'] + 2)
        candidates.append((f"Better sleep quality, e.g. no gaming in the hour before bed ({f['sleep_quality']:g} → {new:g}/10)",
                           {'sleep_quality': new}))
    if f['sports_level'] < 2:
        candidates.append(("Take part in sport regularly (moderate involvement)", {'sports_level': 2}))
    if len(candidates) > 1:
        combined = {k: v for _, change in candidates for k, v in change.items()}
        candidates.append(("⭐ All of the above together", combined))

    rows = []
    for label, change in candidates:
        X = base.copy()
        for k, v in change.items():
            X[k] = v
        X['weekly_hours'] = 5 * X['weekday_hours'] + 2 * X['weekend_hours']
        score = predict_igds(X)
        if score - base_score > -0.2 and not label.startswith("⭐"):
            continue                                   # no meaningful improvement for this student
        proba = igdf.tier_probabilities(score, igd_meta['sigma'])
        rows.append({'Change': label, 'Predicted IGDS9-SF': round(score, 1), 'Change in score': round(score - base_score, 1),
                     'Risk level (answers only)': igdf.RISK_LABELS[int(np.argmax(proba))],
                     'Chance of High risk': f"{proba[2] * 100:.0f}%"})
    rows.sort(key=lambda r: (not r['Change'].startswith("⭐"), r['Change in score']))
    return rows

# ── PDF Report (ReportLab) ────────────────────────────────────────────────────
def generate_pdf_report_v5(results_dict):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=36, leftMargin=36,
                            topMargin=36, bottomMargin=36)
    story = []
    styles = getSampleStyleSheet()

    primary_color = colors.HexColor("#1A365D")
    secondary_color = colors.HexColor("#0D9488")
    text_color = colors.HexColor("#2D3748")

    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'],
        fontName='Helvetica-Bold', fontSize=18, leading=22,
        textColor=primary_color, spaceAfter=8
    )
    h1_style = ParagraphStyle(
        'DocH1', parent=styles['Heading2'],
        fontName='Helvetica-Bold', fontSize=13, leading=16,
        textColor=secondary_color, spaceBefore=10, spaceAfter=6,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'DocBody', parent=styles['Normal'],
        fontName='Helvetica', fontSize=9.5, leading=13,
        textColor=text_color, spaceAfter=4
    )
    bold_body_style = ParagraphStyle('DocBodyBold', parent=body_style, fontName='Helvetica-Bold')
    table_style = TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('PADDING', (0, 0), (-1, -1), 5),
    ])

    story.append(Paragraph("GAMING BEHAVIOUR & COGNITIVE ASSESSMENT REPORT", title_style))
    story.append(Paragraph("<b>Faculty of Computing | General Sir John Kotelawala Defence University (KDU)</b>", body_style))
    story.append(Paragraph("<b>Research prototype by:</b> P R M K Herath | D/BCS/23/0009 | BSc (Hons) Computer Science", body_style))
    story.append(Paragraph(f"<b>Assessment time:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    story.append(Paragraph("<i>This is a screening estimate from a research prototype, not a medical diagnosis.</i>", body_style))
    story.append(Spacer(1, 8))

    # 1. Profile
    story.append(Paragraph("1. Student Profile", h1_style))
    info_data = [
        [Paragraph("<b>Age:</b>", body_style), Paragraph(str(results_dict['age']), body_style),
         Paragraph("<b>Gender:</b>", body_style), Paragraph(results_dict['gender'], body_style)],
        [Paragraph("<b>Weekly gaming:</b>", body_style), Paragraph(f"{results_dict['hours_wk']} h/week", body_style),
         Paragraph("<b>Sleep:</b>", body_style), Paragraph(f"{results_dict['sleep_hours']} h/night", body_style)],
        [Paragraph("<b>Main mode:</b>", body_style), Paragraph(results_dict['mode'], body_style),
         Paragraph("<b>Main genre:</b>", body_style), Paragraph(results_dict['genre'], body_style)],
    ]
    t_info = Table(info_data, colWidths=[110, 140, 110, 140])
    t_info.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
    story.append(t_info)
    story.append(Spacer(1, 8))

    # 2. Fusion result
    story.append(Paragraph("2. Confidence-Weighted Multimodal Assessment", h1_style))
    fusion_data = [
        [Paragraph("<b>Component</b>", bold_body_style), Paragraph("<b>Result</b>", bold_body_style),
         Paragraph("<b>Confidence / Weight</b>", bold_body_style)],
        [Paragraph(f"Behaviour model ({results_dict['model_name']}, trained on real IGDS9-SF data)", body_style),
         Paragraph(f"<b>{results_dict['rf_pred']}</b>", body_style),
         Paragraph(f"Conf: {results_dict['rf_conf']:.2f} (Weight: {results_dict['rf_w']:.2f})", body_style)],
        [Paragraph("Cognitive vote (Go/No-Go response inhibition, literature prior)", body_style),
         Paragraph(f"<b>{results_dict['cog_pred']}</b>", body_style),
         Paragraph(f"Conf: {results_dict['cog_conf']:.2f} (Weight: {results_dict['cog_w']:.2f})", body_style)],
        [Paragraph("Eye-fatigue vote (eye-state CNN, literature prior)", body_style),
         Paragraph(f"<b>{results_dict['eye_pred']}</b>", body_style),
         Paragraph(f"Conf: {results_dict['eye_conf']:.2f} (Weight: {results_dict['eye_w']:.2f})", body_style)],
        [Paragraph("Wellbeing vote (mood PHQ-4 + loneliness UCLA-3, literature prior)", body_style),
         Paragraph(f"<b>{results_dict['wb_pred']}</b>", body_style),
         Paragraph(f"Conf: {results_dict['wb_conf']:.2f} (Weight: {results_dict['wb_w']:.2f})", body_style)],
        [Paragraph("<b>Fused result</b>", bold_body_style), Paragraph(f"<b>{results_dict['risk_level']}</b>", bold_body_style),
         Paragraph("Log-linear opinion pooling", bold_body_style)],
        [Paragraph("Predicted IGDS9-SF score (9-45)", body_style),
         Paragraph(f"<b>{results_dict['pred_igds']:.1f}</b>", bold_body_style),
         Paragraph(f"Typical error +/- {results_dict['mae']:.1f} points", body_style)],
    ]
    t_fusion = Table(fusion_data, colWidths=[200, 130, 170])
    t_fusion.setStyle(table_style)
    t_fusion.setStyle(TableStyle([('BACKGROUND', (0, 5), (-1, 5), colors.HexColor("#FEF3C7"))]))
    story.append(t_fusion)
    story.append(Spacer(1, 8))

    # 3. Behaviour profile
    story.append(Paragraph("3. Behaviour Profile (K-means clusters of the real dataset)", h1_style))
    story.append(Paragraph(f"<b>Profile:</b> {results_dict['archetype']}", bold_body_style))
    story.append(Paragraph(results_dict['archetype_desc'], body_style))
    story.append(Spacer(1, 8))

    # 4. Measured tests
    story.append(Paragraph("4. Measured Cognitive Tests, Eye Fatigue & Wellbeing", h1_style))
    bio_data = [
        [Paragraph("<b>Measure</b>", bold_body_style), Paragraph("<b>Measured value</b>", bold_body_style),
         Paragraph("<b>Interpretation</b>", bold_body_style)],
        [Paragraph("Go reaction time (ms)", body_style),
         Paragraph(f"{results_dict['rt_measured']} ms" if results_dict['rt_measured'] else "Not measured", body_style),
         Paragraph(("Fast" if results_dict['rt_measured'] < 350 else "Typical" if results_dict['rt_measured'] < 500 else "Slow")
                   if results_dict['rt_measured'] else "-", body_style)],
        [Paragraph("Working-memory score", body_style),
         Paragraph(f"{results_dict['mem_measured']}%" if results_dict['mem_measured'] is not None else "Not measured", body_style),
         Paragraph("-", body_style)],
        [Paragraph("Attention / executive control score", body_style),
         Paragraph(f"{results_dict['att_measured']}%" if results_dict['att_measured'] is not None else "Not measured", body_style),
         Paragraph("-", body_style)],
        [Paragraph("Tiredness right now (eye photo)", body_style),
         Paragraph(f"{results_dict['eye_fatigue']}/100" if results_dict['eye_fatigue'] != 'N/A' else "Not captured", body_style),
         Paragraph(str(results_dict['eye_level']), body_style)],
    ]
    ext = results_dict.get('battery', {})
    if 'nback' in ext:
        r = ext['nback']
        bio_data.append([Paragraph("2-Back Working Memory", body_style), Paragraph(f"{r['accuracy']}% (d' {r['d_prime']:.2f})", body_style), Paragraph("Good updating" if r['d_prime'] >= 1.5 else "Weak updating", body_style)])
    if 'go_nogo' in ext:
        r = ext['go_nogo']
        bio_data.append([Paragraph("Go/No-Go Inhibition", body_style), Paragraph(f"{r['commission_rate']}% commission errors", body_style), Paragraph("Impulsive responding" if r['commission_rate'] >= 40 else "Controlled", body_style)])
    if 'task_switching' in ext:
        r = ext['task_switching']
        bio_data.append([Paragraph("Task-Switching Cost", body_style), Paragraph(f"{r['switch_cost']} ms ({r['accuracy']}% acc)", body_style), Paragraph("Flexible" if r['switch_cost'] < 300 else "Reduced flexibility", body_style)])
    if 'mental_rotation' in ext:
        r = ext['mental_rotation']
        bio_data.append([Paragraph("Mental Rotation", body_style), Paragraph(f"{r['accuracy']}% ({r['mean_rt']} ms)", body_style), Paragraph(f"Rate {r['slope_ms_per_deg']} ms/deg", body_style)])
    wb = results_dict.get('wellbeing')
    if wb:
        bio_data.append([Paragraph("Low mood (PHQ-2)", body_style), Paragraph(f"{wb['phq2']} / 6", body_style), Paragraph(wb['phq2_text'], body_style)])
        bio_data.append([Paragraph("Anxiety (GAD-2)", body_style), Paragraph(f"{wb['gad2']} / 6", body_style), Paragraph(wb['gad2_text'], body_style)])
        bio_data.append([Paragraph("Loneliness (UCLA-3)", body_style), Paragraph(f"{wb['lonely']} / 9", body_style), Paragraph(wb['lonely_text'], body_style)])
    t_bio = Table(bio_data, colWidths=[170, 160, 170])
    t_bio.setStyle(table_style)
    story.append(t_bio)
    story.append(Spacer(1, 8))

    # 5. Next steps (stepped care)
    story.append(Paragraph("5. Recommended Next Steps", h1_style))
    steps = {
        0: ["Keep logging your gaming for a week to stay aware of late-night and weekend patterns.",
            "Set one if-then plan for your riskiest moment (e.g., 'If it is 30 minutes before bed, then I stop playing')."],
        1: ["Work through the app's Gaming Reduction Program (CBT-based self-help) over the next 2-4 weeks.",
            "Set a weekly gaming target about 25% below your current level, with if-then plans for your triggers.",
            "Protect sleep: no gaming in the hour before bed."],
        2: ["Talk to a professional: Sri Lanka National Mental Health Helpline 1926 (24/7, free) or your university counselling service.",
            "Use the Gaming Reduction Program alongside professional support.",
            "Protect sleep: no gaming in the hour before bed."],
    }[results_dict['risk_idx']]
    for step in steps:
        story.append(Paragraph(f"- {step}", body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ── Sidebar Navigation: guided patient workflow + researcher tools ────────────
STEPS = ["1️⃣ Student Details", "2️⃣ Cognitive Tests", "3️⃣ Eye Check", "4️⃣ Risk Result", "5️⃣ Solution & Plan"]
PATIENT_PAGES = STEPS + ["📋 Student History"]
RESEARCH_PAGES = ["📊 Model Performance", "🎮 Eye Session Monitor", "👁️ Eye–Gaming Study"]
NAV_MODES = {"🩺 Patient Assessment": PATIENT_PAGES, "🔬 Researcher Tools": RESEARCH_PAGES}
# Researcher tools are for the researcher's own laptop only: hidden on the public online version
if not eye_gaming_study.is_local():
    NAV_MODES = {"🩺 Patient Assessment": PATIENT_PAGES}
PATIENT_KEYS = ['patient_inputs', 'cognitive_results', 'mem_score', 'att_score', 'eye_fatigue_score',
                'eye_fatigue_level', 'eye_fatigue_source', 'eye_skipped', 'last_assessment',
                'assessment_sig'] + rct.RESET_KEYS
DEFAULT_INPUTS = dict(age=21, gender="Male", start_age=10, sports="Minor", weekday_hours=1.5, weekend_hours=2.5,
                      mode=igdf.MODES[0], genre=igdf.GENRES[0], device=igdf.DEVICES[0], sleep_hours=7.0,
                      sleep_quality=7, self_esteem=4, few_friends=2)
SPORTS_LEVELS = ["None", "Minor", "Moderate", "Major"]
POSSIBLE_HIGH_PROB = 0.30          # fused P(High) at or above this shows a "possible High risk" warning
# Wellbeing check: two short validated scales (literature prior in the fusion, see fusion_engine.py)
# PHQ-4 (Kroenke, Spitzer, Williams & Löwe, 2009): items 1-2 = GAD-2 (anxiety), items 3-4 = PHQ-2 (low mood)
PHQ4_ITEMS = ["Feeling nervous, anxious or on edge", "Not being able to stop or control worrying",
              "Little interest or pleasure in doing things", "Feeling down, depressed or hopeless"]
PHQ4_OPTIONS = ["Not at all", "Several days", "More than half the days", "Nearly every day"]   # 0-3
# 3-item UCLA Loneliness Scale (Hughes, Waite, Hawkley & Cacioppo, 2004)
UCLA3_ITEMS = ["How often do you feel that you lack companionship?", "How often do you feel left out?",
               "How often do you feel isolated from others?"]
UCLA3_OPTIONS = ["Hardly ever", "Some of the time", "Often"]                                   # 1-3
# Eye Check verdicts for levels 1-4 (eye_analysis.LEVEL_NAMES): icon, message box, message, what to do
EYE_VERDICTS = [
    ("🟢", st.success,
     "Your eyes look awake and rested right now.",
     "Keep it that way: take regular short breaks from the screen while you play."),
    ("🟡", st.warning,
     "Your eyes look a little tired. This is common after long gaming or screen sessions.",
     "Take a 10–15 minute break from the screen now. Blink often and look out of a window or at something far away."),
    ("🟠", st.warning,
     "Your eyes look tired — a sign you have been gaming or on screens for too long without enough rest.",
     "Stop gaming for now and rest your eyes for at least 20–30 minutes. Drink some water, and don't play again "
     "close to bedtime tonight."),
    ("🔴", st.error,
     "Your eyes look very tired, almost closing. You need sleep, not more gaming.",
     "Stop playing and get some sleep. Gaming when you are this tired hurts your sleep, your focus and your studies."),
]
ss = st.session_state


def wellbeing_scores(inputs):
    """Scale totals with the published cut-offs: PHQ-2 / GAD-2 >= 3 is a positive screen (Kroenke et al., 2009);
    UCLA-3 >= 6 is the usual cut-off for loneliness."""
    phq = inputs['phq4']
    gad2, phq2, lonely = phq[0] + phq[1], phq[2] + phq[3], sum(inputs['ucla3'])
    return {'gad2': gad2, 'phq2': phq2, 'phq4': gad2 + phq2, 'lonely': lonely,
            'gad2_text': "Above screening cut-off" if gad2 >= 3 else "Below cut-off",
            'phq2_text': "Above screening cut-off" if phq2 >= 3 else "Below cut-off",
            'lonely_text': "Often lonely" if lonely >= 6 else "Not lonely"}


def step_done(step):
    return {STEPS[0]: 'patient_inputs' in ss,
            STEPS[1]: 'cognitive_results' in ss,
            STEPS[2]: 'eye_fatigue_score' in ss or ss.get('eye_skipped', False),
            STEPS[3]: 'last_assessment' in ss}.get(step, False)


def go_to(target):
    # Used as a button callback, so it runs before the sidebar widgets are drawn
    for nav_mode, pages in NAV_MODES.items():
        if target in pages:
            ss['nav_mode'] = nav_mode
    ss['nav_page'] = target


def new_patient():
    for k in PATIENT_KEYS + [k for k in list(ss.keys()) if k.startswith('f_')]:   # incl. Step 1 form fields
        ss.pop(k, None)
    go_to(STEPS[0])


def next_button(target, label=None):
    st.button(label or f"Next: {target} ➡️", type="primary", use_container_width=True, on_click=go_to, args=(target,))


st.sidebar.title("🎮 Gaming Risk Assessment")
st.sidebar.markdown("**D/BCS/23/0009 | P R M K Herath**")
st.sidebar.caption("General Sir John Kotelawala Defence University")
st.sidebar.divider()

if len(NAV_MODES) > 1:
    nav_mode = st.sidebar.radio("Mode", list(NAV_MODES), key="nav_mode", label_visibility="collapsed")
else:                                            # online: patient assessment only, no mode switch
    nav_mode = ss['nav_mode'] = next(iter(NAV_MODES))
options = NAV_MODES[nav_mode]
if ss.get('nav_page') not in options:
    ss['nav_page'] = options[0]
page = st.sidebar.radio("Steps" if nav_mode == "🩺 Patient Assessment" else "Tools", options, key="nav_page",
                        format_func=lambda p: f"{p}  ✅" if step_done(p) else p)
if nav_mode == "🩺 Patient Assessment":
    n_done = sum(step_done(s) for s in STEPS[:4])
    st.sidebar.progress(n_done / 4, text=f"{n_done} of 4 assessment steps done")
    st.sidebar.button("🆕 New patient", use_container_width=True, on_click=new_patient,
                      help="Clear all answers and results to start with the next student.")

# ════════════════════════════════════════════════════════════
#  STEP 1: STUDENT DETAILS
# ════════════════════════════════════════════════════════════
if page == STEPS[0]:
    st.title("1️⃣ Student Details")
    st.markdown("Answer a few questions about your gaming, sleep and wellbeing. Next come four short cognitive tests "
                "and an eye check; then you get your risk result and a personal plan.")
    st.caption("Screening tool from a research prototype — not a medical diagnosis.")

    def save_details():
        phq = [ss.get(f'f_phq_{i}') for i in range(len(PHQ4_ITEMS))]
        ucla = [ss.get(f'f_ucla_{i}') for i in range(len(UCLA3_ITEMS))]
        if None in phq + ucla:
            ss['f_missing'] = True               # shown below the form; stay on this step
            return
        ss['f_missing'] = False
        ss['patient_inputs'] = dict(
            age=ss['f_age'], gender=ss['f_gender'], start_age=min(ss['f_start_age'], ss['f_age']),
            sports=ss['f_sports'], sports_level=SPORTS_LEVELS.index(ss['f_sports']),
            weekday_hours=ss['f_weekday'], weekend_hours=ss['f_weekend'], mode=ss['f_mode'], genre=ss['f_genre'],
            device=ss['f_device'], sleep_hours=ss['f_sleep'], sleep_quality=ss['f_sleep_q'],
            self_esteem=ss['f_esteem'], few_friends=ss['f_friends'],
            phq4=[PHQ4_OPTIONS.index(a) for a in phq], ucla3=[UCLA3_OPTIONS.index(a) + 1 for a in ucla])
        ss.pop('last_assessment', None)          # answers changed -> result must be recalculated
        go_to(STEPS[1])

    p = {**DEFAULT_INPUTS, **ss.get('patient_inputs', {})}
    with st.form("details_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**👤 About you**")
            st.slider("Age", 16, 35, p['age'], key="f_age")
            st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(p['gender']), key="f_gender")
            st.slider("Age you started gaming regularly", 3, 35, p['start_age'], key="f_start_age")
            st.selectbox("Involvement in sports", SPORTS_LEVELS, index=SPORTS_LEVELS.index(p['sports']), key="f_sports")
        with col2:
            st.markdown("**🎮 Gaming habits**")
            st.slider("Gaming on a typical weekday (hours/day)", 0.0, 12.0, float(p['weekday_hours']), step=0.5, key="f_weekday")
            st.slider("Gaming on a typical weekend day (hours/day)", 0.0, 14.0, float(p['weekend_hours']), step=0.5, key="f_weekend")
            st.selectbox("Main way you play", igdf.MODES, index=igdf.MODES.index(p['mode']), key="f_mode")
            st.selectbox("Main game genre", igdf.GENRES, index=igdf.GENRES.index(p['genre']), key="f_genre")
            st.selectbox("Main device", igdf.DEVICES, index=igdf.DEVICES.index(p['device']), key="f_device")
        with col3:
            st.markdown("**🌙 Sleep & wellbeing**")
            st.slider("Sleep per night (hours)", 3.0, 10.0, float(p['sleep_hours']), step=0.5, key="f_sleep")
            st.slider("Sleep quality (0 = very poor, 10 = excellent)", 0, 10, int(p['sleep_quality']), key="f_sleep_q")
            st.slider("How would you rate your self-esteem? (1 = very low, 5 = very high)", 1, 5, int(p['self_esteem']), key="f_esteem")
            st.slider("\"I have few or no close friends\" (1 = not true, 5 = very true)", 1, 5, int(p['few_friends']), key="f_friends")
        st.divider()
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("**💬 Mood** — over the **last 2 weeks**, how often have you been bothered by:")
            for i, q in enumerate(PHQ4_ITEMS):
                prev = p['phq4'][i] if 'phq4' in p else None
                st.radio(q, PHQ4_OPTIONS, index=prev, key=f"f_phq_{i}")
        with m2:
            st.markdown("**🤝 Feeling connected** — think about your life in general:")
            for i, q in enumerate(UCLA3_ITEMS):
                prev = p['ucla3'][i] - 1 if 'ucla3' in p else None
                st.radio(q, UCLA3_OPTIONS, index=prev, key=f"f_ucla_{i}")
        st.form_submit_button(f"Save details — Next: {STEPS[1]} ➡️", type="primary", use_container_width=True,
                              on_click=save_details)
    if ss.get('f_missing'):
        st.error("Please answer all the mood and feeling-connected questions.")


# ════════════════════════════════════════════════════════════
#  STEP 2: COGNITIVE TESTS (all four required)
# ════════════════════════════════════════════════════════════
elif page == STEPS[1]:
    st.title("2️⃣ Cognitive Tests")
    st.markdown("Complete all four short tests (about 2–3 minutes). Each comes from peer-reviewed research on cognition "
                "in gamers and gaming disorder.")

    rct.render_battery()
    battery = rct.battery_results()
    st.divider()

    if len(battery) == 4:
        first_completion = 'cognitive_results' not in ss
        mem_pct, att_pct = rct.domain_scores(battery)
        ss['cognitive_results'] = {
            'reaction_time_ms':    battery['go_nogo']['mean_go_rt'],
            'memory_score_pct':    mem_pct,
            'attention_score_pct': att_pct,
            'battery':             battery,
        }
        ss['mem_score'], ss['att_score'] = mem_pct, att_pct
        if first_completion:
            ss.pop('last_assessment', None)
            st.rerun()                           # redraw so the sidebar ticks this step straight away
        st.success("✅ All four tests completed.")
        next_button(STEPS[2])
    else:
        st.caption(f"{len(battery)} of 4 tests completed.")

    if st.button("🔄 Redo the tests"):
        for k in ['cognitive_results', 'mem_score', 'att_score', 'last_assessment'] + rct.RESET_KEYS:
            ss.pop(k, None)
        st.rerun()


# ════════════════════════════════════════════════════════════
#  STEP 3: EYE CHECK (single scan; optional)
# ════════════════════════════════════════════════════════════
elif page == STEPS[2]:
    st.title("3️⃣ Eye Check")

    def skip_eye():
        ss['eye_skipped'] = True
        go_to(STEPS[3])

    st.markdown("Take a photo of your face with the webcam (or upload one). The app finds both eyes and checks how open "
                "they are, to show your **tiredness right now** — drooping or closing eyelids are the main visible sign "
                "of tiredness.")
    st.info("ℹ️ A photo shows how tired your eyes look **at this moment**. It cannot tell how much you game — someone "
            "who games every day can still look wide awake. Your gaming habits are judged from your answers in "
            f"**{STEPS[0]}**.")
    st.caption("Best results: face the camera in good light. The photo is analysed in memory and never saved.")

    # No default selection: the webcam only turns on when chosen, so it stays free for the session monitor
    input_source = st.radio("Choose Input Method:", ["📸 Live Webcam Camera", "📁 Upload Image File"], horizontal=True, index=None)

    pil_img = None
    if input_source == "📸 Live Webcam Camera":
        cam_photo = st.camera_input("Take a photo of your face")
        if cam_photo:
            pil_img = Image.open(cam_photo).convert('RGB')
    elif input_source == "📁 Upload Image File":
        uploaded_file = st.file_uploader("Upload a face or eye photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            pil_img = Image.open(uploaded_file).convert('RGB')

    if pil_img is not None:
        with st.spinner("🔍 Finding your eyes and checking how open they are..."):
            res = eye_analyzer.analyze(np.array(pil_img))
        col1, col2 = st.columns(2)
        col1.image(pil_img, caption="Your photo", use_container_width=True)
        col2.image(res['annotated_rgb'], use_container_width=True,
                   caption=f"Eyes found: {res['eyes_found']} ({res['method']}) — label = eye state and P(closed)")

        if res['fatigue_score'] is None and res['reason'] == 'too_small':
            st.warning("Your eyes are too small in this photo to judge reliably — move closer to the camera so your "
                       "face fills more of the picture, then try again (or skip this step).")
        elif res['fatigue_score'] is None:
            st.warning("No eyes could be found in this photo. Try again facing the camera in good light — "
                       "or skip this step.")
        else:
            fatigue = res['fatigue_score']
            n = eye_analysis.level_index(fatigue)
            level = eye_analysis.LEVEL_NAMES[n]
            icon, box, message, todo = EYE_VERDICTS[n]
            st.divider()
            box(f"### {icon} Tiredness right now: level {n + 1} of 4 — {level}\n\n{message}")
            st.progress(min(1.0, fatigue / 100), text=f"Tiredness right now: {fatigue:.0f} / 100")

            # Link to the student's own gaming time when Step 1 is done
            p_in = ss.get('patient_inputs')
            if p_in:
                weekly = 5 * p_in['weekday_hours'] + 2 * p_in['weekend_hours']
                if n >= 2 and weekly >= 21:
                    st.markdown(f"🎮 You told us you game about **{weekly:.0f} hours a week**. Together with your tired "
                                "eyes, this suggests you are gaming more than your body can recover from — cutting "
                                f"down is the first step (your plan is in **{STEPS[4]}**).")
                elif n >= 1 and weekly >= 14:
                    st.markdown(f"🎮 You game about **{weekly:.0f} hours a week**. Plan regular breaks during play "
                                "and stop at least an hour before bed.")
                elif n == 0 and weekly >= 21:
                    st.markdown(f"🎮 Your eyes look fine right now, but about **{weekly:.0f} hours of gaming a week** "
                                "is a lot — keep taking breaks so it stays that way.")
            st.markdown(f"**👉 What to do now:** {todo}")
            st.caption("Based on one photo taken now — it does not measure long-term eye strain from gaming. Tired eyes "
                       "can also come from lack of sleep, studying or poor lighting.")

            with st.expander("Technical details"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Tiredness right now", f"{fatigue}/100")
                c2.metric("Eyes open (model)", f"{100 - fatigue:.0f}%")
                c3.metric("Redness indicator", f"{res['redness']}" if res['redness'] is not None else "—",
                          help="Uncalibrated colour measure of the white of the eye; shown for information, not scored.")
                m = eye_analyzer.meta
                st.caption("Per-eye probability of a closed eye: " + " · ".join(f"{p:.2f}" for p in res['p_closed'])
                           + f". Eye-state CNN trained on the MRL Eye Dataset"
                             f"{' plus real-world CEW faces' if m.get('version', 1) >= 2 else ''}; on {m['test_people']} "
                             f"people it never saw in training it was {m['accuracy'] * 100:.1f}% accurate (AUC {m['auc']:.3f}).")

            first_scan = ss.get('eye_fatigue_score') != fatigue
            st.session_state['eye_fatigue_score'] = fatigue
            st.session_state['eye_fatigue_level'] = level
            st.session_state['eye_fatigue_source'] = 'single_scan'
            if first_scan:
                ss.pop('last_assessment', None)      # new eye score -> result must be recalculated
                st.rerun()                           # redraw so the sidebar ticks this step straight away
            st.success("✅ Tiredness score saved for your risk result.")

    st.divider()
    if 'eye_fatigue_score' in ss:
        ss.pop('eye_skipped', None)
        next_button(STEPS[3])
    else:
        st.button(f"Skip the eye check — Next: {STEPS[3]} ➡️", use_container_width=True, on_click=skip_eye,
                  help="No webcam? The result is then based on your answers and cognitive tests only.")


# ════════════════════════════════════════════════════════════
#  STEP 4: RISK RESULT (fusion of answers, cognitive tests and eye check)
# ════════════════════════════════════════════════════════════
elif page == STEPS[3]:
    st.title("4️⃣ Risk Result")
    missing = [s for s in STEPS[:3] if not step_done(s)]
    if missing:
        st.warning("Please finish these steps first to get your result:")
        for s in missing:
            st.button(f"Go to {s}", on_click=go_to, args=(s,), key=f"goto_{s}")
    else:
        first_result = 'last_assessment' not in ss
        cv = igd_meta['cv']
        has_cognitive = True
        has_eye = 'eye_fatigue_score' in ss
        inputs = dict(ss['patient_inputs'])
        age, gender, mode, genre = inputs['age'], inputs['gender'], inputs['mode'], inputs['genre']
        weekday_hours, weekend_hours, sleep_hours = inputs['weekday_hours'], inputs['weekend_hours'], inputs['sleep_hours']

        total_hours = round(5 * weekday_hours + 2 * weekend_hours, 1)
        X_user = igdf.features_from_inputs(inputs)

        # 1. Behaviour vote: predicted IGDS9-SF score -> tier probabilities
        pred_igds = predict_igds(X_user)
        beh_proba = igdf.tier_probabilities(pred_igds, igd_meta['sigma'])
        beh_proba, beh_conf = fe.behaviour_vote(beh_proba)

        # 2. Cognitive vote: measured Go/No-Go inhibition (literature prior); absent if tests not done
        if has_cognitive:
            cr = st.session_state['cognitive_results']
            commission = cr['battery']['go_nogo']['commission_rate']
            cog_proba, cog_conf = fe.cognitive_inhibition_vote(commission)
            rt_actual = cr['reaction_time_ms']
        else:
            cog_proba, cog_conf, rt_actual = np.ones(3) / 3, 0.0, None

        # 3. Eye-fatigue vote (literature prior); absent if no eye check
        if has_eye:
            fatigue_val = st.session_state['eye_fatigue_score']
            eye_proba, eye_conf = fe.eye_fatigue_vote(fatigue_val)
        else:
            fatigue_val, eye_proba, eye_conf = None, np.ones(3) / 3, 0.0

        # 4. Wellbeing vote: PHQ-4 mood + UCLA-3 loneliness (literature prior)
        wb = wellbeing_scores(inputs) if 'phq4' in inputs else None
        if wb:
            wb_proba, wb_conf = fe.wellbeing_vote(wb['phq4'], wb['lonely'])
            wb_risk_label = igdf.RISK_LABELS[int(np.argmax(wb_proba))]
        else:
            wb_proba, wb_conf, wb_risk_label = np.ones(3) / 3, 0.0, "Not measured"

        fused_proba, weights_dict = fe.fuse([(beh_proba, beh_conf), (cog_proba, cog_conf), (eye_proba, eye_conf),
                                             (wb_proba, wb_conf)])
        fused_pred_idx = int(np.argmax(fused_proba))
        # Few High cases in the training data pull predictions to the middle, so flag a sizeable High probability
        possible_high = fused_pred_idx < 2 and float(fused_proba[2]) >= POSSIBLE_HIGH_PROB
        risk_labels = igdf.RISK_LABELS
        fused_risk_label = risk_labels[fused_pred_idx]
        beh_risk_label = risk_labels[int(np.argmax(beh_proba))]
        cog_risk_label = risk_labels[int(np.argmax(cog_proba))] if has_cognitive else "Not measured"
        eye_risk_label = risk_labels[int(np.argmax(eye_proba))] if has_eye else "Not measured"

        # Behaviour profile (K-means on the real dataset)
        Xc = igd_clusters['scaler'].transform(X_user[igd_clusters['features']])
        cluster_idx = int(igd_clusters['kmeans'].predict(Xc)[0])
        profile = igd_clusters['profiles'][cluster_idx]
        archetype = profile['name']
        archetype_desc = (f"{profile['description']} Students in this group (n = {profile['n']}) averaged "
                          f"{profile['mean_igds']:.1f} on the IGDS9-SF.")

        color_map = {0: "green", 1: "orange", 2: "red"}
        icon_map = {0: "🟢", 1: "🟡", 2: "🔴"}
        advice_map = {
            0: "✅ Your gaming currently looks balanced. Keep an eye on late nights and weekend sessions.",
            1: "⚠️ Some gaming-disorder symptoms are likely. A guided self-help plan (CBT-based) is recommended.",
            2: "🚨 Several symptoms are likely. Please combine the self-help programme with professional support "
               "(Sri Lanka National Mental Health Helpline: 1926).",
        }

        st.session_state['last_assessment'] = {
            'rf_pred': beh_risk_label, 'rf_conf': beh_conf, 'rf_w': weights_dict['Behaviour_weight'],
            'cog_pred': cog_risk_label, 'cog_conf': cog_conf, 'cog_w': weights_dict['Cognitive_weight'],
            'eye_pred': eye_risk_label, 'eye_conf': eye_conf, 'eye_w': weights_dict['EyeFatigue_weight'],
            'wb_pred': wb_risk_label, 'wb_conf': wb_conf, 'wb_w': weights_dict['Wellbeing_weight'], 'wellbeing': wb,
            'fused_pred': fused_risk_label, 'fused_idx': fused_pred_idx, 'fused_proba': fused_proba,
            'pred_igds': pred_igds, 'archetype': archetype, 'archetype_desc': archetype_desc,
            'total_hours': total_hours, 'sleep_hours': sleep_hours, 'rt_actual': rt_actual, 'fatigue_val': fatigue_val,
            'features': X_user.iloc[0].to_dict(),
            'inputs': {k: v for k, v in inputs.items() if k != 'sports_level'},
            'rf_proba': [float(p) for p in beh_proba],
            'cog_proba': [float(p) for p in cog_proba],
            'eye_proba': [float(p) for p in eye_proba],
            'wb_proba': [float(p) for p in wb_proba],
            'used_measured_cognition': has_cognitive,
            'used_measured_eye': has_eye,
            'possible_high': possible_high,
        }

        # ── DISPLAY RESULTS SECTION ──────────────────────────────────────────
        st.subheader("⚡ Multimodal Late Fusion Assessment Results")
        col_res1, col_res2 = st.columns([1, 1])

        with col_res1:
            st.markdown(f"### Fused Final Risk: :{color_map[fused_pred_idx]}[{icon_map[fused_pred_idx]} {fused_risk_label}]")
            st.info(advice_map[fused_pred_idx])
            if possible_high:
                st.warning(f"⚠️ **Possible High risk:** there is a {fused_proba[2] * 100:.0f}% chance that your "
                           "score is in the High band. The model tends to under-estimate very high scores, because few "
                           "students in its training data were High risk — please treat this result seriously.")
            if fused_pred_idx > 0:
                st.markdown("➡️ Your personal plan is in **5️⃣ Solution & Plan**.")
            st.markdown(f"#### Predicted IGDS9-SF score: **{pred_igds:.1f}** / 45")
            st.progress(float(np.clip((pred_igds - 9) / 36, 0, 1)))
            st.caption(f"Typical error ±{cv['MAE']:.1f} points. Bands: ≤{igdf.TIER_LOW_MAX} low · "
                       f"{igdf.TIER_LOW_MAX + 1}–{igdf.TIER_HIGH_MIN - 1} medium · ≥{igdf.TIER_HIGH_MIN} high "
                       f"(training sample mean {igd_meta['target_mean']:.1f}).")
            st.markdown(f"#### 🧩 Behaviour profile: **{archetype}**")
            st.caption(archetype_desc)

        with col_res2:
            st.markdown("#### 🔬 Modality Weight & Confidence Breakdown")
            df_fusion_vis = pd.DataFrame({
                'Modality': [f'1. Behaviour model ({model_name})', '2. Cognitive vote (Go/No-Go prior)',
                             '3. Eye-fatigue vote (prior)', '4. Wellbeing vote (mood + loneliness prior)',
                             '🏆 Fused Consensus'],
                'Predicted Vote': [beh_risk_label, cog_risk_label, eye_risk_label, wb_risk_label, fused_risk_label],
                'Confidence': [f"{beh_conf * 100:.1f}%", f"{cog_conf * 100:.1f}%", f"{eye_conf * 100:.1f}%",
                               f"{wb_conf * 100:.1f}%", f"{float(np.max(fused_proba)) * 100:.1f}%"],
                'Fusion Weight': [f"{weights_dict['Behaviour_weight'] * 100:.1f}%",
                                  f"{weights_dict['Cognitive_weight'] * 100:.1f}%",
                                  f"{weights_dict['EyeFatigue_weight'] * 100:.1f}%",
                                  f"{weights_dict['Wellbeing_weight'] * 100:.1f}%", "100.0%"],
            })
            st.dataframe(df_fusion_vis, use_container_width=True, hide_index=True)
            st.caption("Low / Medium / High probabilities: " + " · ".join(
                f"{l.split()[0]} {p * 100:.0f}%" for l, p in zip(risk_labels, fused_proba)))
            if int(np.argmax(beh_proba)) != fused_pred_idx:
                st.warning(f"⚡ **Late Fusion Effect:** the cognitive/eye/wellbeing votes moved the result from "
                           f"**{beh_risk_label}** to **{fused_risk_label}**.")

        st.divider()
        st.subheader("🧠 What moves your predicted score")
        st.caption("Each bar shows how much your predicted IGDS9-SF score changes if that answer is replaced by the "
                   "typical answer in the training data (model-based sensitivity, not a causal effect).")
        sens = answer_sensitivity(X_user, pred_igds)
        fig_s, ax_s = plt.subplots(figsize=(7, 3.2))
        ax_s.barh(list(sens.keys()), list(sens.values()),
                  color=['#e74c3c' if v > 0 else '#3498db' for v in sens.values()], height=0.55)
        ax_s.axvline(0, color='grey', linewidth=0.8, linestyle='--')
        ax_s.set_xlabel('Change in predicted IGDS9-SF score vs typical answer (+ raises risk)', fontsize=8)
        ax_s.tick_params(labelsize=8)
        plt.tight_layout()
        st.pyplot(fig_s)
        plt.close(fig_s)

        st.divider()
        st.subheader("📄 Assessment Report (PDF)")
        pdf_payload = {
            'model_name': model_name, 'age': age, 'gender': gender, 'hours_wk': total_hours,
            'sleep_hours': sleep_hours, 'mode': mode, 'genre': genre,
            'rf_pred': beh_risk_label, 'rf_conf': beh_conf, 'rf_w': weights_dict['Behaviour_weight'],
            'cog_pred': cog_risk_label, 'cog_conf': cog_conf, 'cog_w': weights_dict['Cognitive_weight'],
            'eye_pred': eye_risk_label, 'eye_conf': eye_conf, 'eye_w': weights_dict['EyeFatigue_weight'],
            'wb_pred': wb_risk_label, 'wb_conf': wb_conf, 'wb_w': weights_dict['Wellbeing_weight'], 'wellbeing': wb,
            'risk_level': fused_risk_label, 'risk_idx': fused_pred_idx, 'pred_igds': pred_igds, 'mae': cv['MAE'],
            'archetype': archetype, 'archetype_desc': archetype_desc,
            'rt_measured': rt_actual,
            'mem_measured': st.session_state.get('mem_score') if has_cognitive else None,
            'att_measured': st.session_state.get('att_score') if has_cognitive else None,
            'eye_fatigue': st.session_state.get('eye_fatigue_score', 'N/A'),
            'eye_level': st.session_state.get('eye_fatigue_level', 'N/A'),
            'battery': st.session_state['cognitive_results'].get('battery', {}) if has_cognitive else {},
        }
        st.download_button(
            label="📥 Download Assessment Report (PDF)",
            data=generate_pdf_report_v5(pdf_payload),
            file_name=f"Gaming_Assessment_Report_{time.strftime('%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        sig = repr((inputs, ss['cognitive_results']['battery'], ss.get('eye_fatigue_score')))
        if ss.get('assessment_sig') != sig:
            ss['assessment_sig'] = sig
            ss.setdefault('history', []).append({
                'Student': f"#{len(st.session_state['history']) + 1}",
                'Hours/Wk': total_hours,
                'Sleep': sleep_hours,
                'Predicted IGDS9-SF': round(pred_igds, 1),
                'Behaviour Vote': beh_risk_label,
                'Fused Risk': fused_risk_label,
                'Profile': archetype,
            })
        if first_result:
            st.rerun()                           # redraw so the sidebar ticks this step straight away

        st.divider()
        next_button(STEPS[4], f"See your solution — Next: {STEPS[4]} ➡️")


# ════════════════════════════════════════════════════════════
#  STEP 5: SOLUTION & PERSONAL PLAN
# ════════════════════════════════════════════════════════════
elif page == STEPS[4]:
    st.title("5️⃣ Solution & Personal Plan")
    a = ss.get('last_assessment')
    if a is None:
        st.info("Complete steps 1–4 to get a plan matched to your result. You can still use the Gaming Reduction "
                "Program below.")
    else:
        risk = a['fused_pred']
        title, text = reduction_program.STEPS[risk]
        box = st.error if risk == "High Risk" else st.warning if risk == "Medium Risk" else st.success
        box(f"**{title}** — your result: **{risk}** (predicted IGDS9-SF {a['pred_igds']:.1f})\n\n{text}")
        if risk == "High Risk" or a.get('possible_high'):
            st.info(reduction_program.HELP_TEXT)

        # ① Habit changes with the biggest effect on the predicted score (real-data model)
        st.subheader("① Changes that would lower your risk most")
        changes = habit_changes(a)
        if changes:
            st.dataframe(pd.DataFrame(changes), use_container_width=True, hide_index=True)
            st.caption("Estimated by the behaviour model trained on real student data. It shows associations the model "
                       "learned, not a guaranteed effect — but these are the habits most linked to gaming-disorder symptoms.")
        else:
            st.success("Your sleep and gaming habits are already in the healthy range the model recognises. Keep them there.")

        # ② What the cognitive tests suggest
        st.subheader("② What your test results suggest")
        b = ss['cognitive_results']['battery']
        commission = b['go_nogo']['commission_rate']
        if commission >= 40:
            st.markdown(f"- **Impulse control:** you pressed on {commission}% of the red (No-Go) circles. Weaker response "
                        "inhibition is the cognitive pattern most linked to gaming disorder (Argyriou et al., 2017). "
                        "**Urge surfing** and **if-then plans** in the programme below target exactly this.")
        else:
            st.markdown(f"- **Impulse control:** good — you held back on {100 - commission}% of the red (No-Go) circles.")
        wb = a.get('wellbeing')
        if wb:
            if wb['phq2'] >= 3 or wb['gad2'] >= 3:
                feeling = " and ".join(n for n, s in [("low mood", wb['phq2']), ("worry / anxiety", wb['gad2'])] if s >= 3)
                st.markdown(f"- **Mood:** your answers show {feeling} on many days in the last two weeks. Many people game "
                            "to escape these feelings, which makes cutting down harder. **Talk to someone you trust** or the "
                            "university counsellor — or call the National Mental Health Helpline **1926** (free, 24 h).")
            if wb['lonely'] >= 6:
                st.markdown("- **Feeling connected:** you often feel left out or alone. Gaming can fill that gap, but "
                            "offline contact protects against problem gaming. Plan **one activity a week with other people** "
                            "(a sport, club or study group) in the programme below.")
            if wb['phq2'] < 3 and wb['gad2'] < 3 and wb['lonely'] < 6:
                st.markdown("- **Mood & connection:** good — no signs of low mood, anxiety or loneliness in your answers.")
        if b['nback']['d_prime'] < 1.5:
            st.markdown("- **Working memory** (2-Back) was on the lower side. Sleep and fewer late-night sessions help "
                        "concentration; re-test after two weeks of your plan.")
        if b['task_switching']['switch_cost'] >= 300:
            st.markdown("- **Switching between tasks** was slow. Plan study blocks without games or phone notifications "
                        "nearby, so you are not constantly switching.")

        # ③ Eye care
        st.subheader("③ Look after your eyes")
        fatigue = ss.get('eye_fatigue_score')
        weekly = a['total_hours']
        if fatigue is not None and fatigue >= 25 or weekly >= 21:
            st.markdown("- **20-20-20 rule:** every 20 minutes of screen time, look at something 20 feet (6 m) away for "
                        "20 seconds.\n- Keep the screen about an arm's length away, with the top at or slightly below "
                        "eye level, and avoid a screen brighter than the room.")
        else:
            st.markdown("- Your eyes looked awake in the photo. Long gaming sessions still strain the eyes, so keep taking regular screen breaks while you play.")
        with st.expander("⏱️ 20-minute screen-break timer"):
            if not ss.get('pomo_active'):
                if st.button("▶️ Start 20-minute timer"):
                    ss['pomo_active'], ss['pomo_start'] = True, time.time()
                    st.rerun()
            else:
                remaining = max(0, int(1200 - (time.time() - ss['pomo_start'])))
                st.metric("Time until your next eye break", f"{remaining // 60:02d}:{remaining % 60:02d}")
                if remaining == 0:
                    st.warning("🔔 Look at something 20 feet away for 20 seconds now!")
                c1, c2 = st.columns(2)
                c1.button("🔄 Refresh")
                if c2.button("⏹️ Stop timer"):
                    ss['pomo_active'] = False
                    st.rerun()

    # ④ The guided self-help programme
    st.divider()
    st.subheader("④ Your Gaming Reduction Program")
    reduction_program.render_program_page(embedded=True)


# ════════════════════════════════════════════════════════════
#  STUDENT HISTORY
# ════════════════════════════════════════════════════════════
elif page == "📋 Student History":
    st.title("📋 Assessment History (this session)")
    if not ss.get('history'):
        st.info(f"No assessments yet. Start with **{STEPS[0]}**.")
    else:
        st.dataframe(pd.DataFrame(ss['history']), use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear History"):
            ss['history'] = []
            st.rerun()


# ════════════════════════════════════════════════════════════
#  RESEARCHER TOOLS: CONTINUOUS EYE SESSION MONITOR
# ════════════════════════════════════════════════════════════
elif page == "🎮 Eye Session Monitor":
    st.title("🎮 Eye Session Monitor (continuous, while gaming)")
    eye_sessions.render_monitor_tab()


# ════════════════════════════════════════════════════════════
#  RESEARCHER TOOLS: EYE–GAMING STUDY
# ════════════════════════════════════════════════════════════
elif page == "👁️ Eye–Gaming Study":
    eye_gaming_study.render_page()


# ════════════════════════════════════════════════════════════
#  RESEARCHER TOOLS: MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance Dashboard & Dataset Analytics")

    t_perf1, t_perf2, t_perf3, t_perf4 = st.tabs(["Behaviour Model (real data)", "👁️ Eye Fatigue Models",
                                                  "Multimodal Fusion", "Dataset Analytics"])

    with t_perf1:
        cv, base_cv = igd_meta['cv'], igd_meta['baseline_cv']
        st.subheader(f"{model_name} — {igd_meta['cv_protocol']}")
        st.caption(f"Training data: {igd_meta['dataset']}. Target: IGDS9-SF total score (9–45).")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²", f"{cv['R2']:.3f}", f"baseline {base_cv['R2']:.3f}", delta_color="off")
        c2.metric("MAE (points)", f"{cv['MAE']:.2f}", f"baseline {base_cv['MAE']:.2f}", delta_color="off")
        c3.metric("Spearman ρ", f"{cv['Spearman']:.3f}")
        c4.metric("AUC (IGDS9-SF ≥ 21)", f"{cv['auc_ge21']:.3f}")
        st.caption(f"Improvement over the 'predict the mean' baseline across the folds: Wilcoxon p = "
                   f"{igd_meta['wilcoxon_p_vs_baseline']:.1e}. Risk tiers: accuracy {cv['tier_accuracy'] * 100:.1f}%, "
                   f"quadratic-weighted κ = {cv['tier_weighted_kappa']:.2f}.")

        tabs = st.tabs(["Model comparison", "Predicted vs actual", "Feature importance", "Risk tiers"])
        for tab, fname in zip(tabs, ['real1_model_comparison.png', 'real2_predicted_vs_actual.png',
                                     'real3_feature_importance.png', 'real4_tier_confusion.png']):
            path = os.path.join(BASE, 'results', fname)
            with tab:
                if os.path.exists(path):
                    st.image(path, use_container_width=True)
                else:
                    st.warning("Run `python notebooks/07_train_real_igd_model.py` to generate this chart.")

        comp_path = os.path.join(BASE, 'results', 'real_model_comparison.csv')
        if os.path.exists(comp_path):
            with st.expander("All models (cross-validated)"):
                comp = pd.read_csv(comp_path)[['model', 'MAE', 'RMSE', 'R2', 'Spearman', 'auc_ge21',
                                               'tier_accuracy', 'tier_weighted_kappa']]
                st.dataframe(comp.round(3), use_container_width=True, hide_index=True)

    with t_perf2:
        em = eye_analyzer.meta
        st.subheader("Eye-State Model (open vs closed eyes)")
        st.caption(f"MRL Eye Dataset (Fusek, 2018). Subject-wise test: {em['test_people']} people "
                   f"({em['test_images']:,} images) who were never seen in training.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy (unseen people)", f"{em['accuracy'] * 100:.1f}%")
        c2.metric("AUC", f"{em['auc']:.3f}")
        c3.metric("Closed eyes caught", f"{em['recall_closed'] * 100:.1f}%")
        c4.metric("Open eyes correct", f"{em['recall_open'] * 100:.1f}%")
        if 'cew_test' in em:
            ct = em['cew_test']
            st.caption(f"Held-out real-world colour faces (CEW, {ct['images']} eye crops): accuracy "
                       f"{ct['accuracy'] * 100:.1f}%, AUC {ct['auc']:.3f}. Decision threshold {em['threshold']:.2f} "
                       f"chosen on validation people only; predictions averaged with the mirror image.")
        for sub, fname in zip(st.tabs(["Training History", "Confusion Matrix"]),
                              ['eye3_state_training_history.png', 'eye4_state_confusion_matrix.png']):
            path = os.path.join(BASE, 'results', fname)
            with sub:
                if os.path.exists(path):
                    st.image(path, use_container_width=True)
                else:
                    st.warning("Run `python notebooks/08_train_eye_state_model.py` to generate this chart.")
        st.info("Earlier eye models were retired: the baseline CNN was trained on the wrong label (the MRL "
                "'reflections' field instead of eye state), and the app read the ResNet50's output the wrong way round.")

    with t_perf3:
        st.subheader("Confidence-Weighted Late Fusion")
        st.markdown(f"""
| Vote | Source | Maximum influence |
|---|---|---|
| Behaviour | {model_name}, trained on real IGDS9-SF data | its own confidence (max probability) |
| Cognitive | Go/No-Go commission errors — literature prior | capped at r = {fe.COGNITIVE_LITERATURE_WEIGHT:.2f} (d = {fe.INHIBITION_EFFECT_D}, Argyriou et al., 2017) |
| Eye fatigue | Eye-state CNN (share of closed eyes) — literature prior | capped at {fe.EYE_FATIGUE_LITERATURE_WEIGHT} (Liu & Zhou, 2024) |
| Wellbeing | PHQ-4 mood (Kroenke et al., 2009) + UCLA-3 loneliness (Hughes et al., 2004) — literature prior | capped at r = {fe.WELLBEING_LITERATURE_WEIGHT:.2f} (Yoon et al., 2021: depression .33, anxiety .29, loneliness .29) |
""")
        st.info("Whether fusion improves on the behaviour model alone can only be tested on data where the same students "
                "have the questionnaire, cognitive tests and eye check; no public dataset has all of them. The earlier "
                "76.3% → 79.3% fusion result was computed on synthetic data and has been withdrawn.")

    with t_perf4:
        st.subheader("Training Dataset (real survey data)")
        st.caption(igd_meta['dataset'])
        try:
            raw_df = igdf.load_dataset()
            X_all = igdf.features_from_dataset(raw_df)
            c1, c2, c3 = st.columns(3)
            c1.metric("Gamers with IGDS9-SF", len(raw_df))
            c2.metric("Mean IGDS9-SF", f"{raw_df['IGD Total'].mean():.1f}", f"SD {raw_df['IGD Total'].std():.1f}", delta_color="off")
            c3.metric("Low / Medium / High", " / ".join(str(c) for c in igd_meta['tiers']['counts']))
            fig_d, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))
            ax1.hist(raw_df['IGD Total'], bins=range(9, 47, 2), color='#3498db', edgecolor='white')
            for cut in (igdf.TIER_LOW_MAX + 0.5, igdf.TIER_HIGH_MIN - 0.5):
                ax1.axvline(cut, color='#e74c3c', ls='--', lw=1)
            ax1.set_xlabel('IGDS9-SF total', fontsize=8)
            ax1.set_ylabel('Students', fontsize=8)
            ax1.set_title('Distribution of gaming-disorder scores', fontsize=9, fontweight='bold')
            ax2.scatter(X_all['weekly_hours'], raw_df['IGD Total'], s=12, alpha=0.5, color='#8e44ad')
            ax2.set_xlabel('Weekly gaming (h, from questionnaire ranges)', fontsize=8)
            ax2.set_ylabel('IGDS9-SF total', fontsize=8)
            ax2.set_title('Gaming time vs gaming-disorder score', fontsize=9, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_d)
            plt.close(fig_d)
            st.dataframe(X_all.describe().T.round(2), use_container_width=True)
        except FileNotFoundError:
            st.warning("Dataset file not found in data/external/srilanka_igd/.")



# ── Footer ───────────────────────────────────────────────────
st.sidebar.divider()
if os.environ.get('APP_LAN_URL'):
    st.sidebar.info(f"📱 **Open on a phone** (same Wi-Fi):\n\n`{os.environ['APP_LAN_URL']}`")
st.sidebar.caption(f"🎓 BSc (Hons) Computer Science | KDU\n{model_name} on real IGDS9-SF data | "
                   f"CV R² {igd_meta['cv']['R2']:.2f}, ρ {igd_meta['cv']['Spearman']:.2f}")
