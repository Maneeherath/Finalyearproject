# ============================================================
#  FINAL WEB APP v5
#  Behaviour model trained on REAL data: 322 Sri Lankan students with the
#  IGDS9-SF (Manchanayake, 2021; notebooks/07_train_real_igd_model.py)
#  Features:
#    1. Multimodal Late Fusion Engine (Confidence-Weighted Pooling)
#    2. Live Webcam Eye Fatigue Scanner & OpenCV Processing
#    3. Interactive "What-If" Sensitivity Simulator & Risk Lab
#    4. Multidimensional Gamer Persona Radar Chart (Spider Web)
#    5. 20-20-20 Eye Strain Pomodoro & Routine Planner
#    6. Upgraded ReportLab PDF Assessment Generator
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
from tensorflow.keras.models import load_model

# ReportLab imports for PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Import Late Fusion Engine module
import fusion_engine as fe
# Shared feature encoding for the real-data IGD model
import igd_features as igdf
# Research-based cognitive battery (2-Back, Task-Switching, Mental Rotation, Go/No-Go)
import research_cognitive_tests as rct
# Validation study (IGDS9-SF criterion + SUS usability) and continuous eye-session logs
import study_mode
import eye_sessions
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
    # Eye-fatigue ResNet50, trained on the real MRL Eye Dataset (Fusek, 2018)
    resnet_eye_path = os.path.join(BASE, 'models', 'resnet_eye_fatigue_model.h5')
    resnet = load_model(resnet_eye_path, compile=False) if os.path.exists(resnet_eye_path) else None
    return igd['model'], igd['meta'], clusters, resnet


try:
    igd_model, igd_meta, igd_clusters, resnet_eye_model = load_all_models()
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
        [Paragraph("Eye-fatigue vote (ResNet50 + OpenCV, literature prior)", body_style),
         Paragraph(f"<b>{results_dict['eye_pred']}</b>", body_style),
         Paragraph(f"Conf: {results_dict['eye_conf']:.2f} (Weight: {results_dict['eye_w']:.2f})", body_style)],
        [Paragraph("<b>Fused result</b>", bold_body_style), Paragraph(f"<b>{results_dict['risk_level']}</b>", bold_body_style),
         Paragraph("Log-linear opinion pooling", bold_body_style)],
        [Paragraph("Predicted IGDS9-SF score (9-45)", body_style),
         Paragraph(f"<b>{results_dict['pred_igds']:.1f}</b>", bold_body_style),
         Paragraph(f"Typical error +/- {results_dict['mae']:.1f} points", body_style)],
    ]
    t_fusion = Table(fusion_data, colWidths=[200, 130, 170])
    t_fusion.setStyle(table_style)
    t_fusion.setStyle(TableStyle([('BACKGROUND', (0, 4), (-1, 4), colors.HexColor("#FEF3C7"))]))
    story.append(t_fusion)
    story.append(Spacer(1, 8))

    # 3. Behaviour profile
    story.append(Paragraph("3. Behaviour Profile (K-means clusters of the real dataset)", h1_style))
    story.append(Paragraph(f"<b>Profile:</b> {results_dict['archetype']}", bold_body_style))
    story.append(Paragraph(results_dict['archetype_desc'], body_style))
    story.append(Spacer(1, 8))

    # 4. Measured tests
    story.append(Paragraph("4. Measured Cognitive Tests & Eye Fatigue", h1_style))
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
        [Paragraph("Eye fatigue index", body_style),
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

# ── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.title("🎮 Gaming Predictor (v5)")
st.sidebar.markdown("**D/BCS/23/0009 | P R M K Herath**")
st.sidebar.caption("General Sir John Kotelawala Defence University")
st.sidebar.divider()

page = st.sidebar.radio("Navigation Menu", [
    "🏠 Home & Multimodal Prediction",
    "🧠 Cognitive & Biometric Tests",
    "👁️ Eye Fatigue Analysis",
    "🧪 What-If Lab & Radar Profiling",
    "⏱️ Detox & Routine Planner",
    "🛡️ Gaming Reduction Program",
    "📊 Model Performance & Insights",
    "📋 Student History",
    "📝 Validation Study"
], key="nav_page")

# Quick Sidebar Info Badges
if 'cognitive_results' in st.session_state:
    cr = st.session_state['cognitive_results']
    st.sidebar.success(f"🧠 Cognitive Loaded: RT {cr['reaction_time_ms']}ms")
if 'eye_fatigue_score' in st.session_state:
    st.sidebar.info(f"👁️ Eye Fatigue: {st.session_state['eye_fatigue_score']}/100")


# ════════════════════════════════════════════════════════════
#  PAGE 1: HOME & MULTIMODAL PREDICTION
# ════════════════════════════════════════════════════════════
if page == "🏠 Home & Multimodal Prediction":
    st.title("🎮 Gaming Behavior & Cognitive Performance Predictor")
    st.markdown("**Final Year Research Project | BSc (Hons) Computer Science | KDU**")

    cv = igd_meta['cv']
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Behaviour model", model_name, f"trained on real data (n = {igd_meta['n']})", delta_color="off")
    m2.metric("Cross-validated R²", f"{cv['R2']:.2f}", f"MAE {cv['MAE']:.1f} IGDS9-SF points", delta_color="off")
    m3.metric("Rank correlation (ρ)", f"{cv['Spearman']:.2f}", "predicted vs actual IGDS9-SF", delta_color="off")
    m4.metric("Fusion", "3 modalities", "behaviour · cognition · eyes", delta_color="off")
    st.caption(f"Model trained on {igd_meta['dataset']}. Screening estimate only — not a diagnosis.")

    has_cognitive = 'cognitive_results' in st.session_state
    has_eye = 'eye_fatigue_score' in st.session_state
    if has_cognitive or has_eye:
        loaded_str = [s for s, ok in [("Cognitive Tests", has_cognitive), ("Eye Fatigue Check", has_eye)] if ok]
        st.success(f"✅ **Multimodal mode active:** using [{', '.join(loaded_str)}]")
    else:
        st.info("💡 **Tip:** Complete **🧠 Cognitive & Biometric Tests** and **👁️ Eye Fatigue Analysis** first to add the "
                "cognitive and eye-fatigue votes to the fusion.")

    st.divider()
    st.subheader("📋 Your Gaming, Sleep & Wellbeing Profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 About you**")
        age = st.slider("Age", 16, 35, 21)
        gender = st.selectbox("Gender", ["Male", "Female"])
        start_age = st.slider("Age you started gaming regularly", 3, 35, 10)
        sports = st.selectbox("Involvement in sports", ["None", "Minor", "Moderate", "Major"])

    with col2:
        st.markdown("**🎮 Gaming habits**")
        weekday_hours = st.slider("Gaming on a typical weekday (hours/day)", 0.0, 12.0, 1.5, step=0.5)
        weekend_hours = st.slider("Gaming on a typical weekend day (hours/day)", 0.0, 14.0, 2.5, step=0.5)
        mode = st.selectbox("Main way you play", igdf.MODES)
        genre = st.selectbox("Main game genre", igdf.GENRES)
        device = st.selectbox("Main device", igdf.DEVICES)

    with col3:
        st.markdown("**🌙 Sleep & wellbeing**")
        sleep_hours = st.slider("Sleep per night (hours)", 3.0, 10.0, 7.0, step=0.5)
        sleep_quality = st.slider("Sleep quality (0 = very poor, 10 = excellent)", 0, 10, 7)
        self_esteem = st.slider("How would you rate your self-esteem? (1 = very low, 5 = very high)", 1, 5, 4)
        few_friends = st.slider("\"I have few or no close friends\" (1 = not true, 5 = very true)", 1, 5, 2)

    if start_age > age:
        st.warning("The age you started gaming is later than your current age — please check.")

    st.divider()

    if st.button("🚀 Perform Multimodal System Assessment", use_container_width=True, type="primary"):
        total_hours = round(5 * weekday_hours + 2 * weekend_hours, 1)
        inputs = dict(age=age, gender=gender, start_age=min(start_age, age), sports=sports,
                      weekday_hours=weekday_hours, weekend_hours=weekend_hours, mode=mode, genre=genre, device=device,
                      sleep_hours=sleep_hours, sleep_quality=sleep_quality, self_esteem=self_esteem,
                      few_friends=few_friends, sports_level=["None", "Minor", "Moderate", "Major"].index(sports))
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

        fused_proba, weights_dict = fe.fuse([(beh_proba, beh_conf), (cog_proba, cog_conf), (eye_proba, eye_conf)])
        fused_pred_idx = int(np.argmax(fused_proba))
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
            'fused_pred': fused_risk_label, 'fused_idx': fused_pred_idx, 'fused_proba': fused_proba,
            'pred_igds': pred_igds, 'archetype': archetype, 'archetype_desc': archetype_desc,
            'total_hours': total_hours, 'sleep_hours': sleep_hours, 'rt_actual': rt_actual, 'fatigue_val': fatigue_val,
            'features': X_user.iloc[0].to_dict(),
            # Extra fields recorded by the validation study (Study Mode)
            'inputs': {k: v for k, v in inputs.items() if k != 'sports_level'},
            'rf_proba': [float(p) for p in beh_proba],
            'cog_proba': [float(p) for p in cog_proba],
            'eye_proba': [float(p) for p in eye_proba],
            'used_measured_cognition': has_cognitive,
            'used_measured_eye': has_eye,
        }

        # ── DISPLAY RESULTS SECTION ──────────────────────────────────────────
        st.subheader("⚡ Multimodal Late Fusion Assessment Results")
        col_res1, col_res2 = st.columns([1, 1])

        with col_res1:
            st.markdown(f"### Fused Final Risk: :{color_map[fused_pred_idx]}[{icon_map[fused_pred_idx]} {fused_risk_label}]")
            st.info(advice_map[fused_pred_idx])
            if fused_pred_idx > 0:
                st.markdown("➡️ Open **🛡️ Gaming Reduction Program** in the menu for a personalised, research-based plan to cut down.")
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
                             '3. Eye-fatigue vote (prior)', '🏆 Fused Consensus'],
                'Predicted Vote': [beh_risk_label, cog_risk_label, eye_risk_label, fused_risk_label],
                'Confidence': [f"{beh_conf * 100:.1f}%", f"{cog_conf * 100:.1f}%", f"{eye_conf * 100:.1f}%",
                               f"{float(np.max(fused_proba)) * 100:.1f}%"],
                'Fusion Weight': [f"{weights_dict['Behaviour_weight'] * 100:.1f}%",
                                  f"{weights_dict['Cognitive_weight'] * 100:.1f}%",
                                  f"{weights_dict['EyeFatigue_weight'] * 100:.1f}%", "100.0%"],
            })
            st.dataframe(df_fusion_vis, use_container_width=True, hide_index=True)
            st.caption("Low / Medium / High probabilities: " + " · ".join(
                f"{l.split()[0]} {p * 100:.0f}%" for l, p in zip(risk_labels, fused_proba)))
            if int(np.argmax(beh_proba)) != fused_pred_idx:
                st.warning(f"⚡ **Late Fusion Effect:** the cognitive/eye votes moved the result from "
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

        if 'history' not in st.session_state:
            st.session_state['history'] = []
        st.session_state['history'].append({
            'Student': f"#{len(st.session_state['history']) + 1}",
            'Hours/Wk': total_hours,
            'Sleep': sleep_hours,
            'Predicted IGDS9-SF': round(pred_igds, 1),
            'Behaviour Vote': beh_risk_label,
            'Fused Risk': fused_risk_label,
            'Profile': archetype,
        })

# ════════════════════════════════════════════════════════════
#  PAGE 2: COGNITIVE & BIOMETRIC TESTS
# ════════════════════════════════════════════════════════════
elif page == "🧠 Cognitive & Biometric Tests":
    st.title("🧠 Research-Based Cognitive Assessment Battery")
    st.markdown("Four paradigms taken from peer-reviewed studies of cognition in video-game players and internet-gaming "
                "disorder: working-memory updating, response inhibition, cognitive flexibility and visuospatial processing.")

    rct.render_battery()

    st.divider()
    battery = rct.battery_results()

    if len(battery) == 4:
        # Aggregate the battery into the features used by the fusion engine and K-Means profiler
        mem_pct, att_pct = rct.domain_scores(battery)

        st.session_state['cognitive_results'] = {
            'reaction_time_ms':    battery['go_nogo']['mean_go_rt'],
            'memory_score_pct':    mem_pct,
            'attention_score_pct': att_pct,
            'battery':             battery
        }
        
        # Populate old variables so the PDF Generator and Fusion engine don't break
        st.session_state['mem_score'] = mem_pct
        st.session_state['att_score'] = att_pct
        
        st.success("🎉 All four research-based assessments completed and synced to the prediction engine!")
    else:
        st.caption(f"Progress: {len(battery)}/4 tests completed. Results sync to the prediction engine once all four are done.")

    if st.button("🔄 Reset Assessment Suite"):
        for k in ['cognitive_results', 'mem_score', 'att_score'] + rct.RESET_KEYS:
            st.session_state.pop(k, None)
        st.rerun()


# ════════════════════════════════════════════════════════════
#  PAGE 3: EYE FATIGUE ANALYSIS (NOVELTY #2: LIVE WEBCAM SCANNER)
# ════════════════════════════════════════════════════════════
elif page == "👁️ Eye Fatigue Analysis":
    st.title("👁️ Eye Fatigue Analysis & Computer Vision Processing")
    tab_scan, tab_session = st.tabs(["📸 Single Scan", "🎮 Gaming Session Monitor (continuous)"])

    with tab_session:
        eye_sessions.render_monitor_tab()

    with tab_scan:
        st.markdown("Capture a live photo using your webcam or upload a photo to analyze visual strain, redness, and aperture openness.")

        # Choice of Input Method
        # No default selection: the webcam only turns on when chosen, so it stays free for the session monitor
        input_source = st.radio("Choose Input Method:", ["📸 Live Webcam Camera", "📁 Upload Image File"], horizontal=True, index=None)

        pil_img = None

        if input_source == "📸 Live Webcam Camera":
            cam_photo = st.camera_input("Take a photo of your face / eyes")
            if cam_photo:
                pil_img = Image.open(cam_photo).convert('RGB')
        elif input_source == "📁 Upload Image File":
            uploaded_file = st.file_uploader("Upload Eye Photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                pil_img = Image.open(uploaded_file).convert('RGB')

        if pil_img is not None:
            img_array = np.array(pil_img)
        
            col1, col2 = st.columns(2)
            with col1:
                st.image(pil_img, caption="Original Input", use_container_width=True)

            with st.spinner("🔍 Processing OpenCV Eye Biomarkers..."):
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                eyes = eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

                annotated = img_array.copy()
                eye_detected = len(eyes) > 0
                openness = 50.0

                if eye_detected:
                    ex, ey, ew, eh = max(eyes, key=lambda e: e[2]*e[3])
                    cv2.rectangle(annotated, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    cv2.putText(annotated, "Eye Detected", (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    eye_gray = gray[ey:ey+eh, ex:ex+ew]
                    _, bright = cv2.threshold(eye_gray, 180, 255, cv2.THRESH_BINARY)
                    openness = round(np.sum(bright > 0) / bright.size * 100, 2)

                r = img_array[:, :, 0].astype(float)
                g = img_array[:, :, 1].astype(float)
                b = img_array[:, :, 2].astype(float)
                redness = round(float(np.mean(np.clip(r - (g + b) / 2, 0, 255))) / 255 * 100, 2)
                brightness = round(float(np.mean(gray)), 2)
                contrast = round(float(np.std(gray)), 2)

                # Compute OpenCV Biometric Fatigue Heuristics (0-100)
                opencv_fatigue = 0.0
                opencv_fatigue += 35 if openness < 15 else 20 if openness < 30 else 10 if openness < 45 else 0
                opencv_fatigue += 30 if redness > 25 else 15 if redness > 15 else 8 if redness > 8 else 0
                opencv_fatigue += 20 if brightness < 80 else 10 if brightness < 120 else 0
                opencv_fatigue += 15 if contrast < 30 else 8 if contrast < 50 else 0
                opencv_fatigue = round(min(100, opencv_fatigue), 1)

                # Deep ResNet50 Transfer Learning Live Inference
                resnet_fatigue = None
                if resnet_eye_model is not None:
                    try:
                        if eye_detected:
                            crop = img_array[ey:ey+eh, ex:ex+ew]
                        else:
                            crop = img_array
                        crop_resized = cv2.resize(crop, (64, 64))
                        crop_norm = np.expand_dims(crop_resized.astype('float32') / 255.0, axis=0)
                        p_alert = float(resnet_eye_model.predict(crop_norm, verbose=0)[0][0])
                        # MRL model output: 1.0 = Alert, 0.0 = Fatigued
                        resnet_fatigue = round(float(1.0 - p_alert) * 100, 1)
                        # Unified Score: 70% Deep ResNet50 Neural Net + 30% OpenCV Biometrics
                        fatigue = round(0.70 * resnet_fatigue + 0.30 * opencv_fatigue, 1)
                    except Exception as ex_pred:
                        fatigue = opencv_fatigue
                else:
                    fatigue = opencv_fatigue

            with col2:
                st.image(annotated, caption="OpenCV Cascade & Scleral Segmentation", use_container_width=True)

            st.divider()
            level = ("Alert" if fatigue < 25 else "Mildly Fatigued" if fatigue < 50
                     else "Fatigued" if fatigue < 75 else "Severely Fatigued")
            color = "green" if fatigue < 25 else "orange" if fatigue < 50 else "red"
            advice = ("No significant visual fatigue detected." if fatigue < 25
                      else "Mild eye strain detected. Take a brief break." if fatigue < 50
                      else "Noticeable eye strain. Apply 20-20-20 rule." if fatigue < 75
                      else "Severe ocular fatigue. Rest eyes immediately.")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Eye Fatigue Score", f"{fatigue}/100")
            c2.metric("Clinical Level", level)
            c3.metric("Aperture Openness", f"{openness:.1f}%")
            c4.metric("Scleral Redness", f"{redness:.1f}%")
        
            if resnet_fatigue is not None:
                st.info(f"🧠 **Deep ResNet50 Neural Net Output:** {resnet_fatigue:.1f}/100 Visual Strain Probability | **OpenCV Biometrics:** {opencv_fatigue:.1f}/100")
            
            st.markdown(f"**Status:** :{color}[**{advice}**]")

            st.session_state['eye_fatigue_score'] = fatigue
            st.session_state['eye_fatigue_level'] = level
            st.session_state['eye_fatigue_source'] = 'single_scan'
            st.success("✅ Deep ResNet50 & OpenCV Eye Fatigue metrics saved and synced to the Multimodal Late Fusion Engine.")


# ════════════════════════════════════════════════════════════
#  PAGE 4: WHAT-IF LAB & RADAR PROFILING (NOVELTY #3 & #4)
# ════════════════════════════════════════════════════════════
elif page == "🧪 What-If Lab & Radar Profiling":
    st.title("🧪 What-If Simulator & Radar Profiling")

    last = st.session_state.get('last_assessment')
    base_feats = pd.DataFrame([last['features']])[igdf.FEATURES] if last and 'features' in last else \
        pd.DataFrame([igd_meta['feature_medians']])[igdf.FEATURES]
    if not last:
        st.info("Showing a typical student from the training data. Run an assessment on the Home page to start from your own answers.")

    tab_sim, tab_radar = st.tabs(["🧪 What-If Habit Simulator", "🕸️ Radar Profile vs Real Student Groups"])

    with tab_sim:
        st.markdown("Change your habits below and see how the **real-data model's** predicted IGDS9-SF score and risk "
                    "probabilities respond. This shows what the model has learned, not a guaranteed effect.")
        b = base_feats.iloc[0]
        c1, c2 = st.columns(2)
        sim_wd = c1.slider("Weekday gaming (hours/day)", 0.0, 12.0, float(b['weekday_hours']), step=0.5)
        sim_we = c1.slider("Weekend gaming (hours/day)", 0.0, 14.0, float(b['weekend_hours']), step=0.5)
        sim_sleep = c2.slider("Sleep (hours/night)", 3.0, 10.0, float(b['sleep_hours']), step=0.5)
        sim_sq = c2.slider("Sleep quality (0-10)", 0, 10, int(b['sleep_quality']))

        X_sim = base_feats.copy()
        X_sim['weekday_hours'], X_sim['weekend_hours'] = sim_wd, sim_we
        X_sim['weekly_hours'] = 5 * sim_wd + 2 * sim_we
        X_sim['sleep_hours'], X_sim['sleep_quality'] = sim_sleep, sim_sq

        base_score, sim_score = predict_igds(base_feats), predict_igds(X_sim)
        base_p = igdf.tier_probabilities(base_score, igd_meta['sigma'])
        sim_p = igdf.tier_probabilities(sim_score, igd_meta['sigma'])

        st.divider()
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Predicted IGDS9-SF", f"{sim_score:.1f}", f"{sim_score - base_score:+.1f} vs now", delta_color="inverse")
        mc2.metric("P(High risk)", f"{sim_p[2] * 100:.1f}%", f"{(sim_p[2] - base_p[2]) * 100:+.1f} pts", delta_color="inverse")
        mc3.metric("P(Low risk)", f"{sim_p[0] * 100:.1f}%", f"{(sim_p[0] - base_p[0]) * 100:+.1f} pts")
        mc4.metric("Weekly gaming", f"{5 * sim_wd + 2 * sim_we:.1f} h", f"{(5 * sim_wd + 2 * sim_we) - float(b['weekly_hours']):+.1f} h",
                   delta_color="inverse")

        fig_sim, (ax_p, ax_c) = plt.subplots(1, 2, figsize=(10, 3.2))
        x = np.arange(3)
        ax_p.bar(x - 0.18, base_p * 100, width=0.36, label='Now', color='#95a5a6')
        ax_p.bar(x + 0.18, sim_p * 100, width=0.36, label='Simulated', color=['#2ecc71', '#f39c12', '#e74c3c'])
        ax_p.set_xticks(x, ['Low', 'Medium', 'High'])
        ax_p.set_ylabel('Probability (%)', fontsize=8)
        ax_p.set_title('Risk probabilities', fontsize=9, fontweight='bold')
        ax_p.legend(fontsize=7)
        hours = np.arange(0, 10.5, 0.5)
        curve = []
        for h in hours:
            X_h = X_sim.copy()
            X_h['weekday_hours'] = h
            X_h['weekly_hours'] = 5 * h + 2 * sim_we
            curve.append(predict_igds(X_h))
        ax_c.plot(hours, curve, color='#8e44ad', lw=2)
        ax_c.axvline(sim_wd, color='grey', ls='--', lw=0.8)
        for cut in (igdf.TIER_LOW_MAX + 0.5, igdf.TIER_HIGH_MIN - 0.5):
            ax_c.axhline(cut, color='#bdc3c7', ls=':', lw=0.8)
        ax_c.set_xlabel('Weekday gaming (h/day), other answers fixed', fontsize=8)
        ax_c.set_ylabel('Predicted IGDS9-SF', fontsize=8)
        ax_c.set_title('Model response to weekday gaming', fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_sim)
        plt.close(fig_sim)
        st.caption(f"The training data covered weekday gaming up to 9 h/day; predictions beyond the observed range "
                   f"flatten out because tree-based models cannot extrapolate.")

    with tab_radar:
        st.markdown("Your answers compared with the average answers of **real students** in the training data who scored "
                    f"low (IGDS9-SF ≤ {igdf.TIER_LOW_MAX}) or elevated (≥ 21).")
        ref = igd_meta['radar_reference']
        axes = [('weekday_hours', 'Weekday gaming', True), ('weekend_hours', 'Weekend gaming', True),
                ('sleep_hours', 'Sleep hours', False), ('sleep_quality', 'Sleep quality', False),
                ('self_esteem', 'Self-esteem', False), ('few_friends', 'Friendships', True)]

        def to_scale(feat, value, invert):
            lo, hi = ref['ranges'][feat]
            v = (value - lo) / (hi - lo) * 100 if hi > lo else 50
            return float(np.clip(100 - v if invert else v, 0, 100))

        user = [to_scale(f, float(base_feats.iloc[0][f]), inv) for f, _, inv in axes]
        low = [to_scale(f, ref['low_risk'][f], inv) for f, _, inv in axes]
        high = [to_scale(f, ref['elevated'][f], inv) for f, _, inv in axes]
        angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist()
        angles += angles[:1]
        fig_radar, ax_radar = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True))
        ax_radar.set_theta_offset(np.pi / 2)
        ax_radar.set_theta_direction(-1)
        ax_radar.set_xticks(angles[:-1], [f"{lab}\n(higher = healthier)" if i == 0 else lab for i, (_, lab, _) in enumerate(axes)],
                            fontsize=8)
        ax_radar.set_ylim(0, 100)
        ax_radar.set_yticks([25, 50, 75, 100], ["25", "50", "75", "100"], color="grey", fontsize=7)
        for vals, label, color, style in [(user, 'You', '#3498db', 'solid'),
                                          (low, f"Low IGDS9-SF students (n = {ref['n_low']})", '#2ecc71', 'dashed'),
                                          (high, f"Elevated IGDS9-SF students (n = {ref['n_elevated']})", '#e74c3c', 'dotted')]:
            ax_radar.plot(angles, vals + vals[:1], linewidth=2 if label == 'You' else 1.5, linestyle=style, label=label, color=color)
            if label == 'You':
                ax_radar.fill(angles, vals + vals[:1], color, alpha=0.2)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.45, 1.12), fontsize=7)
        ax_radar.set_title('Healthy-habits radar (all axes: higher = healthier)', fontsize=10, fontweight='bold', y=1.1)
        plt.tight_layout()
        st.pyplot(fig_radar)
        plt.close(fig_radar)
        st.caption("Gaming hours and 'few friends' are inverted so that, on every axis, further out means healthier.")

# ════════════════════════════════════════════════════════════
#  PAGE 5: DETOX & ROUTINE PLANNER (NOVELTY #5: 20-20-20 WIDGET)
# ════════════════════════════════════════════════════════════
elif page == "⏱️ Detox & Routine Planner":
    st.title("⏱️ Ergonomic Detox & Routine Planner")
    
    col_pomo, col_budget = st.columns([1, 1])
    
    with col_pomo:
        st.subheader("👁️ 20-20-20 Eye Strain Pomodoro Timer Widget")
        st.markdown("Every 20 minutes of screen usage, look at an object 20 feet away for 20 seconds.")
        
        pomo_state = st.session_state.get('pomo_active', False)
        
        if not pomo_state:
            if st.button("▶️ Start 20-Minute Focus Session", use_container_width=True):
                st.session_state['pomo_active'] = True
                st.session_state['pomo_start'] = time.time()
                st.rerun()
        else:
            elapsed = time.time() - st.session_state.get('pomo_start', time.time())
            remaining = max(0, int(1200 - elapsed))
            mins, secs = divmod(remaining, 60)
            
            st.metric("Session Time Remaining", f"{mins:02d}:{secs:02d}")
            st.progress(min(1.0, elapsed / 1200.0))
            
            if remaining == 0:
                st.balloons()
                st.warning("🔔 **20 Minutes Completed!** Look away at an object 20 feet away for 20 seconds now!")
                
            if st.button("⏹️ Reset Timer"):
                st.session_state['pomo_active'] = False
                st.rerun()

    with col_budget:
        st.subheader("💡 Daily Gaming Time Budget Calculator")
        target_gpa = st.slider("Target Academic GPA", 2.0, 4.0, 3.5, step=0.1)
        needed_sleep = st.slider("Required Sleep Hours", 6.0, 9.0, 7.5, step=0.5)
        study_hours = st.slider("Daily Study & Lecture Hours", 2.0, 10.0, 6.0, step=0.5)
        
        free_hours = max(0.0, 24.0 - needed_sleep - study_hours - 2.5) # 2.5h for meals/hygiene
        rec_gaming = min(free_hours, 2.0 if target_gpa >= 3.2 else 1.0)
        
        st.metric("Recommended Max Gaming Budget", f"{rec_gaming:.1f} Hours/Day")
        st.caption(f"Calculated budget leaves {free_hours - rec_gaming:.1f} hours for social & physical activities.")

    st.divider()
    st.subheader("📖 Personalized Ergonomic Guidelines")
    st.markdown("""
    - 💡 **Lighting Calibration:** Maintain ambient lighting equal to or brighter than screen luminescence to prevent pupillary strain.
    - 🖥️ **Screen Distance:** Position monitor 50–70 cm (arm's length) away with the top of the screen at or slightly below eye level.
    - 🌙 **Circadian Protection:** Enable blue-light filter mode after 20:00 to protect melatonin synthesis.
    """)
    st.info("Want to cut down on gaming itself? Open **🛡️ Gaming Reduction Program** for a guided, research-based plan.")


# ════════════════════════════════════════════════════════════
#  PAGE 5b: GAMING REDUCTION PROGRAM (CBT-BASED SELF-HELP)
# ════════════════════════════════════════════════════════════
elif page == "🛡️ Gaming Reduction Program":
    reduction_program.render_program_page()


# ════════════════════════════════════════════════════════════
#  PAGE 6: MODEL PERFORMANCE & INSIGHTS
# ════════════════════════════════════════════════════════════
elif page == "📊 Model Performance & Insights":
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
        st.subheader("Eye Fatigue Classifiers (MRL Eye Dataset, Fusek 2018)")
        tab_resnet, tab_cnn = st.tabs(["ResNet50 (deployed)", "Baseline CNN"])
        for tab, files in [(tab_resnet, ['resnet_eye_training_history.png', 'resnet_eye_confusion_matrix.png']),
                           (tab_cnn, ['eye1_training_history.png', 'eye2_confusion_matrix.png'])]:
            with tab:
                for sub, fname in zip(st.tabs(["Training History", "Confusion Matrix"]), files):
                    path = os.path.join(BASE, 'results', fname)
                    with sub:
                        if os.path.exists(path):
                            st.image(path, use_container_width=True)
                        else:
                            st.warning(f"{fname} not found.")

    with t_perf3:
        st.subheader("Confidence-Weighted Late Fusion")
        st.markdown(f"""
| Vote | Source | Maximum influence |
|---|---|---|
| Behaviour | {model_name}, trained on real IGDS9-SF data | its own confidence (max probability) |
| Cognitive | Go/No-Go commission errors — literature prior | capped at r = {fe.COGNITIVE_LITERATURE_WEIGHT:.2f} (d = {fe.INHIBITION_EFFECT_D}, Argyriou et al., 2017) |
| Eye fatigue | ResNet50 + OpenCV score — literature prior | capped at {fe.EYE_FATIGUE_LITERATURE_WEIGHT} (Liu & Zhou, 2024) |
""")
        st.info("Whether fusion improves on the behaviour model alone can only be tested on data where the same students "
                "have the questionnaire, cognitive tests and eye check. That evaluation comes from the "
                "**📝 Validation Study** (`analyze_study.py`). The earlier 76.3% → 79.3% fusion result was computed on "
                "synthetic data and has been withdrawn.")

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

# ════════════════════════════════════════════════════════════
#  PAGE 7: STUDENT HISTORY
# ════════════════════════════════════════════════════════════
elif page == "📋 Student History":
    st.title("📋 Session Assessment History")
    if 'history' not in st.session_state or not st.session_state['history']:
        st.info("No predictions executed yet in this session. Return to **🏠 Home & Multimodal Prediction**.")
    else:
        df_h = pd.DataFrame(st.session_state['history'])
        st.dataframe(df_h, use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear History"):
            st.session_state['history'] = []
            st.rerun()


# ════════════════════════════════════════════════════════════
#  PAGE 8: VALIDATION STUDY (FULL-SYSTEM TESTING WITH PARTICIPANTS)
# ════════════════════════════════════════════════════════════
elif page == "📝 Validation Study":
    def reset_participant():
        # Wipe every answer, score and widget value so the next participant starts clean
        st.session_state.clear()
        st.session_state['nav_page'] = "📝 Validation Study"

    study_mode.render_study_page(reset_participant)
    st.divider()
    with st.expander("🔒 Researcher tools"):
        st.markdown("Analyse the collected sessions with `python analyze_study.py` — it writes the tables and charts "
                    "for the Results chapter to `results/study/`.")
        study_mode.render_export()

# ── Footer ───────────────────────────────────────────────────
st.sidebar.divider()
if os.environ.get('APP_LAN_URL'):
    st.sidebar.info(f"📱 **Open on a phone** (same Wi-Fi):\n\n`{os.environ['APP_LAN_URL']}`")
st.sidebar.caption(f"🎓 BSc (Hons) Computer Science | KDU\n{model_name} on real IGDS9-SF data | "
                   f"CV R² {igd_meta['cv']['R2']:.2f}, ρ {igd_meta['cv']['Spearman']:.2f}")
