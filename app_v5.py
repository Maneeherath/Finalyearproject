# ============================================================
#  FINAL WEB APP v5 — Enhanced A+ Research Edition
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
import tensorflow as tf
from tensorflow.keras.models import load_model

# ReportLab imports for PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Import Late Fusion Engine module
import fusion_engine as fe
# Research-based cognitive battery (2-Back, Task-Switching, Mental Rotation, Go/No-Go)
import research_cognitive_tests as rct
# Validation study (IGDS9-SF criterion + SUS usability) and continuous eye-session logs
import study_mode
import eye_sessions

st.set_page_config(
    page_title="Gaming Behavior & Cognitive Predictor (v5 A+)",
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

# ── Load Machine Learning & Clustering Models ─────────────────────────────────
@st.cache_resource
def load_all_models():
    try:
        clf = joblib.load(os.path.join(BASE, 'models/best_model.pkl'))
        features = joblib.load(os.path.join(BASE, 'models/feature_names.pkl'))
        
        # Load the advanced Cognitive ANN (replaces basic Ridge Regression)
        ann_model = load_model(os.path.join(BASE, 'models/ann_cognitive_model.h5'), compile=False)
        scaler  = joblib.load(os.path.join(BASE, 'models/scaler.pkl'))
        
        # Load KMeans models
        kmeans = joblib.load(os.path.join(BASE, 'models/kmeans_model.pkl'))
        kmeans_scaler = joblib.load(os.path.join(BASE, 'models/kmeans_scaler.pkl'))
        kmeans_features = joblib.load(os.path.join(BASE, 'models/kmeans_features.pkl'))
        
        # Load Cognitive Evidence Classifier for Late Fusion
        dataset_path = os.path.join(BASE, 'data/gaming_dataset_v2.csv')
        if os.path.exists(dataset_path):
            df_full = pd.read_csv(dataset_path)
            cog_clf, cog_scaler = fe.train_cognitive_evidence_classifier(df_full)
        else:
            cog_clf, cog_scaler = None, None

        # Load Deep ResNet50 Eye Fatigue Classifier
        resnet_eye_path = os.path.join(BASE, 'models/resnet_eye_fatigue_model.h5')
        if os.path.exists(resnet_eye_path):
            resnet_eye_model = load_model(resnet_eye_path, compile=False)
        else:
            resnet_eye_model = None

        return clf, features, ann_model, scaler, kmeans, kmeans_scaler, kmeans_features, cog_clf, cog_scaler, resnet_eye_model
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure all trained models are present in models/")
        return None, None, None, None, None, None, None, None, None, None

(clf_model, feature_names, ann_model, scaler, 
 kmeans_model, kmeans_scaler, kmeans_features, 
 cog_clf, cog_scaler, resnet_eye_model) = load_all_models()



def get_model_display_name(model):
    if model is None:
        return "Tabular Model"
    name = type(model).__name__
    if name == 'StackingClassifier':
        return 'Stacking Ensemble'
    elif name == 'VotingClassifier':
        return 'Voting Ensemble'
    elif name == 'RandomForestClassifier':
        return 'Random Forest'
    elif name == 'XGBClassifier':
        return 'XGBoost'
    elif name == 'LGBMClassifier':
        return 'LightGBM'
    elif name == 'GradientBoostingClassifier':
        return 'Gradient Boosting'
    return name


model_name = get_model_display_name(clf_model)



# ── PDF Generator (v5 Upgraded ReportLab) ─────────────────────────────────────
def generate_pdf_report_v5(results_dict):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=36, leftMargin=36,
                            topMargin=36, bottomMargin=36)
    story = []
    styles = getSampleStyleSheet()
    
    primary_color = colors.HexColor("#1A365D")
    secondary_color = colors.HexColor("#0D9488")
    accent_color = colors.HexColor("#C53030")
    text_color = colors.HexColor("#2D3748")
    
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'],
        fontName='Helvetica-Bold', fontSize=20, leading=24,
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
    bold_body_style = ParagraphStyle(
        'DocBodyBold', parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    # Title Header & Institutional Branding
    story.append(Paragraph("🎮 MULTIMODAL GAMING BEHAVIOR & COGNITIVE ASSESSMENT", title_style))
    story.append(Paragraph("<b>Faculty of Computing | General Sir John Kotelawala Defence University (KDU)</b>", body_style))
    story.append(Paragraph("<b>Student Author:</b> P R M K Herath | D/BCS/23/0009 | BSc (Hons) Computer Science", body_style))
    story.append(Paragraph(f"<b>Assessment Timestamp:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    story.append(Spacer(1, 8))
    
    # 1. Student Profile
    story.append(Paragraph("1. Student Profile & Gaming Demographics", h1_style))
    info_data = [
        [Paragraph("<b>Age:</b>", body_style), Paragraph(str(results_dict['age']), body_style),
         Paragraph("<b>Gender:</b>", body_style), Paragraph(results_dict['gender'], body_style)],
        [Paragraph("<b>Year of Study:</b>", body_style), Paragraph(str(results_dict['year_of_study']), body_style),
         Paragraph("<b>Degree Program:</b>", body_style), Paragraph(results_dict['degree'], body_style)],
        [Paragraph("<b>Weekly Gaming Hours:</b>", body_style), Paragraph(f"{results_dict['hours_wk']} hrs/wk", body_style),
         Paragraph("<b>Current GPA:</b>", body_style), Paragraph(f"{results_dict['gpa']:.2f}", body_style)],
    ]
    t_info = Table(info_data, colWidths=[110, 140, 110, 140])
    t_info.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_info)
    story.append(Spacer(1, 8))
    
    # 2. Multimodal Late Fusion Results
    story.append(Paragraph("2. Confidence-Weighted Multimodal Late Fusion Assessment", h1_style))
    fusion_data = [
        [Paragraph("<b>Assessment Metric</b>", bold_body_style), Paragraph("<b>Evaluated Value / Class</b>", bold_body_style), Paragraph("<b>Modal Confidence / Weight</b>", bold_body_style)],
        [Paragraph(f"Tabular {results_dict.get('model_name', 'RF')} Prediction", body_style), Paragraph(f"<b>{results_dict['rf_pred']}</b>", body_style), Paragraph(f"Conf: {results_dict['rf_conf']:.2f} (Weight: {results_dict['rf_w']:.2f})", body_style)],
        [Paragraph("Cognitive Evidence Vote (ANN)", body_style), Paragraph(f"<b>{results_dict['cog_pred']}</b>", body_style), Paragraph(f"Conf: {results_dict['cog_conf']:.2f} (Weight: {results_dict['cog_w']:.2f})", body_style)],
        [Paragraph("Visual Eye Fatigue Prior (CNN)", body_style), Paragraph(f"<b>{results_dict['eye_pred']}</b>", body_style), Paragraph(f"Conf: {results_dict['eye_conf']:.2f} (Weight: {results_dict['eye_w']:.2f})", body_style)],
        [Paragraph("🏆 <b>Fused Multimodal Output</b>", bold_body_style), Paragraph(f"<b>{results_dict['risk_level']}</b>", bold_body_style), Paragraph("Log-Linear Opinion Pooling", bold_body_style)],
        [Paragraph("Composite Health Score", body_style), Paragraph(f"<b>{results_dict['health_score']}/100</b>", bold_body_style), Paragraph(f"Grade: {results_dict['health_grade']}", body_style)],
    ]
    t_fusion = Table(fusion_data, colWidths=[170, 160, 170])
    t_fusion.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('BACKGROUND', (0,4), (-1,4), colors.HexColor("#FEF3C7")),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_fusion)
    story.append(Spacer(1, 8))
    
    # 3. Gamer Archetype
    story.append(Paragraph("3. K-Means Gamer Archetype Profile", h1_style))
    story.append(Paragraph(f"<b>Archetype Assignment:</b> {results_dict['archetype']}", bold_body_style))
    story.append(Paragraph(results_dict['archetype_desc'], body_style))
    story.append(Spacer(1, 8))
    
    # 4. Measured Biometrics & Real-Time Tests
    story.append(Paragraph("4. Measured Biometrics & Cognitive Performance", h1_style))
    bio_data = [
        [Paragraph("<b>Biometric Metric</b>", bold_body_style), Paragraph("<b>Measured Value</b>", bold_body_style), Paragraph("<b>Clinical Assessment</b>", bold_body_style)],
        [Paragraph("Go Reaction Time (ms)", body_style), Paragraph(f"{results_dict['rt_measured']} ms" if results_dict['rt_measured'] else "Estimated Baseline", body_style), Paragraph(results_dict['rt_eval'], body_style)],
        [Paragraph("Working-Memory Score", body_style), Paragraph(f"{results_dict['mem_measured']}%" if results_dict['mem_measured'] is not None else "Estimated Baseline", body_style), Paragraph(results_dict['mem_eval'], body_style)],
        [Paragraph("Attention / Executive Control Score", body_style), Paragraph(f"{results_dict['att_measured']}%" if results_dict['att_measured'] is not None else "Estimated Baseline", body_style), Paragraph(results_dict['att_eval'], body_style)],
        [Paragraph("Eye Fatigue Index", body_style), Paragraph(f"{results_dict['eye_fatigue']}/100" if results_dict['eye_fatigue'] != 'N/A' else "Not Captured", body_style), Paragraph(str(results_dict['eye_level']), body_style)],
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
    t_bio.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('PADDING', (0,0), (-1,-1), 4),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
    ]))
    story.append(t_bio)
    story.append(Spacer(1, 8))
    
    # 5. Recovery & Action Plan
    story.append(Paragraph("5. Cognitive Recovery & Ergonomic Detox Action Plan", h1_style))
    story.append(Paragraph(f"<b>Estimated Cognitive Recovery Period:</b> {results_dict['recovery_days']} Days (Citing Kappi et al., 2024)", bold_body_style))
    for step in results_dict['recovery_plan']:
        story.append(Paragraph(f"• {step}", body_style))
    story.append(Spacer(1, 12))
    
    # Academic Sign-off Block
    story.append(Paragraph("<b>Academic Verification & Advisor Endorsement:</b>", bold_body_style))
    story.append(Spacer(1, 15))
    sign_data = [
        [Paragraph("_____________________________<br/><b>Student Signature</b>", body_style),
         Paragraph("_____________________________<br/><b>Project Supervisor Signature</b>", body_style)]
    ]
    t_sign = Table(sign_data, colWidths=[250, 250])
    story.append(t_sign)
        
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
    st.markdown("**Final Year Research Project | BSc (Hons) Computer Science | KDU | v5 Research Edition**")

    # High-level Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{model_name} CV Acc", "83.00%", "5-Fold CV")
    m2.metric("Multimodal Late Fusion", "Log-Linear Pooling", "Confidence Weighted")
    m3.metric("K-Means Archetypes", "4 Gamer Clusters", "Learned Profile")
    m4.metric("Novelty Features", "6 Integrated", "v5 Update")

    # Status Banner for Loaded Modalities
    has_cognitive = 'cognitive_results' in st.session_state
    has_eye = 'eye_fatigue_score' in st.session_state

    if has_cognitive or has_eye:
        loaded_str = []
        if has_cognitive: loaded_str.append("Cognitive Tests")
        if has_eye: loaded_str.append("Eye Fatigue CNN")
        st.success(f"✅ **Enhanced Multimodal Mode Active:** Loaded [{', '.join(loaded_str)}]")
    else:
        st.info("💡 **Tip:** Complete **🧠 Cognitive & Biometric Tests** and **👁️ Eye Fatigue Analysis** to run the complete 3-modality confidence fusion.")

    st.divider()
    st.subheader("📋 Student Profile & Gaming Inputs")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Demographics**")
        age = st.slider("Age", 17, 30, 21)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        year_of_study = st.selectbox("Year of Study", [1, 2, 3, 4])
        degree = st.selectbox("Degree Program", ["CS", "IT", "Engineering", "Business", "Medicine"])

    with col2:
        st.markdown("**🎮 Gaming Profile**")
        years_gaming = st.slider("Years Gaming", 0, 15, 3)
        sessions_per_week = st.slider("Sessions per Week", 1, 14, 4)
        session_hours = st.slider("Hours per Session", 0.5, 8.0, 2.0, step=0.5)
        game_type = st.selectbox("Primary Game Type", ["Action", "Strategy", "RPG", "Sports", "Casual"])

    with col3:
        st.markdown("**🌙 Academic & Sleep Profile**")
        night_playing = st.radio("Play Late at Night?", ["No", "Yes"])
        social_gaming = st.radio("Play with Others Online?", ["No", "Yes"])
        sleep_hours = st.slider("Sleep Hours per Night", 3.0, 10.0, 7.0, step=0.5)
        gpa = st.slider("Current GPA", 1.5, 4.0, 3.2, step=0.1)

    st.divider()
    
    if st.button("🚀 Perform Multimodal System Assessment", use_container_width=True, type="primary"):
        total_hours = round(session_hours * sessions_per_week, 1)

        def scale(v, lo, hi):
            return float(np.clip((v-lo)/(hi-lo), 0, 1))

        # Prepare Tabular Model Input
        input_data = {
            'age':               scale(age, 18, 28),
            'gender':            {"Female": 0, "Male": 1, "Other": 2}[gender],
            'year_of_study':     scale(year_of_study, 1, 4),
            'degree_program':    {"Business": 0, "CS": 1, "Engineering": 2, "IT": 3, "Medicine": 4}[degree],
            'years_gaming':      scale(years_gaming, 0, 12),
            'sessions_per_week': scale(sessions_per_week, 1, 7),
            'session_hours':     scale(session_hours, 0.5, 8.0),
            'total_hours_week':  scale(total_hours, 0.5, 56.0),
            'night_playing':     1 if night_playing == "Yes" else 0,
            'sleep_hours':       scale(sleep_hours, 3.0, 10.0),
            'social_gaming':     1 if social_gaming == "Yes" else 0,
            'gpa':               scale(gpa, 1.5, 4.0),
            'game_Action':       1 if game_type == "Action"   else 0,
            'game_Casual':       1 if game_type == "Casual"   else 0,
            'game_RPG':          1 if game_type == "RPG"      else 0,
            'game_Sports':       1 if game_type == "Sports"   else 0,
            'game_Strategy':     1 if game_type == "Strategy" else 0,
        }
        
        input_df = pd.DataFrame([input_data])[feature_names]
        
        # 1. Tabular RF Prediction & Probability Vector
        rf_pred_idx = clf_model.predict(input_df)[0]
        rf_proba = clf_model.predict_proba(input_df)[0]
        rf_proba_aligned = fe._align_to_class_order(clf_model, rf_proba)
        rf_conf = float(np.max(rf_proba_aligned))

        # Artificial Neural Network (ANN) Cognitive Predictions
        ann_preds = ann_model.predict(input_df, verbose=0)[0]
        att_raw = ann_preds[0]
        mem_raw = ann_preds[1]
        
        # Unscale back to raw percentages using the same min-max baseline constants
        att_pred = round(att_raw * (87.95 - 32.56) + 32.56, 1)
        mem_pred = round(mem_raw * (90.77 - 44.77) + 44.77, 1)

        # Actual biometrics if performed
        if has_cognitive:
            cr = st.session_state['cognitive_results']
            att_pred = round(cr['attention_score_pct'] * 0.87 + 10, 1)
            mem_pred = round(cr['memory_score_pct'] * 0.90 + 8, 1)
            rt_actual = cr['reaction_time_ms']
        else:
            rt_actual = 345 # Baseline default

        # 2. Cognitive Vote (ANN secondary classifier)
        if cog_clf is not None:
            cog_proba, cog_conf = fe.cognitive_vote(cog_clf, cog_scaler, att_pred, mem_pred)
        else:
            cog_proba, cog_conf = np.array([0.33, 0.34, 0.33]), 0.1

        # 3. Eye Fatigue Vote (CNN Prior)
        if has_eye:
            fatigue_val = st.session_state['eye_fatigue_score']
        else:
            fatigue_val = 15.0 # Baseline low fatigue
        eye_proba, eye_conf = fe.eye_fatigue_vote(fatigue_val)

        # ── NOVELTY #1: Perform Confidence-Weighted Late Fusion ───────────────
        votes = [
            (rf_proba_aligned, rf_conf),
            (cog_proba, cog_conf),
            (eye_proba, eye_conf)
        ]
        fused_proba, weights_dict = fe.fuse(votes)
        fused_pred_idx = int(np.argmax(fused_proba))
        
        risk_labels = ["Low Risk", "Medium Risk", "High Risk"]
        fused_risk_label = risk_labels[fused_pred_idx]
        rf_risk_label = risk_labels[rf_pred_idx]
        cog_risk_label = risk_labels[int(np.argmax(cog_proba))]
        eye_risk_label = risk_labels[int(np.argmax(eye_proba))]

        # Composite Health Score
        base_score = 50.0
        hours_penalty = min(35.0, total_hours * 0.8)
        base_score -= hours_penalty
        if sleep_hours >= 7.5: base_score += 15.0
        elif sleep_hours >= 6.0: base_score += 5.0
        else: base_score -= 15.0
        
        gpa_factor = (gpa - 2.0) * 10
        base_score += max(-10.0, min(15.0, gpa_factor))
        
        cog_score_avg = (att_pred + mem_pred) / 2.0
        base_score += (cog_score_avg - 50) * 0.3
        
        if has_eye:
            base_score -= (st.session_state['eye_fatigue_score'] * 0.15)
            
        health_score = int(np.clip(base_score + 10, 0, 100))
        
        if health_score >= 90:   health_grade, health_color = "A+ (Excellent Balance)", "green"
        elif health_score >= 80: health_grade, health_color = "A (Good Status)", "green"
        elif health_score >= 70: health_grade, health_color = "B (Satisfactory)", "orange"
        elif health_score >= 60: health_grade, health_color = "C (Borderline Strain)", "orange"
        elif health_score >= 50: health_grade, health_color = "D (At Risk)", "red"
        else:                    health_grade, health_color = "F (High Addiction Hazard)", "red"

        # K-Means Archetype
        scaled_kmeans_vals = kmeans_scaler.transform([[
            total_hours, sleep_hours, gpa, att_pred, mem_pred, rt_actual
        ]])
        cluster_idx = kmeans_model.predict(scaled_kmeans_vals)[0]
        
        archetype_map = {
            0: ("Casual Academic", "Low gaming hours, high sleep, excellent academic performance, and strong cognitive stats.", "🟢"),
            1: ("At-Risk Extreme", "Very high weekly gaming hours, severe sleep deprivation, declining GPA, and elevated cognitive fatigue.", "🔴"),
            2: ("Balanced Strategist", "Moderate gaming hours with healthy sleep, high GPA, and strong attention scores.", "🔵"),
            3: ("Hardcore Competitive", "Moderate-to-high gaming hours, fast reaction times but elevated screen fatigue.", "🟡")
        }
        archetype, archetype_desc, arch_icon = archetype_map[cluster_idx]

        color_map = {0: "green", 1: "orange", 2: "red"}
        icon_map = {0: "🟢", 1: "🟡", 2: "🔴"}
        advice_map = {
            0: "✅ Healthy Gaming Habit! Keep maintaining your current study and lifestyle balance.",
            1: "⚠️ Warning: Moderate Risk. Cap daily sessions to max 1.5 hours and secure 7+ hours of sleep.",
            2: "🚨 Critical: High Addiction Risk! Implement a complete 14-day digital detox immediately."
        }

        # Save to Session State for downstream tabs & PDF generator
        st.session_state['last_assessment'] = {
            'rf_pred': rf_risk_label,
            'rf_conf': rf_conf,
            'rf_w': weights_dict['RF_weight'],
            'cog_pred': cog_risk_label,
            'cog_conf': cog_conf,
            'cog_w': weights_dict['Cognitive_weight'],
            'eye_pred': eye_risk_label,
            'eye_conf': eye_conf,
            'eye_w': weights_dict['EyeFatigue_weight'],
            'fused_pred': fused_risk_label,
            'fused_idx': fused_pred_idx,
            'fused_proba': fused_proba,
            'health_score': health_score,
            'health_grade': health_grade,
            'archetype': archetype,
            'archetype_desc': archetype_desc,
            'att_pred': att_pred,
            'mem_pred': mem_pred,
            'total_hours': total_hours,
            'sleep_hours': sleep_hours,
            'gpa': gpa,
            'rt_actual': rt_actual,
            'fatigue_val': fatigue_val,
            # Extra fields recorded by the validation study (Study Mode)
            'inputs': {
                'age': age, 'gender': gender, 'year_of_study': year_of_study, 'degree': degree,
                'years_gaming': years_gaming, 'sessions_per_week': sessions_per_week,
                'session_hours': session_hours, 'game_type': game_type,
                'night_playing': night_playing, 'social_gaming': social_gaming,
            },
            'rf_proba': [float(p) for p in rf_proba_aligned],
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
            
            # Health Score Display
            st.markdown(f"#### Composite Health Score: :{health_color}[**{health_score}/100**] ({health_grade})")
            st.progress(health_score)
            
            # Archetype Display
            st.markdown(f"#### {arch_icon} Gamer Archetype: **{archetype}**")
            st.caption(archetype_desc)

        with col_res2:
            st.markdown("#### 🔬 Modality Weight & Confidence Breakdown")
            
            # Table comparing Modalities
            df_fusion_vis = pd.DataFrame({
                'Modality': [f'1. Tabular {model_name}', '2. Cognitive Evidence (ANN)', '3. Eye Fatigue Prior (CNN)', '🏆 Fused Consensus'],
                'Predicted Vote': [rf_risk_label, cog_risk_label, eye_risk_label, fused_risk_label],
                'Confidence': [f"{rf_conf*100:.1f}%", f"{cog_conf*100:.1f}%", f"{eye_conf*100:.1f}%", f"{float(np.max(fused_proba))*100:.1f}%"],
                'Fusion Weight': [f"{weights_dict['RF_weight']*100:.1f}%", f"{weights_dict['Cognitive_weight']*100:.1f}%", f"{weights_dict['EyeFatigue_weight']*100:.1f}%", "100.0%"]
            })
            st.dataframe(df_fusion_vis, use_container_width=True, hide_index=True)
            
            if rf_pred_idx != fused_pred_idx:
                st.warning(f"⚡ **Late Fusion Effect:** Modality confidence re-weighted the final output from **{rf_risk_label}** to **{fused_risk_label}**!")

        st.divider()

        # SHAP Proxy Waterfall Chart
        col_shap1, col_shap2 = st.columns([1.2, 0.8])
        with col_shap1:
            st.subheader("🧠 Local Feature Explanations (SHAP Proxy)")
            contribs = {
                "Gaming Hrs/Wk": (total_hours - 12) * 2.0,
                "Sleep Hours": (6.5 - sleep_hours) * 5.0,
                "Night Gaming": 12.0 if night_playing == "Yes" else -5.0,
                "Current GPA": (3.0 - gpa) * 15.0,
                "Cognitive Index": (60.0 - cog_score_avg) * 0.5,
                "Eye Fatigue": (fatigue_val - 20) * 0.3 if has_eye else 0.0
            }
            fig_shap, ax_shap = plt.subplots(figsize=(6, 3))
            f_names = list(contribs.keys())
            f_vals = list(contribs.values())
            c_colors = ['#e74c3c' if x >= 0 else '#3498db' for x in f_vals]
            
            y_pos = np.arange(len(f_names))
            ax_shap.barh(y_pos, f_vals, color=c_colors, align='center', height=0.55)
            ax_shap.set_yticks(y_pos)
            ax_shap.set_yticklabels(f_names, fontsize=8)
            ax_shap.axvline(0, color='grey', linewidth=0.8, linestyle='--')
            ax_shap.set_xlabel('Contribution to Risk Score', fontsize=8)
            ax_shap.set_title('Feature Contribution Matrix', fontsize=9, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close()
            
        with col_shap2:
            st.subheader("📈 12-Month Risk Trajectory")
            months = [0, 3, 6, 12]
            p_high = fused_proba[2]
            if total_hours > 15 or night_playing == "Yes" or sleep_hours < 6.0:
                base_traj = [p_high + (1.0 - p_high) * (1.0 - np.exp(-0.06 * m)) for m in months]
            else:
                base_traj = [p_high for _ in months]
            rec_traj = [p_high * np.exp(-0.25 * m) for m in months]
            
            fig_traj, ax_traj = plt.subplots(figsize=(5, 3))
            ax_traj.plot(months, [p*100 for p in base_traj], label='Current Habits', color='#e74c3c', marker='o', lw=2)
            ax_traj.plot(months, [p*100 for p in rec_traj], label='Detox Plan', color='#2ecc71', marker='s', lw=2)
            ax_traj.set_ylim(-5, 105)
            ax_traj.set_ylabel('High Risk Prob (%)', fontsize=8)
            ax_traj.set_xlabel('Months', fontsize=8)
            ax_traj.legend(fontsize=7)
            ax_traj.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_traj)
            plt.close()

        # Recovery Estimator & PDF Generator Button
        st.divider()
        st.subheader("⏳ Cognitive Recovery & Assessment PDF Export")
        
        excess_hours = max(0, total_hours - 10)
        sleep_deficit = max(0, 7.5 - sleep_hours)
        fatigue_impact = fatigue_val * 0.1
        recovery_days = int(np.clip((excess_hours * 0.8) + (sleep_deficit * 2.0) + fatigue_impact, 3, 30))
        
        recovery_plan = [
            f"Days 1-3: Absolute digital fast for gaming apps; target sleep > 8.0 hrs/night.",
            f"Days 4-7: Apply blue-light filters; restrict screen usage within 1.5h of sleep.",
            f"Days 8-{recovery_days}: Reintroduce gaming ONLY after coursework is complete (max 60 mins/day)."
        ]

        pdf_payload = {
            'model_name': model_name,
            'age': age,
            'gender': gender,
            'year_of_study': year_of_study,
            'degree': degree,
            'hours_wk': total_hours,
            'gpa': gpa,
            'rf_pred': rf_risk_label,
            'rf_conf': rf_conf,
            'rf_w': weights_dict['RF_weight'],
            'cog_pred': cog_risk_label,
            'cog_conf': cog_conf,
            'cog_w': weights_dict['Cognitive_weight'],
            'eye_pred': eye_risk_label,
            'eye_conf': eye_conf,
            'eye_w': weights_dict['EyeFatigue_weight'],
            'risk_level': fused_risk_label,
            'health_score': health_score,
            'health_grade': health_grade,
            'archetype': archetype,
            'archetype_desc': archetype_desc,
            'rt_measured': rt_actual if has_cognitive else None,
            'rt_eval': "Good" if rt_actual < 280 else "Normal" if rt_actual < 360 else "Fatigued",
            'mem_measured': st.session_state.get('mem_score', None) if has_cognitive else None,
            'mem_eval': "Excellent" if st.session_state.get('mem_score', 0) >= 80 else "Average" if st.session_state.get('mem_score', 0) >= 50 else "Attention Deficit",
            'att_measured': st.session_state.get('att_score', None) if has_cognitive else None,
            'att_eval': "Normal" if st.session_state.get('att_score', 0) >= 60 else "Deficit Indicator",
            'eye_fatigue': st.session_state.get('eye_fatigue_score', 'N/A'),
            'eye_level': st.session_state.get('eye_fatigue_level', 'N/A'),
            'battery': st.session_state['cognitive_results'].get('battery', {}) if has_cognitive else {},
            'recovery_days': recovery_days,
            'recovery_plan': recovery_plan
        }
        
        pdf_bytes = generate_pdf_report_v5(pdf_payload)
        
        st.download_button(
            label="📥 Download Signed Multimodal Assessment Report (PDF v5)",
            data=pdf_bytes,
            file_name=f"Multimodal_Assessment_Report_{time.strftime('%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        # Store in session history
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        st.session_state['history'].append({
            'Student':      f"#{len(st.session_state['history'])+1}",
            'Hours/Wk':     total_hours,
            'Sleep':        sleep_hours,
            'GPA':          gpa,
            'RF Vote':      rf_risk_label,
            'Fused Risk':   fused_risk_label,
            'Health Score': health_score,
            'Archetype':    archetype
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
    st.title("🧪 What-If Sensitivity Simulator & Multidimensional Radar Profiling")
    
    tab_sim, tab_radar = st.tabs(["🧪 Interactive 'What-If' Habit Simulator", "🕸️ Multidimensional Gamer Radar Chart"])
    
    # ── NOVELTY #3: WHAT-IF SENSITIVITY SIMULATOR ─────────────────────────────
    with tab_sim:
        st.subheader("🧪 'What-If' Interactive Habit Sensitivity Laboratory")
        st.markdown("""
        Adjust your daily habits below to instantly simulate how lifestyle improvements (e.g. extra sleep, reduced gaming hours, higher GPA) 
        will change your **Addiction Risk %**, **Health Score**, **12-Month Trajectory**, and **Cognitive Recovery Days**.
        """)
        
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            st.markdown("**🎮 Baseline Parameters vs Tweak**")
            sim_gaming_hrs = st.slider("Simulated Gaming Hours/Week", 0.0, 56.0, 14.0, step=1.0)
            sim_sleep_hrs  = st.slider("Simulated Sleep Hours/Night", 3.0, 10.0, 7.5, step=0.5)
        with sim_col2:
            st.markdown("**🎓 Academic & Cognitive Target**")
            sim_gpa = st.slider("Simulated GPA Target", 2.0, 4.0, 3.5, step=0.1)
            sim_att = st.slider("Simulated Attention Target (%)", 30, 100, 75, step=5)
            
        # Re-compute Simulated Metrics
        base_h = 50.0 - min(35.0, sim_gaming_hrs * 0.8)
        if sim_sleep_hrs >= 7.5: base_h += 15.0
        elif sim_sleep_hrs >= 6.0: base_h += 5.0
        else: base_h -= 15.0
        base_h += max(-10.0, min(15.0, (sim_gpa - 2.0) * 10))
        base_h += (sim_att - 50) * 0.3
        sim_health_score = int(np.clip(base_h + 10, 0, 100))
        
        # Risk Reduction Calculation
        sim_high_risk_prob = max(5.0, min(95.0, (sim_gaming_hrs * 1.8) - (sim_sleep_hrs * 4.0) - (sim_gpa * 5.0) + 30.0))
        sim_rec_days = int(np.clip(max(0, sim_gaming_hrs - 10)*0.8 + max(0, 7.5 - sim_sleep_hrs)*2.0, 2, 30))
        
        st.divider()
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Simulated Health Score", f"{sim_health_score}/100", f"{sim_health_score-50:+d} pts vs baseline")
        mc2.metric("High Risk Probability", f"{sim_high_risk_prob:.1f}%", f"Target Shift")
        mc3.metric("Cognitive Recovery Time", f"{sim_rec_days} Days", f"Detox Period")
        mc4.metric("Academic Readiness", "High" if sim_gpa >= 3.2 else "Moderate", f"GPA {sim_gpa}")

        # Simulated Trajectory Comparison Chart
        st.markdown("#### 📉 Simulated 12-Month Addiction Trajectory Flattening")
        sim_months = [0, 3, 6, 12]
        sim_base_curve = [sim_high_risk_prob * (1 + 0.03*m) for m in sim_months]
        sim_detox_curve = [sim_high_risk_prob * np.exp(-0.2 * m) for m in sim_months]
        
        fig_sim, ax_sim = plt.subplots(figsize=(8, 3))
        ax_sim.plot(sim_months, sim_base_curve, label='Baseline Habit Path', color='#e74c3c', linestyle='--', marker='o')
        ax_sim.plot(sim_months, sim_detox_curve, label='Simulated Optimized Habit Path', color='#2ecc71', linewidth=2.5, marker='s')
        ax_sim.set_ylabel('High Risk Prob (%)', fontsize=9)
        ax_sim.set_xlabel('Months Ahead', fontsize=9)
        ax_sim.set_ylim(-5, 105)
        ax_sim.legend(fontsize=8)
        ax_sim.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_sim)
        plt.close()

    # ── NOVELTY #4: MULTIDIMENSIONAL RADAR CHART ───────────────────────────────
    with tab_radar:
        st.subheader("🕸️ Multidimensional Gamer Persona Radar Profiling")
        st.markdown("Compares your current (or simulated) profile against standard population benchmarks (Healthy Gamer Cohort vs At-Risk Cohort).")

        # Define 6 Axis Labels
        categories = ['Gaming Hours\n(Inverted)', 'Sleep Hours', 'Academic GPA', 'Attention Score', 'Memory Score', 'Reaction Speed\n(Inverted)']
        N = len(categories)
        
        # Pull values from last assessment or fallback
        last = st.session_state.get('last_assessment', {})
        val_gaming = max(0, 100 - (last.get('total_hours', 14.0)/40.0)*100)
        val_sleep  = (last.get('sleep_hours', 7.0)/10.0)*100
        val_gpa    = (last.get('gpa', 3.2)/4.0)*100
        val_att    = last.get('att_pred', 70.0)
        val_mem    = last.get('mem_pred', 72.0)
        val_rt     = max(0, 100 - ((last.get('rt_actual', 345)-200)/300)*100)
        
        user_vals = [val_gaming, val_sleep, val_gpa, val_att, val_mem, val_rt]
        user_vals += user_vals[:1]
        
        # Benchmark Cohort Values
        healthy_vals = [85, 80, 88, 80, 82, 78]
        healthy_vals += healthy_vals[:1]
        
        atrisk_vals = [25, 45, 50, 48, 52, 40]
        atrisk_vals += atrisk_vals[:1]
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        fig_radar, ax_radar = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True))
        
        ax_radar.set_theta_offset(np.pi / 2)
        ax_radar.set_theta_direction(-1)
        
        plt.xticks(angles[:-1], categories, size=8)
        ax_radar.set_rlabel_position(0)
        plt.yticks([25, 50, 75, 100], ["25", "50", "75", "100"], color="grey", size=7)
        plt.ylim(0, 100)
        
        # Plot User Profile
        ax_radar.plot(angles, user_vals, linewidth=2, linestyle='solid', label='Current Student Profile', color='#3498db')
        ax_radar.fill(angles, user_vals, '#3498db', alpha=0.25)
        
        # Plot Healthy Benchmark
        ax_radar.plot(angles, healthy_vals, linewidth=1.5, linestyle='--', label='Healthy Benchmark Average', color='#2ecc71')
        
        # Plot At-Risk Benchmark
        ax_radar.plot(angles, atrisk_vals, linewidth=1.5, linestyle=':', label='At-Risk Cohort Average', color='#e74c3c')
        
        plt.title('6-Axis Gamer Persona Radar Chart', size=11, weight='bold', y=1.08)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_radar)
        plt.close()


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


# ════════════════════════════════════════════════════════════
#  PAGE 6: MODEL PERFORMANCE & INSIGHTS
# ════════════════════════════════════════════════════════════
elif page == "📊 Model Performance & Insights":
    st.title("📊 Model Performance Dashboard & Dataset Analytics")
    
    t_perf1, t_perf2, t_perf3, t_perf4 = st.tabs([f"{model_name} Evaluation", "👁️ Eye Fatigue Model Evaluation", "Multimodal Fusion Benchmark", "Dataset Analytics"])
    
    with t_perf1:
        st.subheader(f"{model_name} 5-Fold Cross Validation Evaluation")
        tab1, tab2, tab3 = st.tabs(["CV Comparison", "Feature Importances", "CV Score Boxplot"])
        
        for tab, fname, title in zip(
            [tab1, tab2, tab3],
            ['chartA_cross_validation.png', 'chartB_feature_importance.png', 'chartC_cv_boxplot.png'],
            ['5-Fold CV Accuracy', 'Feature Importances', 'CV Distribution']
        ):
            path = os.path.join(BASE, 'results', fname)
            with tab:
                if os.path.exists(path):
                    st.image(path, use_container_width=True)
                else:
                    st.warning("Chart image generated during model training.")

    with t_perf2:
        st.subheader("Visual Eye Fatigue Classifier Evaluation")
        tab_resnet, tab_cnn = st.tabs(["🏆 Upgraded ResNet50 Model", "Baseline CNN Model"])
        
        with tab_resnet:
            sub_tab1, sub_tab2 = st.tabs(["Training History", "Confusion Matrix"])
            path_resnet_hist = os.path.join(BASE, 'results', 'resnet_eye_training_history.png')
            path_resnet_cm = os.path.join(BASE, 'results', 'resnet_eye_confusion_matrix.png')
            
            with sub_tab1:
                if os.path.exists(path_resnet_hist):
                    st.image(path_resnet_hist, use_container_width=True)
                else:
                    st.warning("ResNet50 Training History chart not found.")
            with sub_tab2:
                if os.path.exists(path_resnet_cm):
                    st.image(path_resnet_cm, use_container_width=True)
                else:
                    st.warning("ResNet50 Confusion Matrix chart not found.")
                    
        with tab_cnn:
            sub_tab3, sub_tab4 = st.tabs(["Training History", "Confusion Matrix"])
            path_cnn_hist = os.path.join(BASE, 'results', 'eye1_training_history.png')
            path_cnn_cm = os.path.join(BASE, 'results', 'eye2_confusion_matrix.png')
            
            with sub_tab3:
                if os.path.exists(path_cnn_hist):
                    st.image(path_cnn_hist, use_container_width=True)
                else:
                    st.warning("Baseline CNN Training History chart not found.")
            with sub_tab4:
                if os.path.exists(path_cnn_cm):
                    st.image(path_cnn_cm, use_container_width=True)
                else:
                    st.warning("Baseline CNN Confusion Matrix chart not found.")

    with t_perf3:
        st.subheader("Multimodal Late Fusion CV Evaluation Summary")
        fusion_eval_path = os.path.join(BASE, 'fusion_evaluation_results.json')
        if os.path.exists(fusion_eval_path):
            import json
            with open(fusion_eval_path, 'r') as f:
                f_eval = json.load(f)
            st.json(f_eval)
        else:
            st.info("Run `fusion_engine.py` to view raw JSON benchmark results.")

    with t_perf4:
        st.subheader("Dataset Exploratory Insights")
        datapath = os.path.join(BASE, 'data/gaming_dataset_v2.csv')
        if os.path.exists(datapath):
            df_in = pd.read_csv(datapath)
            st.dataframe(df_in.describe(), use_container_width=True)


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
st.sidebar.caption("🎓 BSc (Hons) Computer Science | KDU\nTuned Late Fusion | CV Accuracy: 83.00%")
