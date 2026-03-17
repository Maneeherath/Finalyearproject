# ============================================================
#  STAGE 5: STREAMLIT WEB APPLICATION
#  Project: Computational Modeling of Cognitive Performance
#           and Gaming Behavior
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import streamlit as st
import pandas as pd
import numpy as npdir
import joblib
import os

# ── Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title="Gaming Behavior Predictor",
    page_icon="🎮",
    layout="centered"
)

# ── Load Model ───────────────────────────────────────────────
MODEL_PATH = 'models/best_model.pkl'
FEATURE_PATH = 'models/feature_names.pkl'

@st.cache_resource
def load_model():
    model    = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
    return model, features

model, feature_names = load_model()

# ── Header ───────────────────────────────────────────────────
st.title("🎮 Gaming Behavior & Cognitive Performance Predictor")
st.markdown("**Final Year Project — D/BCS/23/0009 | P R M K Herath**")
st.markdown("Enter your gaming habits below to predict your **addiction risk level** "
            "and understand how gaming may affect your cognitive performance.")
st.divider()

# ── Input Form ───────────────────────────────────────────────
st.subheader("📋 Your Information")

col1, col2 = st.columns(2)

with col1:
    age           = st.slider("Age", 17, 30, 21)
    gender        = st.selectbox("Gender", ["Male", "Female", "Other"])
    year_of_study = st.selectbox("Year of Study", [1, 2, 3, 4])
    degree        = st.selectbox("Degree Program",
                                  ["CS", "IT", "Engineering", "Business", "Medicine"])

with col2:
    years_gaming       = st.slider("Years of Gaming Experience", 0, 15, 3)
    sessions_per_week  = st.slider("Gaming Sessions per Week", 1, 14, 4)
    session_hours      = st.slider("Hours per Gaming Session", 0.5, 8.0, 2.0, step=0.5)
    game_type          = st.selectbox("Primary Game Type",
                                       ["Action", "Strategy", "RPG", "Sports", "Casual"])
    night_playing      = st.radio("Do you play late at night?", ["No", "Yes"])

st.divider()

# ── Predict Button ───────────────────────────────────────────
if st.button("🔍 Predict My Addiction Risk", use_container_width=True, type="primary"):

    # ── Build input row matching training features ───────────
    total_hours_week = round(session_hours * sessions_per_week, 1)

    # Encode categoricals exactly as in preprocessing
    gender_enc = {"Female": 0, "Male": 1, "Other": 2}[gender]
    degree_enc = {"Business": 0, "CS": 1, "Engineering": 2, "IT": 3, "Medicine": 4}[degree]
    night_enc  = 1 if night_playing == "Yes" else 0

    # One-hot encode game type
    game_action   = 1 if game_type == "Action"   else 0
    game_casual   = 1 if game_type == "Casual"   else 0
    game_rpg      = 1 if game_type == "RPG"      else 0
    game_sports   = 1 if game_type == "Sports"   else 0
    game_strategy = 1 if game_type == "Strategy" else 0

    # MinMax scale numeric features (using training dataset ranges)
    def scale(val, min_val, max_val):
        return (val - min_val) / (max_val - min_val)

    input_data = {
        'age':               scale(age, 18, 25),
        'gender':            gender_enc,
        'year_of_study':     scale(year_of_study, 1, 4),
        'degree_program':    degree_enc,
        'years_gaming':      scale(years_gaming, 0, 11),
        'sessions_per_week': scale(sessions_per_week, 1, 7),
        'session_hours':     scale(session_hours, 0.5, 6.0),
        'total_hours_week':  scale(total_hours_week, 0.9, 41.3),
        'night_playing':     night_enc,
        'attention_score':   0.5,   # placeholder (not known at prediction time)
        'memory_score':      0.5,
        'reaction_time':     0.5,
        'game_Action':       game_action,
        'game_Casual':       game_casual,
        'game_RPG':          game_rpg,
        'game_Sports':       game_sports,
        'game_Strategy':     game_strategy,
    }

    input_df = pd.DataFrame([input_data])[feature_names]

    # ── Predict ──────────────────────────────────────────────
    prediction    = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    label_map = {0: "🟢 Low", 1: "🟡 Medium", 2: "🔴 High"}
    color_map = {0: "green",  1: "orange",    2: "red"}
    advice_map = {
        0: "Your gaming habits appear healthy! Keep maintaining balance between gaming and studies.",
        1: "Moderate addiction risk detected. Consider setting daily gaming time limits.",
        2: "High addiction risk detected. We recommend speaking with a counselor and reducing gaming hours significantly."
    }

    result_label = label_map[prediction]
    result_color = color_map[prediction]
    advice       = advice_map[prediction]

    # ── Display Results ──────────────────────────────────────
    st.divider()
    st.subheader("📊 Prediction Results")

    st.markdown(f"### Addiction Risk Level: :{result_color}[{result_label}]")
    st.info(f"💡 {advice}")

    # Probability breakdown
    st.markdown("#### Confidence Breakdown")
    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 Low Risk",    f"{probabilities[0]*100:.1f}%")
    col2.metric("🟡 Medium Risk", f"{probabilities[1]*100:.1f}%")
    col3.metric("🔴 High Risk",   f"{probabilities[2]*100:.1f}%")

    # Gaming summary
    st.divider()
    st.subheader("📈 Your Gaming Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Hours/Week",    f"{total_hours_week} hrs")
    col2.metric("Sessions/Week",       f"{sessions_per_week}")
    col3.metric("Years Gaming",        f"{years_gaming} yrs")

    # Cognitive impact estimate
    st.divider()
    st.subheader("🧠 Estimated Cognitive Impact")

    est_attention = max(20, round(80 - (total_hours_week * 0.6) - (night_enc * 8)))
    est_memory    = max(20, round(78 - (total_hours_week * 0.5) + (3 if game_type == "Strategy" else 0)))

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Estimated Attention Score", f"{est_attention}/100",
                  delta=f"{est_attention - 80} from baseline")
    with col2:
        st.metric("Estimated Memory Score", f"{est_memory}/100",
                  delta=f"{est_memory - 78} from baseline")

    st.caption("⚠️ These are model-based estimates. For accurate cognitive assessment, "
               "please consult a professional.")

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.caption("⚠ Final Year Research Project | BSc(Hons) Computer Science | KDU | Model Accuracy: 84.44%")