# ============================================================
#  ENHANCED WEB APP v2 — Multi-Page with Full Features
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib, os

st.set_page_config(
    page_title="Gaming Behavior Predictor",
    page_icon="🎮",
    layout="wide"
)

# ── Load models ──────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), '..')

@st.cache_resource
def load_models():
    clf      = joblib.load(os.path.join(BASE,'models','best_model.pkl'))
    features = joblib.load(os.path.join(BASE,'models','feature_names.pkl'))
    reg_att  = joblib.load(os.path.join(BASE,'models','reg_attention_score.pkl'))
    reg_mem  = joblib.load(os.path.join(BASE,'models','reg_memory_score.pkl'))
    scaler   = joblib.load(os.path.join(BASE,'models','scaler.pkl'))
    return clf, features, reg_att, reg_mem, scaler

clf_model, feature_names, reg_att, reg_mem, scaler = load_models()

# ── Sidebar Navigation ───────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/controller.png", width=80)
st.sidebar.title("🎮 Gaming Predictor")
st.sidebar.markdown("**D/BCS/23/0009 | P R M K Herath**")
st.sidebar.divider()

page = st.sidebar.radio("Navigate", [
    "🏠 Home & Prediction",
    "🧠 Cognitive Tests",
    "👁️ Eye Fatigue Analysis",
    "📊 Model Performance",
    "📈 Data Insights",
    "📋 Student History"
])

# ════════════════════════════════════════════════════════════
#  PAGE 1: HOME & PREDICTION
# ════════════════════════════════════════════════════════════
if page == "🏠 Home & Prediction":
    st.title("🎮 Gaming Behavior & Cognitive Performance Predictor")
    st.markdown("**Final Year Research Project | BSc (Hons) Computer Science | KDU**")

    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    col_info1.metric("Model Accuracy", "82.00%", "5-Fold CV")
    col_info2.metric("Dataset Size",   "300 Students")
    col_info3.metric("ML Models Tested", "4")
    col_info4.metric("Best Model", "Random Forest")
    st.divider()

    st.subheader("📋 Enter Your Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Personal Details**")
        age           = st.slider("Age", 17, 30, 21)
        gender        = st.selectbox("Gender", ["Male","Female","Other"])
        year_of_study = st.selectbox("Year of Study", [1,2,3,4])
        degree        = st.selectbox("Degree Program",
                            ["CS","IT","Engineering","Business","Medicine"])

    with col2:
        st.markdown("**🎮 Gaming Habits**")
        years_gaming      = st.slider("Years Gaming", 0, 15, 3)
        sessions_per_week = st.slider("Sessions per Week", 1, 14, 4)
        session_hours     = st.slider("Hours per Session", 0.5, 8.0, 2.0, step=0.5)
        game_type         = st.selectbox("Primary Game Type",
                               ["Action","Strategy","RPG","Sports","Casual"])

    with col3:
        st.markdown("**🌙 Lifestyle**")
        night_playing = st.radio("Play Late at Night?", ["No","Yes"])
        social_gaming = st.radio("Play with Others Online?", ["No","Yes"])
        sleep_hours   = st.slider("Sleep Hours per Night", 3.0, 10.0, 7.0, step=0.5)
        gpa           = st.slider("Current GPA", 1.5, 4.0, 3.2, step=0.1)

    st.divider()

    if st.button("🔍 Predict Now", use_container_width=True, type="primary"):
        total_hours_week = round(session_hours * sessions_per_week, 1)

        # Encode
        gender_enc = {"Female":0,"Male":1,"Other":2}[gender]
        degree_enc = {"Business":0,"CS":1,"Engineering":2,"IT":3,"Medicine":4}[degree]
        night_enc  = 1 if night_playing == "Yes" else 0
        social_enc = 1 if social_gaming == "Yes" else 0

        # Scale numeric using saved scaler ranges
        def scale(val, min_val, max_val):
            return float(np.clip((val - min_val) / (max_val - min_val), 0, 1))

        input_data = {
            'age':               scale(age, 18, 28),
            'gender':            gender_enc,
            'year_of_study':     scale(year_of_study, 1, 4),
            'degree_program':    degree_enc,
            'years_gaming':      scale(years_gaming, 0, 12),
            'sessions_per_week': scale(sessions_per_week, 1, 7),
            'session_hours':     scale(session_hours, 0.5, 8.0),
            'total_hours_week':  scale(total_hours_week, 0.5, 56.0),
            'night_playing':     night_enc,
            'sleep_hours':       scale(sleep_hours, 3.0, 10.0),
            'social_gaming':     social_enc,
            'gpa':               scale(gpa, 1.5, 4.0),
            'game_Action':       1 if game_type=="Action"   else 0,
            'game_Casual':       1 if game_type=="Casual"   else 0,
            'game_RPG':          1 if game_type=="RPG"      else 0,
            'game_Sports':       1 if game_type=="Sports"   else 0,
            'game_Strategy':     1 if game_type=="Strategy" else 0,
        }

        input_df   = pd.DataFrame([input_data])[feature_names]
        prediction = clf_model.predict(input_df)[0]
        proba      = clf_model.predict_proba(input_df)[0]

        # Regression predictions
        att_pred_scaled = reg_att.predict(input_df)[0]
        mem_pred_scaled = reg_mem.predict(input_df)[0]
        att_pred = round(att_pred_scaled * (87.95 - 32.56) + 32.56, 1)
        mem_pred = round(mem_pred_scaled * (90.77 - 44.77) + 44.77, 1)

        label_map  = {0:"🟢 Low", 1:"🟡 Medium", 2:"🔴 High"}
        color_map  = {0:"green",  1:"orange",    2:"red"}
        advice_map = {
            0: "✅ Your gaming habits appear healthy! Keep maintaining balance.",
            1: "⚠️ Moderate risk detected. Consider setting daily time limits (max 2hrs/day).",
            2: "🚨 High addiction risk. We strongly recommend reducing gaming hours and seeking support."
        }

        # ── Results ──────────────────────────────────────────
        st.subheader("📊 Your Prediction Results")
        r1, r2 = st.columns([1,2])

        with r1:
            st.markdown(f"### Addiction Risk")
            level_color = color_map[prediction]
            st.markdown(f"## :{level_color}[{label_map[prediction]}]")
            st.info(advice_map[prediction])

        with r2:
            st.markdown("### Confidence Breakdown")
            fig, ax = plt.subplots(figsize=(5, 2.5))
            bars = ax.barh(['Low','Medium','High'],
                           [p*100 for p in proba],
                           color=['#2ecc71','#f39c12','#e74c3c'])
            for bar, val in zip(bars, proba):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{val*100:.1f}%', va='center', fontweight='bold')
            ax.set_xlim(0, 115)
            ax.set_xlabel('Probability (%)')
            ax.set_title('Model Confidence')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.divider()
        st.subheader("🧠 Cognitive Performance Estimates")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Attention Score",    f"{att_pred}/100",
                  delta=f"{att_pred-75:.1f} vs avg")
        c2.metric("Memory Score",       f"{mem_pred}/100",
                  delta=f"{mem_pred-76:.1f} vs avg")
        c3.metric("Total Gaming Hrs/Wk", f"{total_hours_week}")
        c4.metric("Sleep Hours/Night",   f"{sleep_hours}")

        # ── Feature Impact ───────────────────────────────────
        st.divider()
        st.subheader("🔍 What's Driving Your Risk?")
        risk_factors = []
        if total_hours_week > 21: risk_factors.append(("🕒 High weekly gaming hours",    "High", total_hours_week))
        if night_enc == 1:        risk_factors.append(("🌙 Late-night gaming",            "Medium", "Yes"))
        if sleep_hours < 6:       risk_factors.append(("😴 Insufficient sleep",           "High", f"{sleep_hours}hrs"))
        if gpa < 2.5:             risk_factors.append(("📚 Low GPA",                      "Medium", gpa))
        if years_gaming > 8:      risk_factors.append(("🎮 Long gaming history",          "Low", f"{years_gaming}yrs"))

        if risk_factors:
            for factor, severity, value in risk_factors:
                color = "🔴" if severity=="High" else "🟡"
                st.markdown(f"{color} **{factor}** — {value}")
        else:
            st.success("✅ No major risk factors detected in your profile!")

        # Save to session history
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        st.session_state['history'].append({
            'Name': f"Student {len(st.session_state['history'])+1}",
            'Hours/Week': total_hours_week,
            'Night Gaming': night_playing,
            'Sleep': sleep_hours,
            'GPA': gpa,
            'Addiction Risk': label_map[prediction],
            'Attention Est.': att_pred,
            'Memory Est.': mem_pred
        })

# ════════════════════════════════════════════════════════════
#  PAGE 2: MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance Dashboard")
    st.markdown("Detailed evaluation of all 4 machine learning models trained in this project.")

    tab1, tab2, tab3 = st.tabs(["Cross-Validation", "Feature Importance", "CV Distribution"])

    for tab, chart_file, title in zip(
        [tab1, tab2, tab3],
        ['chartA_cross_validation.png','chartB_feature_importance.png','chartC_cv_boxplot.png'],
        ['5-Fold Cross-Validation Comparison','Top 15 Feature Importances','CV Score Distribution']
    ):
        chart_path = os.path.join(BASE, 'results', chart_file)
        with tab:
            st.subheader(title)
            if os.path.exists(chart_path):
                st.image(chart_path, use_container_width=True)
            else:
                st.warning(f"Chart not found. Run 02_model_training_v2.py first.")

    st.divider()
    st.subheader("📋 Model Comparison Summary")
    summary = pd.DataFrame({
        'Model':          ['Decision Tree','Random Forest','Gradient Boosting','SVM'],
        'Test Accuracy':  ['66.67%','80.00%','83.33%','76.67%'],
        'CV Accuracy':    ['67.00%','82.00%','78.00%','68.67%'],
        'CV Std Dev':     ['±3.71%','±5.52%','±5.31%','±2.87%'],
        'Selected':       ['❌','✅ Best','❌','❌']
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.caption("Random Forest selected as best model based on 5-fold cross-validation accuracy.")

# ════════════════════════════════════════════════════════════
#  PAGE 3: DATA INSIGHTS
# ════════════════════════════════════════════════════════════
elif page == "📈 Data Insights":
    st.title("📈 Dataset Insights & Visualizations")

    df_raw = pd.read_csv(os.path.join(BASE,'data','gaming_dataset_v2.csv'))

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", len(df_raw))
    col2.metric("Average Gaming Hrs/Wk", f"{df_raw['total_hours_week'].mean():.1f}")
    col3.metric("Night Gamers", f"{df_raw['night_playing'].mean()*100:.0f}%")

    st.divider()
    tab1, tab2, tab3 = st.tabs(["Addiction Distribution","Gaming vs Cognition","Correlations"])

    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        colors = ['#2ecc71','#f39c12','#e74c3c']
        counts = df_raw['addiction_level'].value_counts().reindex(['Low','Medium','High'])
        axes[0].bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=1.5)
        axes[0].set_title('Addiction Level Distribution')
        axes[0].set_ylabel('Number of Students')
        for i, v in enumerate(counts.values):
            axes[0].text(i, v+1, str(v), ha='center', fontweight='bold')

        axes[1].pie(counts.values, labels=counts.index, colors=colors,
                    autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Proportion of Each Risk Level')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        color_map_raw = {'Low':'#2ecc71','Medium':'#f39c12','High':'#e74c3c'}
        for level, color in color_map_raw.items():
            sub = df_raw[df_raw['addiction_level']==level]
            axes[0].scatter(sub['total_hours_week'], sub['attention_score'],
                           c=color, label=level, alpha=0.6, edgecolors='white', s=40)
            axes[1].scatter(sub['total_hours_week'], sub['memory_score'],
                           c=color, label=level, alpha=0.6, edgecolors='white', s=40)
        for ax, title in zip(axes, ['Hours/Week vs Attention Score','Hours/Week vs Memory Score']):
            ax.set_xlabel('Total Gaming Hours/Week')
            ax.set_ylabel('Score')
            ax.set_title(title)
            ax.legend(title='Risk Level')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab3:
        numeric_cols = ['total_hours_week','sleep_hours','gpa',
                        'attention_score','memory_score','reaction_time']
        corr = df_raw[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        import matplotlib.colors as mcolors
        im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels([c.replace('_',' ') for c in numeric_cols], rotation=45, ha='right')
        ax.set_yticklabels([c.replace('_',' ') for c in numeric_cols])
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                ax.text(j, i, f'{corr.values[i,j]:.2f}', ha='center', va='center', fontsize=9)
        ax.set_title('Feature Correlation Matrix', fontsize=13)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════
#  PAGE 4: STUDENT HISTORY
# ════════════════════════════════════════════════════════════
elif page == "📋 Student History":
    st.title("📋 Session Prediction History")
    st.markdown("All predictions made during this session are tracked here.")

    if 'history' not in st.session_state or not st.session_state['history']:
        st.info("No predictions yet. Go to **🏠 Home & Prediction** and run a prediction first!")
    else:
        history_df = pd.DataFrame(st.session_state['history'])
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        st.divider()
        fig, ax = plt.subplots(figsize=(8, 3))
        risk_counts = history_df['Addiction Risk'].str.extract(r'(Low|Medium|High)')[0].value_counts()
        colors = {'Low':'#2ecc71','Medium':'#f39c12','High':'#e74c3c'}
        ax.bar(risk_counts.index, risk_counts.values,
               color=[colors.get(k,'gray') for k in risk_counts.index])
        ax.set_title('Risk Level Distribution — This Session')
        ax.set_ylabel('Count')
        st.pyplot(fig); plt.close()

        if st.button("🗑️ Clear History"):
            st.session_state['history'] = []
            st.rerun()

# ── Footer ───────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("🎓 BSc (Hons) Computer Science\nKDU | 2026\nModel: Random Forest\nCV Accuracy: 82.00%")