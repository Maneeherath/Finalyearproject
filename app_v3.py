# ============================================================
#  FINAL WEB APP v3 — 6 Pages
#  Added: Cognitive Tests + Eye Fatigue Analysis
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib, os, cv2, time, random
from PIL import Image

st.set_page_config(page_title="Gaming Behavior Predictor",
                   page_icon="🎮", layout="wide")

BASE = os.path.join(os.path.dirname(__file__), '..')

# ── Load models ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    clf = joblib.load('models/best_model.pkl')
    features = joblib.load('models/feature_names.pkl')           # or feature_attention_score.pkl ?
    reg_att = joblib.load('models/reg_attention_score.pkl')
    reg_mem = joblib.load('models/reg_memory_score.pkl')
    scaler  = joblib.load('models/scaler.pkl')
    return clf, features, reg_att, reg_mem, scaler

clf_model, feature_names, reg_att, reg_mem, scaler = load_models()

# ── Sidebar ──────────────────────────────────────────────────
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

# Show if cognitive/eye data is available
if 'cognitive_results' in st.session_state:
    cr = st.session_state['cognitive_results']
    st.sidebar.success(f"🧠 Cognitive data loaded\nRT: {cr['reaction_time_ms']}ms")
if 'eye_fatigue_score' in st.session_state:
    st.sidebar.info(f"👁️ Eye fatigue: {st.session_state['eye_fatigue_score']}/100")

# ════════════════════════════════════════════════════════════
#  PAGE 1: HOME & PREDICTION
# ════════════════════════════════════════════════════════════
if page == "🏠 Home & Prediction":
    st.title("🎮 Gaming Behavior & Cognitive Performance Predictor")
    st.markdown("**Final Year Research Project | BSc (Hons) Computer Science | KDU**")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("CV Accuracy","82.00%","5-Fold")
    c2.metric("Dataset","300 Students")
    c3.metric("Models Tested","4")
    c4.metric("Best Model","Random Forest")

    # Show if enhanced data available
    has_cognitive = 'cognitive_results' in st.session_state
    has_eye       = 'eye_fatigue_score' in st.session_state
    if has_cognitive or has_eye:
        st.success("✅ Enhanced prediction mode: Using your measured cognitive & eye data!")
    else:
        st.info("💡 Tip: Complete **🧠 Cognitive Tests** and **👁️ Eye Fatigue Analysis** first for a more accurate prediction!")

    st.divider()
    st.subheader("📋 Enter Your Gaming Information")
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
        total_hours = round(session_hours * sessions_per_week, 1)

        def scale(v, lo, hi):
            return float(np.clip((v-lo)/(hi-lo), 0, 1))

        input_data = {
            'age':               scale(age,18,28),
            'gender':            {"Female":0,"Male":1,"Other":2}[gender],
            'year_of_study':     scale(year_of_study,1,4),
            'degree_program':    {"Business":0,"CS":1,"Engineering":2,"IT":3,"Medicine":4}[degree],
            'years_gaming':      scale(years_gaming,0,12),
            'sessions_per_week': scale(sessions_per_week,1,7),
            'session_hours':     scale(session_hours,0.5,8.0),
            'total_hours_week':  scale(total_hours,0.5,56.0),
            'night_playing':     1 if night_playing=="Yes" else 0,
            'sleep_hours':       scale(sleep_hours,3.0,10.0),
            'social_gaming':     1 if social_gaming=="Yes" else 0,
            'gpa':               scale(gpa,1.5,4.0),
            'game_Action':       1 if game_type=="Action"   else 0,
            'game_Casual':       1 if game_type=="Casual"   else 0,
            'game_RPG':          1 if game_type=="RPG"      else 0,
            'game_Sports':       1 if game_type=="Sports"   else 0,
            'game_Strategy':     1 if game_type=="Strategy" else 0,
        }
        input_df   = pd.DataFrame([input_data])[feature_names]
        prediction = clf_model.predict(input_df)[0]
        proba      = clf_model.predict_proba(input_df)[0]

        # Regression cognitive predictions
        att_raw = reg_att.predict(input_df)[0]
        mem_raw = reg_mem.predict(input_df)[0]
        att_pred = round(att_raw*(87.95-32.56)+32.56, 1)
        mem_pred = round(mem_raw*(90.77-44.77)+44.77, 1)

        # Override with measured scores if available
        if has_cognitive:
            cr = st.session_state['cognitive_results']
            att_pred = round(cr['attention_score_pct'] * 0.87 + 10, 1)
            mem_pred = round(cr['memory_score_pct'] * 0.90 + 8, 1)
            rt_actual = cr['reaction_time_ms']
        else:
            rt_actual = None

        label_map = {0:"🟢 Low",1:"🟡 Medium",2:"🔴 High"}
        color_map = {0:"green",1:"orange",2:"red"}
        advice_map = {
            0:"✅ Your gaming habits appear healthy! Keep maintaining balance.",
            1:"⚠️ Moderate risk. Consider setting daily time limits (max 2hrs/day).",
            2:"🚨 High addiction risk. Strongly recommend reducing gaming and seeking support."
        }

        st.subheader("📊 Prediction Results")
        r1, r2 = st.columns([1,2])
        with r1:
            lc = color_map[prediction]
            st.markdown(f"### :{lc}[{label_map[prediction]}]")
            st.info(advice_map[prediction])
            if has_eye:
                ef = st.session_state['eye_fatigue_score']
                el = st.session_state['eye_fatigue_level']
                st.markdown(f"**👁️ Eye Fatigue:** {ef}/100 ({el})")
        with r2:
            fig, ax = plt.subplots(figsize=(5,2.5))
            bars = ax.barh(['Low','Medium','High'],
                           [p*100 for p in proba],
                           color=['#2ecc71','#f39c12','#e74c3c'])
            for bar,val in zip(bars,proba):
                ax.text(bar.get_width()+1,bar.get_y()+bar.get_height()/2,
                        f'{val*100:.1f}%',va='center',fontweight='bold')
            ax.set_xlim(0,115)
            ax.set_xlabel('Probability (%)')
            ax.set_title('Model Confidence')
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        st.divider()
        st.subheader("🧠 Cognitive Performance")
        data_source = "🔬 Measured" if has_cognitive else "🤖 Estimated"
        c1,c2,c3,c4 = st.columns(4)
        c1.metric(f"Attention ({data_source})", f"{att_pred}/100")
        c2.metric(f"Memory ({data_source})",    f"{mem_pred}/100")
        c3.metric("Gaming Hrs/Week", f"{total_hours}")
        if rt_actual:
            c4.metric("Reaction Time 🔬", f"{rt_actual} ms")
        else:
            c4.metric("Sleep Hours", f"{sleep_hours}")

        # Risk factors
        st.divider()
        st.subheader("🔍 Risk Factors Detected")
        risks = []
        if total_hours > 21:    risks.append(("🔴","High weekly gaming hours",    f"{total_hours} hrs/wk"))
        if night_playing=="Yes":risks.append(("🟡","Late-night gaming",           "Yes"))
        if sleep_hours < 6:     risks.append(("🔴","Insufficient sleep",          f"{sleep_hours} hrs"))
        if gpa < 2.5:           risks.append(("🟡","Low GPA",                     str(gpa)))
        if has_eye and st.session_state['eye_fatigue_score'] > 50:
            risks.append(("🔴","Eye fatigue detected",
                          f"{st.session_state['eye_fatigue_score']}/100"))
        if has_cognitive and st.session_state['cognitive_results']['reaction_time_ms'] > 400:
            risks.append(("🟡","Slow reaction time",
                          f"{st.session_state['cognitive_results']['reaction_time_ms']} ms"))

        if risks:
            for icon,factor,val in risks:
                st.markdown(f"{icon} **{factor}** — {val}")
        else:
            st.success("✅ No major risk factors in your profile!")

        # Save history
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        st.session_state['history'].append({
            'Student':      f"#{len(st.session_state['history'])+1}",
            'Hours/Wk':     total_hours,
            'Night Gaming': night_playing,
            'Sleep':        sleep_hours,
            'GPA':          gpa,
            'Risk Level':   label_map[prediction],
            'Attention':    att_pred,
            'Memory':       mem_pred,
            'Eye Fatigue':  st.session_state.get('eye_fatigue_score','N/A'),
            'Data Source':  data_source
        })

# ════════════════════════════════════════════════════════════
#  PAGE 2: COGNITIVE TESTS
# ════════════════════════════════════════════════════════════
elif page == "🧠 Cognitive Tests":
    st.title("🧠 Cognitive Assessment Battery")
    st.markdown("Complete all 3 tests. Your **real measured scores** will replace estimated ones in the prediction.")
    st.info("💡 These tests mirror methods used in Kappi et al. (2024) and Farah et al. (2020) — the papers cited in your literature review.")
    st.divider()

    for k,d in [('rt_list',[]),('rt_phase','start'),('rt_start',0),('rt_done',False),
                ('mem_seq',[]),('mem_phase','show'),('mem_score',None),
                ('att_items',[]),('att_target',0),('att_score',None)]:
        if k not in st.session_state:
            st.session_state[k] = d

    # ── TEST 1: REACTION TIME ────────────────────────────────
    with st.expander("⚡ Test 1 — Reaction Time Test", expanded=True):
        st.markdown("Click **Start**, then click the green button **as fast as possible** when it appears. Repeat 3 times.")

        if not st.session_state['rt_done']:
            attempt = len(st.session_state['rt_list']) + 1
            phase   = st.session_state['rt_phase']

            if phase == 'start':
                if st.button("▶️ Start Test", key="rt_go"):
                    st.session_state['rt_phase'] = 'wait'
                    st.rerun()

            elif phase == 'wait':
                st.warning(f"⏳ Attempt {attempt}/3 — Wait for green...")
                time.sleep(random.uniform(1.5, 3.5))
                st.session_state['rt_phase']  = 'click'
                st.session_state['rt_start']  = time.time()
                st.rerun()

            elif phase == 'click':
                st.success("🟢 **CLICK NOW!**")
                if st.button("🟢 CLICK!", key=f"click_{attempt}", type="primary"):
                    ms = round((time.time()-st.session_state['rt_start'])*1000)
                    st.session_state['rt_list'].append(ms)
                    if len(st.session_state['rt_list']) >= 3:
                        st.session_state['rt_done']  = True
                        st.session_state['rt_phase'] = 'done'
                    else:
                        st.session_state['rt_phase'] = 'wait'
                    st.rerun()

        if st.session_state['rt_done']:
            ts  = st.session_state['rt_list']
            avg = round(sum(ts)/len(ts))
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Try 1", f"{ts[0]} ms")
            c2.metric("Try 2", f"{ts[1]} ms")
            c3.metric("Try 3", f"{ts[2]} ms")
            c4.metric("⭐ Avg", f"{avg} ms",
                      "Excellent" if avg<250 else "Average" if avg<350 else "Slow")

    # ── TEST 2: MEMORY ───────────────────────────────────────
    with st.expander("🔢 Test 2 — Memory Sequence Test", expanded=True):
        st.markdown("Memorize the 7-digit sequence, then type it back.")

        if st.session_state['mem_score'] is None:
            if not st.session_state['mem_seq']:
                st.session_state['mem_seq']   = [random.randint(1,9) for _ in range(7)]
                st.session_state['mem_phase'] = 'show'

            seq = st.session_state['mem_seq']
            if st.session_state['mem_phase'] == 'show':
                st.markdown(f"## `{'  '.join(map(str,seq))}`")
                if st.button("✅ I've memorized it", key="mem_done"):
                    st.session_state['mem_phase'] = 'recall'
                    st.rerun()
            else:
                ans = st.text_input("Type the sequence (space separated):",
                                    placeholder="e.g. 3 7 1 9 4 2 8")
                if st.button("Submit", key="mem_sub"):
                    try:
                        nums    = list(map(int, ans.strip().split()))
                        correct = sum(a==b for a,b in zip(nums,seq))
                        st.session_state['mem_score'] = round(correct/len(seq)*100)
                        st.rerun()
                    except:
                        st.error("Enter numbers separated by spaces")

        if st.session_state['mem_score'] is not None:
            s = st.session_state['mem_score']
            st.metric("Memory Score", f"{s}%",
                      "Excellent" if s>=80 else "Average" if s>=50 else "Low")

    # ── TEST 3: ATTENTION ────────────────────────────────────
    with st.expander("👁️ Test 3 — Attention Test", expanded=True):
        st.markdown("Count the 🎯 targets in the grid below.")

        if st.session_state['att_score'] is None:
            if not st.session_state['att_items']:
                t = random.randint(5,12)
                items = ['🎯']*t + ['⭐']*(25-t)
                random.shuffle(items)
                st.session_state['att_items']  = items
                st.session_state['att_target'] = t

            items = st.session_state['att_items']
            for row in [items[i:i+5] for i in range(0,25,5)]:
                st.markdown("  ".join(row))

            n = st.number_input("How many 🎯?", 0, 25, 0)
            if st.button("Submit Count", key="att_sub"):
                correct = st.session_state['att_target']
                diff    = abs(n - correct)
                score   = max(0, round((1 - diff/correct)*100)) if correct > 0 else 0
                st.session_state['att_score'] = score
                st.session_state['att_correct'] = correct
                st.rerun()

        if st.session_state['att_score'] is not None:
            s = st.session_state['att_score']
            st.markdown(f"Correct count: **{st.session_state.get('att_correct',0)}**")
            st.metric("Attention Score", f"{s}%",
                      "Excellent" if s>=80 else "Average" if s>=50 else "Low")

    # ── Save combined results ────────────────────────────────
    st.divider()
    rt_done  = st.session_state['rt_done']
    mem_done = st.session_state['mem_score'] is not None
    att_done = st.session_state['att_score'] is not None
    c1,c2,c3 = st.columns(3)
    c1.metric("⚡ Reaction", "✅" if rt_done  else "⏳")
    c2.metric("🔢 Memory",   "✅" if mem_done else "⏳")
    c3.metric("👁️ Attention","✅" if att_done else "⏳")

    if rt_done and mem_done and att_done:
        avg_rt = round(sum(st.session_state['rt_list'])/3)
        st.session_state['cognitive_results'] = {
            'reaction_time_ms':    avg_rt,
            'memory_score_pct':    st.session_state['mem_score'],
            'attention_score_pct': st.session_state['att_score']
        }
        st.success("🎉 All tests complete! Go to **🏠 Home & Prediction** for enhanced results.")
        if st.button("🔄 Reset Tests"):
            for k in ['rt_list','rt_phase','rt_start','rt_done',
                      'mem_seq','mem_phase','mem_score',
                      'att_items','att_target','att_score','cognitive_results']:
                st.session_state.pop(k, None)
            st.rerun()

# ════════════════════════════════════════════════════════════
#  PAGE 3: EYE FATIGUE
# ════════════════════════════════════════════════════════════
elif page == "👁️ Eye Fatigue Analysis":
    st.title("👁️ Eye Fatigue Analysis")
    st.markdown("Upload a clear photo of your eyes. OpenCV will analyze fatigue indicators.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📸 Photo Tips**")
        st.markdown("- Good lighting\n- Look directly at camera\n- Eyes fully visible\n- Phone selfie works great")
    with col2:
        st.markdown("**🔬 What We Analyze**")
        st.markdown("- Eye openness ratio\n- Redness level (scleral)\n- Brightness contrast\n- Dark region (pupil) ratio")

    st.divider()
    uploaded = st.file_uploader("Upload Eye Photo (JPG/PNG)", type=["jpg","jpeg","png"])

    if uploaded:
        pil_img   = Image.open(uploaded).convert('RGB')
        img_array = np.array(pil_img)

        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_img, caption="Uploaded", use_container_width=True)

        with st.spinner("🔍 Analyzing..."):
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))

            annotated = img_array.copy()
            eye_detected = len(eyes) > 0
            openness = 50.0

            if eye_detected:
                ex,ey,ew,eh = max(eyes, key=lambda e: e[2]*e[3])
                cv2.rectangle(annotated,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                cv2.putText(annotated,"Eye Detected",(ex,ey-10),
                           cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
                eye_gray = gray[ey:ey+eh, ex:ex+ew]
                _,bright = cv2.threshold(eye_gray,180,255,cv2.THRESH_BINARY)
                openness  = round(np.sum(bright>0)/bright.size*100, 2)

            r = img_array[:,:,0].astype(float)
            g = img_array[:,:,1].astype(float)
            b = img_array[:,:,2].astype(float)
            redness   = round(float(np.mean(np.clip(r-(g+b)/2,0,255)))/255*100, 2)
            brightness= round(float(np.mean(gray)), 2)
            contrast  = round(float(np.std(gray)), 2)

            # Fatigue score
            fatigue = 0.0
            fatigue += 35 if openness<15 else 20 if openness<30 else 10 if openness<45 else 0
            fatigue += 30 if redness>25  else 15 if redness>15  else 8  if redness>8   else 0
            fatigue += 20 if brightness<80 else 10 if brightness<120 else 0
            fatigue += 15 if contrast<30  else 8  if contrast<50    else 0
            fatigue  = round(min(100, fatigue), 1)

        with col2:
            st.image(annotated, caption="Analysis", use_container_width=True)

        st.divider()
        level = ("Alert" if fatigue<25 else "Mildly Fatigued" if fatigue<50
                 else "Fatigued" if fatigue<75 else "Severely Fatigued")
        color = "green" if fatigue<25 else "orange" if fatigue<50 else "red"
        advice= ("No significant fatigue. Good to go!" if fatigue<25
                 else "Some strain. Consider a short break." if fatigue<50
                 else "Noticeable fatigue. Take a 20-min break." if fatigue<75
                 else "High fatigue. Rest your eyes immediately.")

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Fatigue Score",  f"{fatigue}/100")
        c2.metric("Level",          level)
        c3.metric("Eye Openness",   f"{openness:.1f}%")
        c4.metric("Redness",        f"{redness:.1f}%")
        st.markdown(f"**:{color}[{advice}]**")

        st.divider()
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("**Visual Indicators**")
            for name,val in [("Eye Openness",openness),
                              ("Brightness",  min(100,brightness/255*100)),
                              ("Contrast",    min(100,contrast/128*100))]:
                icon = "🟢" if val>60 else "🟡" if val>30 else "🔴"
                st.markdown(f"{icon} **{name}:** {val:.1f}%")
                st.progress(int(min(100,val)))
        with mc2:
            st.markdown("**Strain Indicators**")
            for name,val in [("Redness",  min(100,redness*2)),
                              ("Fatigue",  fatigue)]:
                icon = "🔴" if val>60 else "🟡" if val>30 else "🟢"
                st.markdown(f"{icon} **{name}:** {val:.1f}%")
                st.progress(int(val))

        if not eye_detected:
            st.warning("⚠️ Eye region not detected — try a closer photo for better results.")

        st.session_state['eye_fatigue_score'] = fatigue
        st.session_state['eye_fatigue_level'] = level
        st.success("✅ Saved! Go to **🏠 Home & Prediction** to include this in your result.")

        with st.expander("📚 Academic Context"):
            st.markdown("""
            **Why eye analysis matters for this research:**
            - Liu & Zhou (2024) identified visual fatigue as a neuropsychological correlate of gaming disorder
            - Eye Aspect Ratio (EAR) technique from Soukupová & Čech (2016) used in driver fatigue detection
            - Scleral redness is used clinically to assess prolonged screen exposure
            - This adds **objective biometric measurement** to complement self-reported gaming data
            """)

# ════════════════════════════════════════════════════════════
#  PAGE 4: MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance Dashboard")
    tab1,tab2,tab3 = st.tabs(["Cross-Validation","Feature Importance","CV Distribution"])
    for tab, fname, title in zip(
        [tab1,tab2,tab3],
        ['chartA_cross_validation.png','chartB_feature_importance.png','chartC_cv_boxplot.png'],
        ['5-Fold CV Comparison','Feature Importances','CV Score Distribution']
    ):
        path = os.path.join(BASE,'results',fname)
        with tab:
            st.subheader(title)
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.warning("Run 02_model_training_v2.py to generate this chart.")

    st.divider()
    st.subheader("Model Comparison Summary")
    st.dataframe(pd.DataFrame({
        'Model':         ['Decision Tree','Random Forest','Gradient Boosting','SVM'],
        'Test Accuracy': ['66.67%','80.00%','83.33%','76.67%'],
        'CV Accuracy':   ['67.00%','82.00%','78.00%','68.67%'],
        'CV Std':        ['±3.71%','±5.52%','±5.31%','±2.87%'],
        'Selected':      ['❌','✅ Best','❌','❌']
    }), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════
#  PAGE 5: DATA INSIGHTS
# ════════════════════════════════════════════════════════════
elif page == "📈 Data Insights":
    st.title("📈 Dataset Insights")
    datapath = 'data/Gaming_Dataset_v2.csv'  # no 'data/' prefix  # or better: 'data/gaming_dataset_v2.csv'
    if not os.path.exists(datapath):
        st.error(f"Dataset not found at: {datapath}")
        st.info(f"Current working dir: {os.getcwd()}")
        st.stop()

    df = pd.read_csv(datapath)
    c1,c2,c3 = st.columns(3)
    c1.metric("Students",     len(df))
    c2.metric("Avg Hrs/Week", f"{df['total_hours_week'].mean():.1f}")
    c3.metric("Night Gamers", f"{df['night_playing'].mean()*100:.0f}%")

    st.divider()
    tab1,tab2,tab3 = st.tabs(["Distribution","Gaming vs Cognition","Correlations"])

    with tab1:
        fig,axes = plt.subplots(1,2,figsize=(11,4))
        colors   = ['#2ecc71','#f39c12','#e74c3c']
        counts   = df['addiction_level'].value_counts().reindex(['Low','Medium','High'])
        axes[0].bar(counts.index, counts.values, color=colors, edgecolor='white')
        axes[0].set_title('Addiction Level Distribution')
        for i,v in enumerate(counts.values):
            axes[0].text(i, v+1, str(v), ha='center', fontweight='bold')
        axes[1].pie(counts.values, labels=counts.index, colors=colors, autopct='%1.1f%%')
        axes[1].set_title('Proportion')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab2:
        fig,axes = plt.subplots(1,2,figsize=(12,4))
        cmap = {'Low':'#2ecc71','Medium':'#f39c12','High':'#e74c3c'}
        for lvl,col in cmap.items():
            sub = df[df['addiction_level']==lvl]
            axes[0].scatter(sub['total_hours_week'],sub['attention_score'],
                           c=col,label=lvl,alpha=0.6,s=40)
            axes[1].scatter(sub['total_hours_week'],sub['memory_score'],
                           c=col,label=lvl,alpha=0.6,s=40)
        for ax,t in zip(axes,['Hours vs Attention','Hours vs Memory']):
            ax.set_xlabel('Hours/Week'); ax.set_ylabel('Score')
            ax.set_title(t); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        cols = ['total_hours_week','sleep_hours','gpa','attention_score','memory_score','reaction_time']
        corr = df[cols].corr()
        fig,ax = plt.subplots(figsize=(8,6))
        im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar(im,ax=ax)
        ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
        ax.set_xticklabels([c.replace('_',' ') for c in cols],rotation=45,ha='right')
        ax.set_yticklabels([c.replace('_',' ') for c in cols])
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j,i,f'{corr.values[i,j]:.2f}',ha='center',va='center',fontsize=9)
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════
#  PAGE 6: HISTORY
# ════════════════════════════════════════════════════════════
elif page == "📋 Student History":
    st.title("📋 Session Prediction History")
    if 'history' not in st.session_state or not st.session_state['history']:
        st.info("No predictions yet. Go to **🏠 Home & Prediction** first!")
    else:
        df_h = pd.DataFrame(st.session_state['history'])
        st.dataframe(df_h, use_container_width=True, hide_index=True)
        fig,ax = plt.subplots(figsize=(7,3))
        rc = df_h['Risk Level'].str.extract(r'(Low|Medium|High)')[0].value_counts()
        colors = {'Low':'#2ecc71','Medium':'#f39c12','High':'#e74c3c'}
        ax.bar(rc.index, rc.values, color=[colors.get(k,'gray') for k in rc.index])
        ax.set_title('Risk Distribution This Session')
        st.pyplot(fig); plt.close()
        if st.button("🗑️ Clear History"):
            st.session_state['history'] = []
            st.rerun()

# ── Footer ───────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("🎓 BSc (Hons) Computer Science\nKDU | 2026\nRandom Forest | CV: 82.00%")