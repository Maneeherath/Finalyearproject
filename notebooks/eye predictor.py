# ============================================================
#  EYE FATIGUE PREDICTOR — eye_predictor.py
#  Used by the Streamlit app to predict fatigue from photos
#  Loads the trained CNN model from 04_eye_model_train.py
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import streamlit as st
import numpy as np
import cv2
import os
import joblib
from PIL import Image

def load_eye_model(BASE):
    """Load trained CNN model and metadata"""
    model_path = os.path.join(BASE, 'models', 'eye_fatigue_model.h5')
    meta_path  = os.path.join(BASE, 'models', 'eye_model_metadata.pkl')

    if not os.path.exists(model_path):
        return None, None

    try:
        import tensorflow as tf
        model    = tf.keras.models.load_model(model_path)
        metadata = joblib.load(meta_path) if os.path.exists(meta_path) else {}
        return model, metadata
    except Exception as e:
        return None, None


def preprocess_eye_image(image_array, img_size=64):
    """
    Preprocess uploaded image for CNN prediction.
    Steps match exactly what was done during training.
    """
    # Convert to grayscale
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array

    # Try to detect and crop eye region using Haar cascade
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))

    eye_detected = len(eyes) > 0
    if eye_detected:
        # Use largest detected eye
        ex, ey, ew, eh = max(eyes, key=lambda e: e[2]*e[3])
        gray = gray[ey:ey+eh, ex:ex+ew]

    # Resize to model input size
    resized = cv2.resize(gray, (img_size, img_size))

    # Apply histogram equalization (same as training)
    equalized = cv2.equalizeHist(resized)

    # Normalize and reshape
    processed = equalized.astype('float32') / 255.0
    processed = processed.reshape(1, img_size, img_size, 1)

    return processed, eye_detected, eyes


def predict_eye_fatigue(model, metadata, image_array):
    """
    Run CNN prediction on preprocessed image.
    Returns fatigue probability and label.
    """
    img_size  = metadata.get('img_size', 64)
    threshold = metadata.get('threshold', 0.5)

    processed, eye_detected, eye_boxes = preprocess_eye_image(
        image_array, img_size
    )

    # Get prediction probability
    prob_fatigue = float(model.predict(processed, verbose=0)[0][0])
    prob_alert   = 1 - prob_fatigue

    # Apply threshold
    label     = 'Fatigued' if prob_fatigue >= threshold else 'Alert'
    label_idx = 1 if label == 'Fatigued' else 0

    # Convert to 0-100 fatigue score
    fatigue_score = round(prob_fatigue * 100, 1)

    return {
        'label':         label,
        'label_idx':     label_idx,
        'fatigue_score': fatigue_score,
        'prob_fatigued': round(prob_fatigue*100, 1),
        'prob_alert':    round(prob_alert*100, 1),
        'eye_detected':  eye_detected,
        'eye_boxes':     eye_boxes
    }


def show_eye_fatigue_page(BASE):
    """Full Streamlit page for eye fatigue analysis"""
    st.title("👁️ Eye Fatigue Analysis — CNN Model")

    # Load model
    model, metadata = load_eye_model(BASE)
    model_available = model is not None

    if model_available:
        acc = metadata.get('test_acc', 'N/A')
        n   = metadata.get('total_images', 'N/A')
        st.success(f"✅ Trained CNN loaded — Accuracy: **{acc}%** "
                   f"| Dataset: MRL Eye Dataset ({n:,} images)")
    else:
        st.warning("""
        ⚠️ Trained model not found. Using rule-based fallback.

        **To enable the full CNN model:**
        1. Download MRL Eye Dataset: http://mrl.cs.vsb.cz/eyedataset
        2. Extract to `data/eye_dataset/`
        3. Run: `python notebooks/04_eye_model_train.py`
        4. Refresh this page
        """)

    st.markdown("""
    Upload a clear photo of your eyes. The system will analyze fatigue
    using a **CNN trained on the MRL Eye Dataset** (84,898 eye images).
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📸 Photo Tips**")
        st.markdown("- Good lighting\n- Look directly at camera\n"
                    "- Eyes clearly visible\n- Phone selfie works well")
    with col2:
        st.markdown("**🔬 Model Details**")
        st.markdown("- Dataset: MRL Eye Dataset (Fusek 2018)\n"
                    "- Architecture: 3-block CNN\n"
                    "- Input: 64×64 grayscale\n"
                    "- Output: Alert / Fatigued")

    st.divider()

    uploaded = st.file_uploader("📁 Upload Eye Photo (JPG/PNG)",
                                type=["jpg","jpeg","png"])

    if uploaded:
        pil_img   = Image.open(uploaded).convert('RGB')
        img_array = np.array(pil_img)

        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_img, caption="Uploaded Photo",
                     use_container_width=True)

        with st.spinner("🔍 Analyzing eye fatigue..."):
            if model_available:
                # Use trained CNN
                result = predict_eye_fatigue(model, metadata, img_array)
                method = "CNN Model"
            else:
                # Fallback: rule-based
                result = _rule_based_fallback(img_array)
                method = "Rule-based (no model)"

        # Draw eye detection boxes
        annotated = img_array.copy()
        if result['eye_detected'] and len(result.get('eye_boxes',[])) > 0:
            for (ex,ey,ew,eh) in result['eye_boxes']:
                cv2.rectangle(annotated,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.putText(annotated, "Eye Detected", (10,30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        with col2:
            st.image(annotated, caption=f"Detected ({method})",
                     use_container_width=True)

        # ── Results ──────────────────────────────────────────
        st.divider()
        st.subheader("📊 Fatigue Prediction")

        fatigue = result['fatigue_score']
        label   = result['label']

        if label == 'Alert':
            color  = "green"
            icon   = "🟢"
            advice = "Eyes appear alert and well-rested. Good to study!"
        elif fatigue < 65:
            color  = "orange"
            icon   = "🟡"
            advice = "Mild fatigue detected. Consider a short break."
        else:
            color  = "red"
            icon   = "🔴"
            advice = "High fatigue detected. Rest your eyes — 20 min break recommended."

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Prediction",     f"{icon} {label}")
        c2.metric("Fatigue Score",  f"{fatigue}/100")
        c3.metric("Alert Prob",     f"{result['prob_alert']}%")
        c4.metric("Fatigue Prob",   f"{result['prob_fatigued']}%")

        st.markdown(f"**:{color}[{icon} {advice}]**")

        # Confidence bar
        st.markdown("**Model Confidence:**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"🟢 Alert: {result['prob_alert']}%")
            st.progress(int(result['prob_alert']))
        with col2:
            st.markdown(f"🔴 Fatigued: {result['prob_fatigued']}%")
            st.progress(int(result['prob_fatigued']))

        if not result['eye_detected']:
            st.warning("⚠️ Eye region not detected automatically. "
                       "Upload a closer photo for better accuracy.")

        # Model accuracy callout
        if model_available:
            st.info(f"🤖 Prediction made by CNN trained on "
                    f"{metadata.get('total_images',0):,} real eye images "
                    f"(MRL Dataset) — Model accuracy: {metadata.get('test_acc')}%")

        # Academic context
        with st.expander("📚 Academic Context"):
            st.markdown(f"""
            **Dataset:** MRL Eye Dataset (Fusek, R., 2018)
            - 84,898 images from 37 subjects
            - Captured under varied lighting and head positions
            - Used in peer-reviewed fatigue detection research

            **Model Architecture:** 3-block CNN
            - Block 1: Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout
            - Block 2: Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout
            - Block 3: Conv2D(128) → BatchNorm → MaxPool → Dropout
            - Head: Dense(256) → Dense(64) → Sigmoid output

            **Relevance to Research:**
            Liu & Zhou (2024) identified visual fatigue as a neuropsychological
            correlate of gaming disorder. This module provides **objective biometric
            measurement** of fatigue using a trained CNN rather than heuristic rules.
            """)

        # Save result
        st.session_state['eye_fatigue_score'] = fatigue
        st.session_state['eye_fatigue_label'] = label
        st.session_state['eye_model_used']    = method
        st.success("✅ Eye fatigue result saved — used in your next prediction!")


def _rule_based_fallback(img_array):
    """Fallback when CNN model not available"""
    import cv2
    gray       = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    cascade    = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
    eyes       = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    detected   = len(eyes) > 0
    brightness = float(np.mean(gray))
    redness    = float(np.mean(np.clip(
        img_array[:,:,0].astype(float) -
        (img_array[:,:,1].astype(float)+img_array[:,:,2].astype(float))/2,
        0, 255
    ))) / 255 * 100
    fatigue = min(100, round(
        (20 if brightness<100 else 0) +
        (30 if redness>20 else 10 if redness>10 else 0)
    ))
    return {
        'label':         'Fatigued' if fatigue>=50 else 'Alert',
        'label_idx':     1 if fatigue>=50 else 0,
        'fatigue_score': fatigue,
        'prob_fatigued': fatigue,
        'prob_alert':    100-fatigue,
        'eye_detected':  detected,
        'eye_boxes':     eyes
    }