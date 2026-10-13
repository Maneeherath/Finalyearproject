# ============================================================
#  EYE ANALYSIS (shared by the app's Eye Check and eye_monitor.py)
#  1. Locate the eyes: YuNet face detector (OpenCV model zoo, MIT licence) gives
#     both eye centres; for close-up photos of a single eye, fall back to
#     OpenCV's Haar eye detector.
#  2. Crop each eye the way the MRL training images are framed (square, eye in
#     the middle), convert to grayscale and equalise, like the training data.
#  3. Eye-state CNN (notebooks/08_train_eye_state_model.py) -> P(eye closed).
#  Eye fatigue score = 100 x mean P(closed): drooping or closing eyelids are the
#  classic visible sign of drowsiness (as in PERCLOS; Wierwille et al., 1994).
#  Redness is reported as an uncalibrated indicator only.
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import json
import os

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, 'models', 'eye_state_cnn.keras')
META_PATH = os.path.join(BASE, 'models', 'eye_state_cnn.json')
YUNET_PATH = os.path.join(BASE, 'models', 'face_detection_yunet_2023mar.onnx')
CROP_PER_IOD = 0.8          # crop side relative to the distance between the eyes (matches MRL framing)
CROP_PER_EYEBOX = 1.6       # crop side relative to a Haar eye box (close-up photos)
MAX_SIDE = 960              # downscale large photos before detection
MIN_CROP = 40               # eye crops smaller than this (px) are too blurry to judge; MRL crops are ~60-120 px


def level_of(score):
    return ("Alert" if score < 25 else "Mildly Fatigued" if score < 50
            else "Fatigued" if score < 75 else "Severely Fatigued")


class EyeAnalyzer:
    def __init__(self):
        from tensorflow.keras.models import load_model
        self.model = load_model(MODEL_PATH, compile=False)
        with open(META_PATH, encoding='utf-8') as f:
            self.meta = json.load(f)
        self.size = self.meta['img_size']
        self.face = cv2.FaceDetectorYN.create(YUNET_PATH, '', (320, 320), 0.7, 0.3, 5000)
        self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # ── locating eyes ────────────────────────────────────────────────────────
    @staticmethod
    def _square(img, cx, cy, side):
        h, w = img.shape[:2]
        half = max(8, int(side / 2))
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
        x1, y1 = int(min(w, cx + half)), int(min(h, cy + half))
        return (x0, y0, x1, y1) if x1 - x0 > 10 and y1 - y0 > 10 else None

    def find_eyes(self, bgr):
        """Returns (list of (x0, y0, x1, y1) boxes, method)."""
        h, w = bgr.shape[:2]
        self.face.setInputSize((w, h))
        _, faces = self.face.detect(bgr)
        if faces is not None and len(faces):
            f = max(faces, key=lambda r: r[14])                  # highest-confidence face
            right, left = (f[4], f[5]), (f[6], f[7])
            iod = float(np.hypot(left[0] - right[0], left[1] - right[1]))
            boxes = [b for b in (self._square(bgr, x, y, CROP_PER_IOD * iod) for x, y in (right, left)) if b]
            if boxes:
                return boxes, 'face'
        gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        eyes = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(max(24, w // 12), max(24, w // 12)))
        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        boxes = [b for b in (self._square(bgr, x + ew / 2, y + eh / 2, CROP_PER_EYEBOX * max(ew, eh))
                             for x, y, ew, eh in eyes) if b]
        return boxes, ('eye close-up' if boxes else 'none')

    # ── scoring ──────────────────────────────────────────────────────────────
    def _prep(self, bgr_crop):
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA))

    def p_closed(self, bgr, boxes):
        if not boxes:
            return []
        X = np.stack([self._prep(bgr[y0:y1, x0:x1]) for x0, y0, x1, y1 in boxes])[..., None].astype('float32')
        return [float(p) for p in self.model.predict(X, verbose=0).ravel()]

    @staticmethod
    def redness(bgr, boxes):
        """Uncalibrated indicator: mean red-green shift (LAB a*) of the bright, white-ish pixels (sclera)."""
        vals = []
        for x0, y0, x1, y1 in boxes:
            lab = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2LAB)
            L, a = lab[..., 0].astype(float), lab[..., 1].astype(float) - 128
            mask = L > np.percentile(L, 80)                       # brightest 20% ~ sclera
            if mask.sum() > 20:
                vals.append(float(np.clip(a[mask].mean() / 30 * 100, 0, 100)))
        return round(float(np.mean(vals)), 1) if vals else None

    def analyze(self, rgb):
        """Full analysis of one RGB photo -> dict with score, per-eye results and an annotated image."""
        bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        scale = 1.0
        if max(bgr.shape[:2]) > MAX_SIDE:
            scale = MAX_SIDE / max(bgr.shape[:2])
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        boxes, method = self.find_eyes(bgr)
        too_small = bool(boxes) and min(x1 - x0 for x0, _, x1, _ in boxes) < MIN_CROP
        probs = [] if too_small else self.p_closed(bgr, boxes)
        annotated = bgr.copy()
        for x0, y0, x1, y1 in (boxes if too_small else []):
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 165, 255), 2)       # orange: found but too small
        for (x0, y0, x1, y1), p in zip(boxes, probs):
            color = (0, 0, 255) if p >= 0.5 else (0, 200, 0)
            cv2.rectangle(annotated, (x0, y0), (x1, y1), color, max(2, bgr.shape[1] // 300))
            cv2.putText(annotated, f"{'closed' if p >= 0.5 else 'open'} {p:.2f}", (x0, max(15, y0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.5, bgr.shape[1] / 1200), color, 2)
        score = round(100 * float(np.mean(probs)), 1) if probs else None
        reason = 'too_small' if too_small else ('no_eyes' if not boxes else None)
        return {
            'eyes_found': len(boxes), 'method': method, 'p_closed': probs, 'fatigue_score': score, 'reason': reason,
            'level': level_of(score) if score is not None else None,
            'redness': self.redness(bgr, boxes) if boxes else None,
            'annotated_rgb': cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            'crops': [cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB) for x0, y0, x1, y1 in boxes],
        }
