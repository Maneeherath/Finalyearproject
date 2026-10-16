# System-level tests of the real models and pipeline. Writes results/testing/system_tests.json.
# Run: python tests/system_tests.py
# Simulated profiles are labelled as simulated; all numbers are produced by the project's own models.
import glob
import itertools
import json
import os
import sys
import time

import cv2
import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import eye_analysis as ea  # noqa: E402
import fusion_engine as fe  # noqa: E402
import igd_features as F  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'testing')
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)
igd = joblib.load(os.path.join(ROOT, 'models', 'igd_risk_model.pkl'))
model, meta = igd['model'], igd['meta']
SIGMA = meta['sigma']
R = {}


def predict(inp):
    return float(np.clip(model.predict(F.features_from_inputs(inp))[0], 9, 45))


def profile(**kw):
    base = dict(age=21, gender='Male', start_age=10, sports='Minor', sports_level=1, weekday_hours=1.5,
                weekend_hours=2.5, mode=F.MODES[0], genre=F.GENRES[0], device=F.DEVICES[0], sleep_hours=7.0,
                sleep_quality=7, self_esteem=4, few_friends=2)
    base.update(kw)
    return base


def random_profile():
    return profile(age=int(rng.integers(16, 36)), gender=str(rng.choice(['Male', 'Female'])),
                   start_age=int(rng.integers(3, 20)), sports_level=int(rng.integers(0, 4)),
                   weekday_hours=float(rng.choice(np.arange(0, 12.5, 0.5))),
                   weekend_hours=float(rng.choice(np.arange(0, 14.5, 0.5))),
                   mode=str(rng.choice(F.MODES)), genre=str(rng.choice(F.GENRES)), device=str(rng.choice(F.DEVICES)),
                   sleep_hours=float(rng.choice(np.arange(3, 10.5, 0.5))), sleep_quality=int(rng.integers(0, 11)),
                   self_esteem=int(rng.integers(1, 6)), few_friends=int(rng.integers(1, 6)))


# ── A. Exhaustive input-space test: every categorical combination × hour/sleep extremes ─────────
t0 = time.perf_counter()
preds, n = [], 0
for mode, genre, device, gender in itertools.product(F.MODES, F.GENRES, F.DEVICES, ['Male', 'Female']):
    for wd, we, sl, sq in itertools.product([0, 6, 12], [0, 7, 14], [3, 10], [0, 10]):
        preds.append(predict(profile(mode=mode, genre=genre, device=device, gender=gender, weekday_hours=wd,
                                     weekend_hours=we, sleep_hours=sl, sleep_quality=sq)))
        n += 1
preds = np.array(preds)
R['input_space'] = {'combinations': n, 'errors': 0, 'min_pred': round(float(preds.min()), 2),
                    'max_pred': round(float(preds.max()), 2), 'all_in_range': bool(((preds >= 9) & (preds <= 45)).all()),
                    'seconds': round(time.perf_counter() - t0, 1)}

# ── B. Monotonicity: more gaming should never lower the predicted score ─────────────────────────
def monotonic_check(field, values, expect='increasing', n_profiles=200):
    viol, worst = 0, 0.0
    for _ in range(n_profiles):
        base = random_profile()
        s = [predict({**base, field: v}) for v in values]
        d = np.diff(s)
        bad = d < -0.05 if expect == 'increasing' else d > 0.05
        viol += int(bad.any())
        worst = max(worst, float(np.max(-d if expect == 'increasing' else d)))
    return {'profiles': n_profiles, 'profiles_with_violation': viol, 'largest_reverse_step': round(worst, 2)}


R['monotonicity'] = {
    'weekday_hours': monotonic_check('weekday_hours', np.arange(0, 12.5, 1.0)),
    'weekend_hours': monotonic_check('weekend_hours', np.arange(0, 14.5, 1.0)),
    'sleep_quality': monotonic_check('sleep_quality', np.arange(0, 11, 1), expect='decreasing'),
}

# ── C. Reproducibility: same input -> same output ───────────────────────────────────────────────
p = random_profile()
outs = {round(predict(p), 6) for _ in range(100)}
R['reproducibility'] = {'repeats': 100, 'distinct_outputs': len(outs)}

# ── D. Batch test: 100 simulated students through behaviour model + fusion ───────────────────────
rows = []
t0 = time.perf_counter()
for i in range(100):
    inp = random_profile()
    score = predict(inp)
    beh_p, beh_c = fe.behaviour_vote(F.tier_probabilities(score, SIGMA))
    cog = fe.cognitive_inhibition_vote(float(rng.choice([0, 25, 50, 75, 100])))
    eye = fe.eye_fatigue_vote(float(rng.uniform(0, 100))) if rng.random() > 0.2 else (np.ones(3) / 3, 0.0)
    wb = fe.wellbeing_vote(int(rng.integers(0, 7)), int(rng.integers(0, 7)), int(rng.integers(3, 10)))
    fused, w = fe.fuse([(beh_p, beh_c), cog, eye, wb])
    rows.append({'score': score, 'beh': int(np.argmax(beh_p)), 'fused': int(np.argmax(fused)),
                 'p_high': float(fused[2]), 'w_beh': w['Behaviour_weight'],
                 'possible_high': bool(np.argmax(fused) < 2 and fused[2] >= 0.30)})
elapsed = time.perf_counter() - t0
beh = np.array([r['beh'] for r in rows]); fus = np.array([r['fused'] for r in rows])
R['batch'] = {
    'students': 100, 'errors': 0, 'ms_per_assessment': round(elapsed / 100 * 1000, 1),
    'behaviour_tiers_LMH': [int((beh == k).sum()) for k in range(3)],
    'fused_tiers_LMH': [int((fus == k).sum()) for k in range(3)],
    'tier_changed_by_fusion': int((beh != fus).sum()),
    'changed_up': int((fus > beh).sum()), 'changed_down': int((fus < beh).sum()),
    'possible_high_warnings': int(sum(r['possible_high'] for r in rows)),
    'behaviour_weight_min_median_max': [round(float(np.min([r['w_beh'] for r in rows])), 2),
                                        round(float(np.median([r['w_beh'] for r in rows])), 2),
                                        round(float(np.max([r['w_beh'] for r in rows])), 2)],
    'score_min_median_max': [round(float(np.min([r['score'] for r in rows])), 1),
                             round(float(np.median([r['score'] for r in rows])), 1),
                             round(float(np.max([r['score'] for r in rows])), 1)],
}

# ── E. Timing of the main computations ──────────────────────────────────────────────────────────
X = F.features_from_inputs(profile())
t = []
for _ in range(50):
    s = time.perf_counter(); model.predict(X); t.append(time.perf_counter() - s)
t_f = []
for _ in range(1000):
    s = time.perf_counter(); fe.fuse([fe.behaviour_vote([.2, .5, .3]), fe.cognitive_inhibition_vote(50)]); t_f.append(time.perf_counter() - s)
R['timing_ms'] = {'behaviour_prediction_median': round(float(np.median(t)) * 1000, 1),
                  'fusion_median': round(float(np.median(t_f)) * 1000, 3)}

# ── F. Eye analysis: speed and robustness to image changes on labelled real-world photos ───────
an = ea.EyeAnalyzer()
labelled = [(f, 0) for f in sorted(glob.glob(os.path.join(ROOT, 'demo_run', 'eye_test', 'open', '*.jpg')))] + \
           [(f, 1) for f in sorted(glob.glob(os.path.join(ROOT, 'demo_run', 'eye_test', 'closed', '*.jpg')))] + \
           [(f, 0) for f in sorted(glob.glob(os.path.join(ROOT, 'demo_run', 'user_eye_photos', '*.jpg')))]


def rotate(img, deg):
    h, w = img.shape[:2]
    return cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0), (w, h), borderMode=cv2.BORDER_REFLECT)


def jpeg(img, q):
    return cv2.imdecode(cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


CONDITIONS = {
    'original': lambda im: im,
    'darker (×0.5)': lambda im: cv2.convertScaleAbs(im, alpha=0.5),
    'brighter (×1.5)': lambda im: cv2.convertScaleAbs(im, alpha=1.5),
    'blur (7×7)': lambda im: cv2.GaussianBlur(im, (7, 7), 0),
    'noise (σ = 15)': lambda im: np.clip(im + np.random.default_rng(1).normal(0, 15, im.shape), 0, 255).astype(np.uint8),
    'rotated 10°': lambda im: rotate(im, 10),
    'rotated 20°': lambda im: rotate(im, 20),
    'half size': lambda im: cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
    'quarter size': lambda im: cv2.resize(im, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA),
    'JPEG quality 20': lambda im: jpeg(im, 20),
}
rob, times = {}, []
for name, fn in CONDITIONS.items():
    scored = correct = 0
    for f, lab in labelled:
        img = fn(cv2.imread(f))
        s = time.perf_counter()
        r = an.analyze(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if name == 'original':
            times.append(time.perf_counter() - s)
        if r['fatigue_score'] is None:
            continue
        scored += 1
        correct += int((r['fatigue_score'] >= 50) == bool(lab))
    rob[name] = {'photos': len(labelled), 'scored': scored, 'correct': correct,
                 'accuracy_of_scored': round(correct / scored, 3) if scored else None}
R['eye_robustness'] = {'photos': len(labelled), 'open': sum(1 for _, l in labelled if l == 0),
                       'closed': sum(1 for _, l in labelled if l == 1), 'conditions': rob}
R['timing_ms']['eye_analysis_per_photo_median'] = round(float(np.median(times)) * 1000, 1)

json.dump(R, open(os.path.join(OUT, 'system_tests.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
