# ============================================================
#  EYE-STATE MODEL (open vs closed) — version 2
#  Data:
#    - MRL Eye Dataset (Fusek, 2018): 84,898 infrared eye crops of 37 people.
#      File names: subject_image_gender_glasses_EYESTATE_reflections_lighting_sensor
#      -> eye state is field 4 (0 = closed, 1 = open).
#    - CEW, Closed Eyes in the Wild (Song et al., 2014): 2,423 real colour faces
#      (research use only; git-ignored). Eye crops are cut with the app's own
#      pipeline (eye_analysis.find_eye_boxes) so they match what the app scores.
#  Improvements over version 1 (94.9% on unseen MRL people):
#    more MRL images per person, real-world CEW crops, label smoothing,
#    decision threshold chosen on validation people, mirror-image averaging (TTA).
#  Evaluation: the SAME 8 MRL test people as version 1 (never seen in training)
#  and a held-out set of CEW faces. Nothing from either test set is used for tuning.
#  Output: P(eye closed).
#  Run:  python notebooks/08_train_eye_state_model.py
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import glob
import json
import os
import random
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, BASE)
import eye_analysis as ea  # noqa: E402

MRL_DIR = os.path.join(BASE, 'data', 'eye_dataset')
CEW_DIR = os.path.join(BASE, 'data', 'external', 'cew')
MODEL_PATH = os.path.join(BASE, 'models', 'eye_state_cnn.keras')
META_PATH = os.path.join(BASE, 'models', 'eye_state_cnn.json')
RESULTS = os.path.join(BASE, 'results')
IMG = 48
MRL_PER_SUBJECT_CLASS = 1200      # v1 used 700
CEW_OVERSAMPLE = 3                # CEW crops are few next to MRL; repeat them in training
V1_TEST_PEOPLE = ['s0005', 's0006', 's0010', 's0013', 's0021', 's0023', 's0027', 's0028']
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── 1. MRL: index files and split by person (same split as version 1) ────────
records = []
for f in glob.glob(os.path.join(MRL_DIR, '**', '*.png'), recursive=True):
    parts = os.path.basename(f)[:-4].split('_')
    records.append((f, parts[0], 1 - int(parts[4])))        # label 1 = closed
subjects = sorted({r[1] for r in records})
rng = random.Random(SEED)
shuffled = subjects[:]
rng.shuffle(shuffled)
test_subj, val_subj, train_subj = set(shuffled[:8]), set(shuffled[8:12]), set(shuffled[12:])
assert sorted(test_subj) == V1_TEST_PEOPLE, "MRL test people must match version 1 for a fair comparison"
by_key = {}
for r in records:
    by_key.setdefault((r[1], r[2]), []).append(r)


def sample(subj_set, cap):
    out = []
    for s in sorted(subj_set):
        for label in (0, 1):
            group = by_key.get((s, label), [])[:]
            rng.shuffle(group)
            out += group[:cap]
    rng.shuffle(out)
    return out


def load_mrl(recs):
    X = np.stack([ea.prep_crop(cv2.imread(f, cv2.IMREAD_GRAYSCALE), IMG) for f, _, _ in recs])[..., None]
    return X, np.array([lab for _, _, lab in recs], dtype='float32'), np.array([s for _, s, _ in recs])


X_mtr, y_mtr, _ = load_mrl(sample(train_subj, MRL_PER_SUBJECT_CLASS))
X_mva, y_mva, _ = load_mrl(sample(val_subj, 400))
X_mte, y_mte, s_mte = load_mrl(sample(test_subj, 10 ** 6))
print(f"MRL: train {len(y_mtr)}, val {len(y_mva)}, test {len(y_mte)} images "
      f"({len(train_subj)}/{len(val_subj)}/{len(test_subj)} people)")

# ── 2. CEW: crop both eyes with the app's pipeline; split by face ────────────
face_det, haar = ea.make_face_detector(), ea.make_eye_cascade()
cew = []
for label, folder in ((0, 'OpenFace'), (1, 'ClosedFace')):
    files = sorted(f for f in glob.glob(os.path.join(CEW_DIR, folder, '*')) if f.lower().endswith('.jpg'))
    for f in files:
        big = cv2.resize(cv2.imread(f), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        boxes, method = ea.find_eye_boxes(big, face_det, haar)
        if method == 'face':
            cew.append((os.path.basename(f), label, [ea.prep_crop(big[y0:y1, x0:x1], IMG) for x0, y0, x1, y1 in boxes]))
rng.shuffle(cew)
n = len(cew)
splits = {'train': cew[:int(0.7 * n)], 'val': cew[int(0.7 * n):int(0.85 * n)], 'test': cew[int(0.85 * n):]}


def stack_cew(faces):
    X = np.stack([c for _, _, crops in faces for c in crops])[..., None]
    y = np.array([lab for _, lab, crops in faces for _ in crops], dtype='float32')
    return X, y


X_ctr, y_ctr = stack_cew(splits['train'])
X_cva, y_cva = stack_cew(splits['val'])
X_cte, y_cte = stack_cew(splits['test'])
print(f"CEW: {n} faces with eyes found -> crops train {len(y_ctr)}, val {len(y_cva)}, test {len(y_cte)}")

X_tr = np.concatenate([X_mtr] + [X_ctr] * CEW_OVERSAMPLE)
y_tr = np.concatenate([y_mtr] + [y_ctr] * CEW_OVERSAMPLE)
X_va, y_va = np.concatenate([X_mva, X_cva]), np.concatenate([y_mva, y_cva])
perm = np.random.permutation(len(y_tr))
X_tr, y_tr = X_tr[perm], y_tr[perm]
print(f"Training set: {len(y_tr)} crops (closed {y_tr.mean():.0%}); validation {len(y_va)}")

# ── 3. Model (same architecture as v1, so gains come from data and training) ─
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomTranslation(0.08, 0.08),
    tf.keras.layers.RandomZoom((-0.15, 0.15)),
    tf.keras.layers.RandomBrightness(0.25, value_range=(0, 1)),
    tf.keras.layers.RandomContrast(0.35),
    tf.keras.layers.GaussianNoise(0.03),
], name='augment')


def conv_block(x, filters):
    for _ in range(2):
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    return tf.keras.layers.MaxPooling2D()(x)


inp = tf.keras.Input((IMG, IMG, 1))
x = tf.keras.layers.Rescaling(1 / 255.0)(inp)
x = augment(x)
for filters in (24, 48, 96):
    x = conv_block(x, filters)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
out = tf.keras.layers.Dense(1, activation='sigmoid', name='p_closed')(x)
model = tf.keras.Model(inp, out)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=15, batch_size=128, verbose=2,
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=4,
                                                             restore_best_weights=True),
                            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=1)])


def predict_tta(X):
    p = model.predict(X, batch_size=256, verbose=0).ravel()
    return (p + model.predict(X[:, :, ::-1, :], batch_size=256, verbose=0).ravel()) / 2


# ── 4. Decision threshold from the VALIDATION people/faces only ──────────────
p_va = predict_tta(X_va)
grid = np.linspace(0.2, 0.8, 61)
bal = [0.5 * (recall_score(y_va, p_va >= t) + recall_score(1 - y_va, p_va < t)) for t in grid]
threshold = float(grid[int(np.argmax(bal))])
print(f"Threshold chosen on validation: {threshold:.2f} (balanced accuracy {max(bal):.3f})")


def evaluate(X, y):
    p = predict_tta(X)
    pred = (p >= threshold).astype(int)
    return p, pred, {
        'images': int(len(y)), 'accuracy': float(accuracy_score(y, pred)), 'auc': float(roc_auc_score(y, p)),
        'recall_closed': float(recall_score(y, pred)), 'precision_closed': float(precision_score(y, pred)),
        'recall_open': float(recall_score(1 - y, 1 - pred)),
        'balanced_accuracy': float(0.5 * (recall_score(y, pred) + recall_score(1 - y, 1 - pred))),
    }


p_mte, pred_mte, mrl = evaluate(X_mte, y_mte)
_, _, cewm = evaluate(X_cte, y_cte)
mrl['per_person_accuracy'] = {s: float(accuracy_score(y_mte[s_mte == s], pred_mte[s_mte == s])) for s in V1_TEST_PEOPLE}
print("\nMRL test (8 people never seen in training):")
for k in ('accuracy', 'balanced_accuracy', 'auc', 'recall_closed', 'recall_open'):
    print(f"  {k:18s} {mrl[k]:.3f}")
print("  per person:", {k: round(v, 3) for k, v in mrl['per_person_accuracy'].items()})
print("CEW test (held-out real-world faces):")
for k in ('accuracy', 'balanced_accuracy', 'auc', 'recall_closed', 'recall_open'):
    print(f"  {k:18s} {cewm[k]:.3f}")

model.save(MODEL_PATH)
meta = {
    'version': 2, 'img_size': IMG, 'threshold': threshold, 'tta_flip': True,
    'preprocess': 'grayscale, resize 48x48 (INTER_AREA), cv2.equalizeHist, /255 inside model',
    'output': 'P(eye closed); app rescales so the threshold maps to 0.5',
    'dataset': 'MRL Eye Dataset (Fusek, 2018) + CEW (Song et al., 2014; research use only)',
    'split': 'MRL subject-wise 25/4/8 people (same 8 test people as v1); CEW split by face 70/15/15',
    'test_subjects': V1_TEST_PEOPLE, 'epochs_trained': len(hist.history['loss']),
    # top-level figures (shown in the app) refer to the MRL unseen-people test, as in v1
    'test_people': len(V1_TEST_PEOPLE), 'test_images': mrl['images'], 'accuracy': mrl['accuracy'],
    'auc': mrl['auc'], 'recall_closed': mrl['recall_closed'], 'recall_open': mrl['recall_open'],
    'precision_closed': mrl['precision_closed'], 'balanced_accuracy': mrl['balanced_accuracy'],
    'per_person_accuracy': mrl['per_person_accuracy'], 'cew_test': cewm,
}
with open(META_PATH, 'w', encoding='utf-8') as fh:
    json.dump(meta, fh, indent=2)

# ── 5. Charts ────────────────────────────────────────────────────────────────
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
a1.plot(hist.history['accuracy'], label='train')
a1.plot(hist.history['val_accuracy'], label='validation (unseen people + CEW faces)')
a1.set_title('Accuracy', fontsize=9, fontweight='bold')
a1.set_xlabel('Epoch', fontsize=8)
a1.legend(fontsize=7)
a2.plot(hist.history['loss'], label='train')
a2.plot(hist.history['val_loss'], label='validation')
a2.set_title('Loss', fontsize=9, fontweight='bold')
a2.set_xlabel('Epoch', fontsize=8)
a2.legend(fontsize=7)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS, 'eye3_state_training_history.png'), dpi=150)
plt.close(fig)

cm = confusion_matrix(y_mte, pred_mte)
fig, ax = plt.subplots(figsize=(4.6, 4))
ax.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
ax.set_xticks([0, 1], ['Open', 'Closed'])
ax.set_yticks([0, 1], ['Open', 'Closed'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f"Unseen MRL people (n = {mrl['images']:,}): acc {mrl['accuracy']:.1%}, AUC {mrl['auc']:.3f}",
             fontsize=9, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(RESULTS, 'eye4_state_confusion_matrix.png'), dpi=150)
plt.close(fig)
print(f"\nSaved {MODEL_PATH} (threshold {threshold:.2f}, TTA) and charts in results/")
