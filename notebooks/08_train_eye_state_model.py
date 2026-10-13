# ============================================================
#  EYE-STATE MODEL (open vs closed) — replaces the earlier eye models
#  Dataset: MRL Eye Dataset (Fusek, 2018), 84,898 infrared eye crops of 37 people.
#  File names: subject_image_gender_glasses_EYESTATE_reflections_lighting_sensor
#    -> eye state is field 4 (0 = closed, 1 = open).
#  (The earlier baseline CNN read field 5 = reflections by mistake, and the app
#   read the ResNet50 output the wrong way round; both are retired.)
#
#  Evaluation is SUBJECT-WISE: the people in the test set never appear in
#  training, so the score reflects performance on new people.
#  Output: P(eye closed). The app averages it over both eyes as the eye
#  fatigue signal (eye closure / drooping lids is the classic drowsiness sign).
#  Run:  python notebooks/08_train_eye_state_model.py
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import glob
import json
import os
import random

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score)

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA_DIR = os.path.join(BASE, 'data', 'eye_dataset')
MODEL_PATH = os.path.join(BASE, 'models', 'eye_state_cnn.keras')
META_PATH = os.path.join(BASE, 'models', 'eye_state_cnn.json')
RESULTS = os.path.join(BASE, 'results')
IMG = 48
PER_SUBJECT_CLASS = 700           # cap per person and class, so no one person dominates training
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def preprocess(gray):
    """Shared with the app (eye_analysis.py): grayscale -> 48x48 -> histogram equalisation."""
    return cv2.equalizeHist(cv2.resize(gray, (IMG, IMG), interpolation=cv2.INTER_AREA))


# ── 1. Index files and split by person ───────────────────────────────────────
files = glob.glob(os.path.join(DATA_DIR, '**', '*.png'), recursive=True)
records = []
for f in files:
    parts = os.path.basename(f)[:-4].split('_')
    records.append((f, parts[0], 1 - int(parts[4])))        # label 1 = closed
subjects = sorted({r[1] for r in records})
rng = random.Random(SEED)
shuffled = subjects[:]
rng.shuffle(shuffled)
test_subj, val_subj = set(shuffled[:8]), set(shuffled[8:12])
train_subj = set(shuffled[12:])
print(f"{len(records)} images, {len(subjects)} people -> train {len(train_subj)}, val {len(val_subj)}, test {len(test_subj)} people")


def sample(subj_set, cap):
    out = []
    for s in sorted(subj_set):
        for label in (0, 1):
            group = [r for r in records if r[1] == s and r[2] == label]
            rng.shuffle(group)
            out += group[:cap]
    rng.shuffle(out)
    return out


def load(recs):
    X = np.stack([preprocess(cv2.imread(f, cv2.IMREAD_GRAYSCALE)) for f, _, _ in recs])[..., None]
    y = np.array([lab for _, _, lab in recs], dtype='float32')
    return X, y, np.array([s for _, s, _ in recs])


train_recs, val_recs = sample(train_subj, PER_SUBJECT_CLASS), sample(val_subj, 400)
test_recs = sample(test_subj, 10 ** 6)                       # every image of the test people
X_tr, y_tr, _ = load(train_recs)
X_va, y_va, _ = load(val_recs)
X_te, y_te, s_te = load(test_recs)
print(f"train {len(y_tr)} (closed {y_tr.mean():.0%}), val {len(y_va)}, test {len(y_te)} (closed {y_te.mean():.0%})")

# ── 2. Model: small CNN with augmentation for webcam-like variation ──────────
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
for f in (24, 48, 96):
    x = conv_block(x, f)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
out = tf.keras.layers.Dense(1, activation='sigmoid', name='p_closed')(x)
model = tf.keras.Model(inp, out)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
model.summary(print_fn=lambda s: None)
print(f"Parameters: {model.count_params():,}")

hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=12, batch_size=128, verbose=2,
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=3,
                                                             restore_best_weights=True),
                            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=1)])

# ── 3. Evaluate on the unseen people ─────────────────────────────────────────
p = model.predict(X_te, batch_size=256, verbose=0).ravel()
pred = (p >= 0.5).astype(int)
metrics = {
    'test_people': len(test_subj), 'test_images': int(len(y_te)),
    'accuracy': float(accuracy_score(y_te, pred)), 'auc': float(roc_auc_score(y_te, p)),
    'recall_closed': float(recall_score(y_te, pred)), 'precision_closed': float(precision_score(y_te, pred)),
    'recall_open': float(recall_score(1 - y_te, 1 - pred)),
}
per_person = {s: float(accuracy_score(y_te[s_te == s], pred[s_te == s])) for s in sorted(test_subj)}
metrics['per_person_accuracy'] = per_person
print("\nSubject-wise test results (people never seen in training):")
for k in ('accuracy', 'auc', 'recall_closed', 'precision_closed', 'recall_open'):
    print(f"  {k:17s} {metrics[k]:.3f}")
print("  per person:", {k: round(v, 3) for k, v in per_person.items()})

model.save(MODEL_PATH)
with open(META_PATH, 'w', encoding='utf-8') as fh:
    json.dump({'img_size': IMG, 'preprocess': 'grayscale, resize 48x48 (INTER_AREA), cv2.equalizeHist, /255 inside model',
               'output': 'P(eye closed)', 'threshold': 0.5, 'dataset': 'MRL Eye Dataset (Fusek, 2018), eye-state field',
               'split': 'subject-wise: 25 train / 4 validation / 8 test people', 'test_subjects': sorted(test_subj),
               'epochs_trained': len(hist.history['loss']), **metrics}, fh, indent=2)

# ── 4. Charts ────────────────────────────────────────────────────────────────
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
a1.plot(hist.history['accuracy'], label='train')
a1.plot(hist.history['val_accuracy'], label='validation (unseen people)')
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

cm = confusion_matrix(y_te, pred)
fig, ax = plt.subplots(figsize=(4.6, 4))
ax.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
ax.set_xticks([0, 1], ['Open', 'Closed'])
ax.set_yticks([0, 1], ['Open', 'Closed'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f"Unseen people (n = {len(y_te):,}): acc {metrics['accuracy']:.1%}, AUC {metrics['auc']:.3f}",
             fontsize=9, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(RESULTS, 'eye4_state_confusion_matrix.png'), dpi=150)
plt.close(fig)
print(f"\nSaved {MODEL_PATH} and charts in results/")
