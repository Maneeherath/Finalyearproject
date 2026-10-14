# ============================================================
#  EYE FATIGUE MODEL TRAINING
#  Dataset: MRL Eye Dataset (Fusek, R., 2018)
#  Source:  http://mrl.cs.vsb.cz/eyedataset
#  Model:   CNN Binary Classifier (Alert vs Fatigued)
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================
#
#  HOW TO GET THE DATASET:
#  1. Go to: http://mrl.cs.vsb.cz/eyedataset
#  2. Download: mrlEyes_2018_01.zip (or any subject folders)
#  3. Extract to: data/eye_dataset/
#
#  Expected folder structure after extraction:
#  data/eye_dataset/
#  ├── s0001/                  ← subject folders
#  │   ├── 0/                  ← label 0 = closed/fatigued
#  │   │   ├── image001.png
#  │   │   └── ...
#  │   └── 1/                  ← label 1 = open/alert
#  │       ├── image001.png
#  │       └── ...
#  ├── s0002/
#  └── ...
#
#  NOTE: MRL dataset uses 0=closed, 1=open in filenames
#  We map: open=Alert(0), closed=Fatigued(1)
# ============================================================

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import joblib

# ── Configuration ────────────────────────────────────────────
IMG_SIZE    = 64          # resize all eyes to 64x64
BATCH_SIZE  = 32
EPOCHS      = 20
DATA_DIR    = '../data/eye_dataset'
MODEL_PATH  = '../models/eye_fatigue_model.h5'
os.makedirs('../models',  exist_ok=True)
os.makedirs('../results', exist_ok=True)

print("=" * 60)
print("EYE FATIGUE MODEL TRAINING")
print("Dataset: MRL Eye Dataset (Fusek 2018)")
print("=" * 60)

# ── STEP 1: Load and Label Images ────────────────────────────
print("\n📂 Loading images from MRL dataset...")

images = []
labels = []
loaded = 0
skipped= 0

# MRL dataset structure: each subject has subfolders
# Files named like: s0001_00001_0_0_1_0_01_01.png
# The 6th field (index 5) = 0 if eye closed, 1 if open
# We use folder structure OR filename to determine label

for root, dirs, files in os.walk(DATA_DIR):
    for fname in files:
        if not fname.lower().endswith(('.png','.jpg','.jpeg')):
            continue
        fpath = os.path.join(root, fname)

        # ── Determine label from filename ───────────────────
        # MRL filename: s0001_00001_0_0_1_0_01_01.png
        # Field index 5 (0-indexed) = eye state: 0=closed, 1=open
        try:
            parts     = os.path.splitext(fname)[0].split('_')
            eye_state = int(parts[5]) if len(parts) > 5 else -1
        except (IndexError, ValueError):
            eye_state = -1

        # Fallback: use parent folder name (0 or 1)
        if eye_state == -1:
            parent = os.path.basename(root)
            eye_state = int(parent) if parent in ['0','1'] else -1

        if eye_state == -1:
            skipped += 1
            continue

        # Label: 0=Alert (open eye), 1=Fatigued (closed eye)
        label = 0 if eye_state == 1 else 1

        # Load and preprocess image
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.equalizeHist(img)   # enhance contrast
        images.append(img)
        labels.append(label)
        loaded += 1

        if loaded % 5000 == 0:
            print(f"   Loaded {loaded} images...")

print(f"\n✅ Total loaded : {loaded} images")
print(f"   Skipped      : {skipped} images")
if loaded == 0:
    print("\n❌ No images found! Check your data folder structure.")
    print(f"   Expected: {DATA_DIR}/s0001/... or {DATA_DIR}/0/ and {DATA_DIR}/1/")
    exit(1)

# Convert to numpy arrays
X = np.array(images, dtype='float32') / 255.0  # normalize to 0-1
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)         # add channel dimension
y = np.array(labels)

alert_count   = np.sum(y == 0)
fatigue_count = np.sum(y == 1)
print(f"\n   Alert (open)    : {alert_count} images")
print(f"   Fatigued (closed): {fatigue_count} images")
print(f"   Class balance    : {alert_count/loaded*100:.1f}% / {fatigue_count/loaded*100:.1f}%")

# ── STEP 2: Train/Validation/Test Split ──────────────────────
print("\n📊 Splitting dataset...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
print(f"   Training   : {len(X_train)} images")
print(f"   Validation : {len(X_val)} images")
print(f"   Testing    : {len(X_test)} images")

# ── STEP 3: Data Augmentation ────────────────────────────────
datagen = ImageDataGenerator(
    rotation_range    = 10,
    width_shift_range = 0.1,
    height_shift_range= 0.1,
    horizontal_flip   = True,
    zoom_range        = 0.1
)
datagen.fit(X_train)

# ── STEP 4: Build CNN Model ──────────────────────────────────
print("\n🏗️  Building CNN model...")

model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3,3), activation='relu', padding='same',
                  input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    # Block 2
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    # Block 3
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    # Classifier head
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')   # binary output
])

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    loss      = 'binary_crossentropy',
    metrics   = ['accuracy',
                 tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
)

model.summary()
total_params = model.count_params()
print(f"\n✅ Model built — {total_params:,} parameters")

# ── STEP 5: Train ────────────────────────────────────────────
print("\n🚀 Training...")

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                    save_best_only=True, verbose=1)
]

# Handle class imbalance
total   = len(y_train)
w0      = total / (2 * np.sum(y_train==0))  # Alert weight
w1      = total / (2 * np.sum(y_train==1))  # Fatigue weight
cw      = {0: w0, 1: w1}

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    epochs            = EPOCHS,
    validation_data   = (X_val, y_val),
    class_weight      = cw,
    callbacks         = callbacks,
    verbose           = 1
)

# ── STEP 6: Evaluate ─────────────────────────────────────────
print("\n📋 Evaluating on test set...")
test_results = model.evaluate(X_test, y_test, verbose=0)
test_acc     = test_results[1]
test_auc     = test_results[2]

y_pred_prob = model.predict(X_test, verbose=0).flatten()
y_pred      = (y_pred_prob >= 0.5).astype(int)

print(f"\n  Test Accuracy : {test_acc*100:.2f}%")
print(f"  Test AUC      : {test_auc:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Alert (Open)', 'Fatigued (Closed)']))

# Save metadata for app use
metadata = {
    'img_size':    IMG_SIZE,
    'test_acc':    round(test_acc*100, 2),
    'test_auc':    round(float(test_auc), 4),
    'labels':      {0: 'Alert', 1: 'Fatigued'},
    'threshold':   0.5,
    'dataset':     'MRL Eye Dataset (Fusek 2018)',
    'total_images':loaded
}
joblib.dump(metadata, '../models/eye_model_metadata.pkl')

# ── STEP 7: Training Charts ──────────────────────────────────
print("\n📊 Saving training charts...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy
axes[0].plot(history.history['accuracy'],     label='Train', color='#3498db', lw=2)
axes[0].plot(history.history['val_accuracy'], label='Val',   color='#e74c3c', lw=2)
axes[0].set_title(f'CNN Training Accuracy\n(Best Val: {max(history.history["val_accuracy"])*100:.1f}%)')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend(); axes[0].grid(alpha=0.3)

# Loss
axes[1].plot(history.history['loss'],     label='Train', color='#3498db', lw=2)
axes[1].plot(history.history['val_loss'], label='Val',   color='#e74c3c', lw=2)
axes[1].set_title('CNN Training Loss')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.suptitle('Eye Fatigue CNN — Training History', fontsize=13)
plt.tight_layout()
plt.savefig('../results/eye1_training_history.png', dpi=150)
plt.close()

# Confusion Matrix
fig, ax = plt.subplots(figsize=(6,5))
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Alert','Fatigued'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Eye Fatigue CNN — Confusion Matrix\n'
             f'Test Accuracy: {test_acc*100:.2f}%', fontsize=12)
plt.tight_layout()
plt.savefig('../results/eye2_confusion_matrix.png', dpi=150)
plt.close()

print("   ✅ Saved: results/eye1_training_history.png")
print("   ✅ Saved: results/eye2_confusion_matrix.png")

print("\n" + "=" * 60)
print("EYE MODEL TRAINING COMPLETE")
print("=" * 60)
print(f"  Model saved   : {MODEL_PATH}")
print(f"  Test Accuracy : {test_acc*100:.2f}%")
print(f"  Test AUC      : {test_auc:.4f}")
print(f"  Dataset       : MRL Eye Dataset — {loaded} images")
print(f"\n  ✅ Ready to use in the app!")
print("=" * 60)