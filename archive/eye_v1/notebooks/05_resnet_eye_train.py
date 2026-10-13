# ============================================================
#  RESNET EYE FATIGUE TRANSFER LEARNING
#  Dataset: MRL Eye Dataset (Fusek, R., 2018)
#  Model:   ResNet50 Transfer Learning (Alert vs Fatigued)
#  Student: P R M K Herath | D/BCS/23/0009
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
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE    = 64          # resize all eyes to 64x64
BATCH_SIZE  = 32
EPOCHS      = 10          # 10 epochs is enough for transfer learning
DATA_DIR    = os.path.join(BASE_DIR, '../data/eye_dataset')
MODEL_PATH  = os.path.join(BASE_DIR, '../models/resnet_eye_fatigue_model.h5')
RESULTS_DIR = os.path.join(BASE_DIR, '../results')
os.makedirs(os.path.join(BASE_DIR, '../models'),  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("RESNET50 TRANSFER LEARNING FOR EYE FATIGUE")
print("Dataset: MRL Eye Dataset (Fusek 2018)")
print("=" * 60)

# ── STEP 1: Load and Label Images ────────────────────────────
print("\n[STEP 1] Loading images from MRL dataset...")

all_image_paths = []
all_labels = []

# MRL dataset structure: each subject has subfolders
# Files named like: s0001_00001_0_0_1_0_01_01.png
# The 6th field (index 5) = 0 if eye closed, 1 if open

for root, dirs, files in os.walk(DATA_DIR):
    for fname in files:
        if not fname.lower().endswith(('.png','.jpg','.jpeg')):
            continue
        fpath = os.path.join(root, fname)

        # Determine label from filename
        # MRL format: subject_imageid_gender_glasses_eyestate_reflections_lighting_sensor
        # eye state is at index 4: 0=closed (fatigued), 1=open (alert)
        try:
            parts     = os.path.splitext(fname)[0].split('_')
            eye_state = int(parts[4]) if len(parts) > 4 else -1
        except (IndexError, ValueError):
            eye_state = -1

        # Fallback: use parent folder name (0 or 1)
        if eye_state == -1:
            parent = os.path.basename(root)
            eye_state = int(parent) if parent in ['0','1'] else -1

        if eye_state == -1:
            continue

        # Label: 0=Alert (open eye), 1=Fatigued (closed eye)
        label = 0 if eye_state == 1 else 1
        all_image_paths.append((fpath, label))

print(f"Total images found in folder structure: {len(all_image_paths)}")
if len(all_image_paths) == 0:
    print("\n[ERROR] No images found! Check your data folder structure.")
    exit(1)

# To train quickly on standard hardware, let's take a balanced subset of 10,000 images
alert_paths = [x for x in all_image_paths if x[1] == 0]
fatigue_paths = [x for x in all_image_paths if x[1] == 1]

np.random.seed(42)
subset_size = min(5000, len(alert_paths), len(fatigue_paths))
print(f"Sampling a balanced subset of {subset_size*2} images ({subset_size} per class)...")

selected_alert = [alert_paths[i] for i in np.random.choice(len(alert_paths), subset_size, replace=False)]
selected_fatigue = [fatigue_paths[i] for i in np.random.choice(len(fatigue_paths), subset_size, replace=False)]
selected_paths = selected_alert + selected_fatigue
np.random.shuffle(selected_paths)

images = []
labels = []
loaded = 0

for fpath, label in selected_paths:
    # Read color image (3 channels) for ResNet50
    img = cv2.imread(fpath, cv2.IMREAD_COLOR)
    if img is None:
        continue
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    images.append(img)
    labels.append(label)
    loaded += 1

print(f"[SUCCESS] Balanced subset loaded: {loaded} images")

# Convert to numpy arrays
X = np.array(images, dtype='float32') / 255.0  # normalize to 0-1
y = np.array(labels)

# ── STEP 2: Train/Validation/Test Split ──────────────────────
print("\n[STEP 2] Splitting dataset...")
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

# ── STEP 4: Build ResNet50 Transfer Learning Model ───────────
print("\n[STEP 4] Building ResNet50 Transfer Learning model...")

# Load pre-trained ResNet50 without the top classifier head
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze the base ResNet50 layers
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
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
                 tf.keras.metrics.AUC(name='auc')]
)

model.summary()
total_params = model.count_params()
print(f"\n[SUCCESS] Model built - {total_params:,} parameters")

# ── STEP 5: Train ────────────────────────────────────────────
print("\n[STEP 5] Training ResNet50 (Transfer Learning)...")

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=3,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                    save_best_only=True, verbose=1)
]

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    epochs            = EPOCHS,
    validation_data   = (X_val, y_val),
    callbacks         = callbacks,
    verbose           = 1
)

# ── STEP 6: Evaluate ─────────────────────────────────────────
print("\n[STEP 6] Evaluating ResNet50 on test set...")
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
    'dataset':     'MRL Eye Dataset (ResNet50 fine-tuned)',
    'total_images':loaded
}
joblib.dump(metadata, os.path.join(BASE_DIR, '../models/resnet_eye_metadata.pkl'))

# ── STEP 7: Save Charts ──────────────────────────────────────
print("\n[STEP 7] Saving training charts...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy
axes[0].plot(history.history['accuracy'],     label='Train', color='#3498db', lw=2)
axes[0].plot(history.history['val_accuracy'], label='Val',   color='#e74c3c', lw=2)
axes[0].set_title(f'ResNet50 Accuracy\n(Best Val: {max(history.history["val_accuracy"])*100:.1f}%)')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend(); axes[0].grid(alpha=0.3)

# Loss
axes[1].plot(history.history['loss'],     label='Train', color='#3498db', lw=2)
axes[1].plot(history.history['val_loss'], label='Val',   color='#e74c3c', lw=2)
axes[1].set_title('ResNet50 Loss')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.suptitle('ResNet50 Transfer Learning — Training History', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'resnet_eye_training_history.png'), dpi=150)
plt.close()

# Confusion Matrix
fig, ax = plt.subplots(figsize=(6,5))
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Alert','Fatigued'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'ResNet50 — Confusion Matrix\n'
             f'Test Accuracy: {test_acc*100:.2f}%', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'resnet_eye_confusion_matrix.png'), dpi=150)
plt.close()

print("   [SUCCESS] Saved: results/resnet_eye_training_history.png")
print("   [SUCCESS] Saved: results/resnet_eye_confusion_matrix.png")

print("\n" + "=" * 60)
print("RESNET TRAINING COMPLETE")
print("=" * 60)
