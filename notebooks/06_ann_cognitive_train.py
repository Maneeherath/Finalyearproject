# ============================================================
#  ARTIFICIAL NEURAL NETWORK (ANN) FOR COGNITIVE PREDICTION
#  Dataset: Gaming Behavior and Cognitive Performance Dataset
#  Model:   Keras MLP Multi-Output Regressor
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

# Robust Path Resolution
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE_DIR, '../data/gaming_dataset_v2.csv')
MODEL_DIR    = os.path.join(BASE_DIR, '../models')
RESULTS_DIR  = os.path.join(BASE_DIR, '../results')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("ANN FOR COGNITIVE PERFORMANCE PREDICTION")
print("=" * 60)

# ── STEP 1: Load and Preprocess Data ─────────────────────────
print(f"[STEP 1] Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"[SUCCESS] Loaded dataset: {df.shape[0]} rows")

# Label encoding for categoricals
le_gender = LabelEncoder()
le_degree = LabelEncoder()
df['gender']         = le_gender.fit_transform(df['gender'])
df['degree_program'] = le_degree.fit_transform(df['degree_program'])

# One-hot encoding for game_type
df = pd.get_dummies(df, columns=['game_type'], prefix='game')

# Drop student_id
if 'student_id' in df.columns:
    df.drop(columns=['student_id'], inplace=True)

# Normalize features and targets
cols_to_scale = ['age','year_of_study','years_gaming','sessions_per_week',
                 'session_hours','total_hours_week','sleep_hours','gpa',
                 'attention_score','memory_score','reaction_time']

scaler = MinMaxScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# Save the scaler for use in deployment
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

# Define features (gaming, demographic, lifestyle)
reg_features = [c for c in df.columns if c not in
                ['addiction_level','attention_score','memory_score','reaction_time']]

X = df[reg_features]
y = df[['attention_score', 'memory_score']] # Multi-output target

print(f"Features list ({len(reg_features)} features):")
for f in reg_features:
    print(f"  - {f}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── STEP 2: Build Artificial Neural Network (ANN) ───────────
print("\n[STEP 2] Building Keras ANN Regressor model...")

model = models.Sequential([
    layers.Input(shape=(len(reg_features),)),
    
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(16, activation='relu'),
    layers.Dense(2, activation='sigmoid') # Sigmoid since targets scaled [0,1]
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
    loss='mse',
    metrics=['mae']
)

model.summary()

# ── STEP 3: Train Model ──────────────────────────────────────
print("\n[STEP 3] Training Keras ANN...")

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.15,
    callbacks=[early_stopping],
    verbose=1
)

# ── STEP 4: Evaluate Model ───────────────────────────────────
print("\n[STEP 4] Evaluating model on test set...")
y_pred = model.predict(X_test)

# Calculate metrics for Attention Score
att_r2 = r2_score(y_test.iloc[:, 0], y_pred[:, 0])
att_rmse = np.sqrt(mean_squared_error(y_test.iloc[:, 0], y_pred[:, 0]))

# Calculate metrics for Memory Score
mem_r2 = r2_score(y_test.iloc[:, 1], y_pred[:, 1])
mem_rmse = np.sqrt(mean_squared_error(y_test.iloc[:, 1], y_pred[:, 1]))

print(f"\n  Attention Score Predictions:")
print(f"    R2 Score : {att_r2:.4f}")
print(f"    RMSE     : {att_rmse:.4f}")

print(f"  Memory Score Predictions:")
print(f"    R2 Score : {mem_r2:.4f}")
print(f"    RMSE     : {mem_rmse:.4f}")

# Save ANN model
model_save_path = os.path.join(MODEL_DIR, 'ann_cognitive_model.h5')
model.save(model_save_path)
print(f"\n[SUCCESS] Saved Keras ANN: {model_save_path}")

# Save evaluation metadata
metadata = {
    'att_r2':   round(float(att_r2), 4),
    'att_rmse': round(float(att_rmse), 4),
    'mem_r2':   round(float(mem_r2), 4),
    'mem_rmse': round(float(mem_rmse), 4)
}
metadata_save_path = os.path.join(MODEL_DIR, 'ann_cognitive_metadata.pkl')
joblib.dump(metadata, metadata_save_path)
print(f"[SUCCESS] Saved Metadata: {metadata_save_path}")
print("=" * 60)
