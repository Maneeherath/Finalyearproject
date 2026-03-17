# ============================================================
#  STAGE 2: DATA PREPROCESSING
#  Project: Computational Modeling of Cognitive Performance
#           and Gaming Behavior
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================
#
#  WHY DO WE PREPROCESS?
#  Machine learning models only understand numbers, not text.
#  Also, data from surveys can have missing values or columns
#  on very different scales (e.g. 0-100 vs 200-600).
#  Preprocessing fixes all of this.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ── STEP 1: Load the dataset ─────────────────────────────────
print("=" * 55)
print("STEP 1: Loading Dataset")
print("=" * 55)

df = pd.read_csv('../data/gaming_dataset.csv')

print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns")
print("\nColumn names:")
for col in df.columns:
    print(f"   • {col}")

# ── STEP 2: Check for missing values ────────────────────────
print("\n" + "=" * 55)
print("STEP 2: Checking for Missing Values")
print("=" * 55)

missing = df.isnull().sum()
if missing.sum() == 0:
    print("✅ No missing values found! Dataset is clean.")
else:
    print("⚠️  Missing values found:")
    print(missing[missing > 0])
    # Fill numeric columns with mean, text columns with mode
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)
    print("✅ Missing values filled.")

# ── STEP 3: Drop columns not needed for modeling ────────────
print("\n" + "=" * 55)
print("STEP 3: Dropping Non-Useful Columns")
print("=" * 55)

# student_id is just an ID number, not a feature
df.drop(columns=['student_id'], inplace=True)
print("✅ Dropped: student_id (just an identifier, not useful for ML)")

# ── STEP 4: Encode categorical (text) columns ───────────────
print("\n" + "=" * 55)
print("STEP 4: Converting Text Columns to Numbers")
print("=" * 55)

# gender: Male=0, Female=1, Other=2
le_gender = LabelEncoder()
df['gender'] = le_gender.fit_transform(df['gender'])
print(f"✅ gender encoded  → {dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))}")

# degree_program
le_degree = LabelEncoder()
df['degree_program'] = le_degree.fit_transform(df['degree_program'])
print(f"✅ degree_program encoded → {dict(zip(le_degree.classes_, le_degree.transform(le_degree.classes_)))}")

# game_type: use one-hot encoding (creates a separate 0/1 column per genre)
# This is better than label encoding for categories with no natural order
df = pd.get_dummies(df, columns=['game_type'], prefix='game')
game_cols = [c for c in df.columns if c.startswith('game_')]
print(f"✅ game_type → one-hot encoded into: {game_cols}")

# addiction_level: Low=0, Medium=1, High=2
addiction_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['addiction_level'] = df['addiction_level'].map(addiction_map)
print(f"✅ addiction_level encoded → {addiction_map}")

# ── STEP 5: Normalize numeric features ──────────────────────
print("\n" + "=" * 55)
print("STEP 5: Normalizing Numeric Features (scale to 0-1)")
print("=" * 55)

# WHY? If session_hours is 0-6 and reaction_time is 180-600,
# the model will unfairly give more weight to reaction_time.
# Scaling puts everyone on equal footing.

cols_to_scale = [
    'age', 'year_of_study', 'years_gaming',
    'sessions_per_week', 'session_hours', 'total_hours_week',
    'attention_score', 'memory_score', 'reaction_time'
]

scaler = MinMaxScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
print(f"✅ Scaled {len(cols_to_scale)} columns to range [0, 1]")

# ── STEP 6: Final check ──────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 6: Final Dataset Check")
print("=" * 55)
print(f"Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nAddiction level distribution:")
counts = df['addiction_level'].value_counts().sort_index()
labels = {0: 'Low', 1: 'Medium', 2: 'High'}
for k, v in counts.items():
    print(f"   {labels[k]:8s} ({k}): {v} students")

# ── STEP 7: Save the cleaned dataset ────────────────────────
df.to_csv('../data/gaming_dataset_cleaned.csv', index=False)
print("\n✅ Cleaned dataset saved as: gaming_dataset_cleaned.csv")
print("   This file will be used in the next stage (Model Building)")
print("=" * 55)

