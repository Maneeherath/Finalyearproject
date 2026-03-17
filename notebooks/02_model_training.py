# ============================================================
#  STAGE 3: MODEL TRAINING
#  Project: Computational Modeling of Cognitive Performance
#           and Gaming Behavior
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

print("=" * 55)
print("STAGE 3: MODEL TRAINING")
print("=" * 55)

# ── STEP 1: Load cleaned data ────────────────────────────────
df = pd.read_csv('../data/gaming_dataset_cleaned.csv')
print(f"✅ Loaded cleaned dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ── STEP 2: Split features and target ───────────────────────
# X = input features (everything except addiction_level)
# y = what we want to predict (addiction_level)
X = df.drop(columns=['addiction_level'])
y = df['addiction_level']

print(f"\n✅ Features (X): {X.shape[1]} columns")
print(f"✅ Target  (y): addiction_level (0=Low, 1=Medium, 2=High)")

# ── STEP 3: Split into training and testing sets ─────────────
# 70% for training, 30% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\n✅ Training set : {X_train.shape[0]} students")
print(f"✅ Testing set  : {X_test.shape[0]} students")

# ── STEP 4: Define models to compare ────────────────────────
print("\n" + "=" * 55)
print("STEP 4: Training & Comparing Models")
print("=" * 55)

models = {
    "Decision Tree"        : DecisionTreeClassifier(random_state=42),
    "Random Forest"        : RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting"    : GradientBoostingClassifier(random_state=42),
    "Support Vector Machine": SVC(kernel='rbf', random_state=42)
}

results = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Test the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    results[name] = {"model": model, "accuracy": accuracy, "predictions": predictions}

    print(f"\n{'─'*45}")
    print(f"  {name}")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"{'─'*45}")
    print(classification_report(y_test, predictions,
          target_names=['Low', 'Medium', 'High']))

# ── STEP 5: Pick the best model ─────────────────────────────
print("\n" + "=" * 55)
print("STEP 5: Model Comparison Summary")
print("=" * 55)

best_name = max(results, key=lambda x: results[x]['accuracy'])

print(f"\n{'Model':<30} {'Accuracy':>10}")
print("-" * 42)
for name, res in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
    marker = " ← BEST ✅" if name == best_name else ""
    print(f"{name:<30} {res['accuracy']*100:>9.2f}%{marker}")

# ── STEP 6: Save the best model ─────────────────────────────
print("\n" + "=" * 55)
print("STEP 6: Saving Best Model")
print("=" * 55)


best_model = results[best_name]['model']
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(list(X.columns), 'feature_names.pkl')

print(f"✅ Best model  : {best_name}")
print(f"✅ Saved to    : ../models/best_model.pkl")
print(f"✅ Features    : ../models/feature_names.pkl")
print("\n🎉 Model training complete! Ready for Stage 4: Evaluation")
print("=" * 55)