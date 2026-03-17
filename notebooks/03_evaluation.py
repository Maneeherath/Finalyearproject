# ============================================================
#  STAGE 4: MODEL EVALUATION & VISUALIZATION
#  Project: Computational Modeling of Cognitive Performance
#           and Gaming Behavior
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import joblib
import os

os.makedirs('../results', exist_ok=True)

print("=" * 55)
print("STAGE 4: MODEL EVALUATION & VISUALIZATION")
print("=" * 55)

# ── Load data and model ──────────────────────────────────────
df = pd.read_csv('../data/gaming_dataset_cleaned.csv')
X = df.drop(columns=['addiction_level'])
y = df['addiction_level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = joblib.load('../models/best_model.pkl')
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model loaded: Random Forest")
print(f"✅ Test Accuracy: {accuracy*100:.2f}%")

# ── CHART 1: Confusion Matrix ────────────────────────────────
print("\n📊 Generating Chart 1: Confusion Matrix...")
fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Low', 'Medium', 'High'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix — Random Forest\n'
             '(How well the model predicted addiction level)',
             fontsize=12, pad=15)
plt.tight_layout()
plt.savefig('../results/chart1_confusion_matrix.png', dpi=150)
plt.close()
print("   ✅ Saved: results/chart1_confusion_matrix.png")

# ── CHART 2: Feature Importance ─────────────────────────────
print("📊 Generating Chart 2: Feature Importance...")
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True).tail(12)

fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#e74c3c' if v > importances.mean() else '#3498db'
          for v in importances.values]
importances.plot(kind='barh', ax=ax, color=colors)
ax.set_title('Top 12 Features Influencing Addiction Prediction',
             fontsize=12, pad=15)
ax.set_xlabel('Importance Score')
ax.axvline(importances.mean(), color='gray', linestyle='--',
           alpha=0.7, label='Average')
ax.legend()
plt.tight_layout()
plt.savefig('../results/chart2_feature_importance.png', dpi=150)
plt.close()
print("   ✅ Saved: results/chart2_feature_importance.png")

# ── CHART 3: Addiction Level Distribution ───────────────────
print("📊 Generating Chart 3: Addiction Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

labels = ['Low', 'Medium', 'High']
colors = ['#2ecc71', '#f39c12', '#e74c3c']

# Actual
actual_counts = pd.Series(y_test).map({0:'Low',1:'Medium',2:'High'}).value_counts()
actual_counts = actual_counts.reindex(labels)
axes[0].bar(labels, actual_counts.values, color=colors, edgecolor='white', linewidth=1.5)
axes[0].set_title('Actual Addiction Levels\n(Test Set)', fontsize=11)
axes[0].set_ylabel('Number of Students')
for i, v in enumerate(actual_counts.values):
    axes[0].text(i, v + 0.3, str(v), ha='center', fontweight='bold')

# Predicted
pred_counts = pd.Series(y_pred).map({0:'Low',1:'Medium',2:'High'}).value_counts()
pred_counts = pred_counts.reindex(labels)
axes[1].bar(labels, pred_counts.values, color=colors, edgecolor='white',
            linewidth=1.5, alpha=0.8)
axes[1].set_title('Predicted Addiction Levels\n(Model Output)', fontsize=11)
axes[1].set_ylabel('Number of Students')
for i, v in enumerate(pred_counts.values):
    axes[1].text(i, v + 0.3, str(v), ha='center', fontweight='bold')

plt.suptitle('Actual vs Predicted Addiction Level Distribution',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('../results/chart3_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: results/chart3_distribution.png")

# ── CHART 4: Gaming Hours vs Attention Score ────────────────
print("📊 Generating Chart 4: Gaming Hours vs Attention Score...")
raw = pd.read_csv('../data/gaming_dataset.csv')
colors_map = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}

fig, ax = plt.subplots(figsize=(8, 5))
for level, color in colors_map.items():
    subset = raw[raw['addiction_level'] == level]
    ax.scatter(subset['total_hours_week'], subset['attention_score'],
               c=color, label=level, alpha=0.7, edgecolors='white', s=60)

ax.set_xlabel('Total Gaming Hours per Week', fontsize=11)
ax.set_ylabel('Attention Score', fontsize=11)
ax.set_title('Gaming Hours vs Attention Score\n(colored by Addiction Level)',
             fontsize=12)
ax.legend(title='Addiction Level')
plt.tight_layout()
plt.savefig('../results/chart4_hours_vs_attention.png', dpi=150)
plt.close()
print("   ✅ Saved: results/chart4_hours_vs_attention.png")

# ── Final Summary ────────────────────────────────────────────
print("\n" + "=" * 55)
print("EVALUATION SUMMARY")
print("=" * 55)
print(f"\n  Best Model   : Random Forest")
print(f"  Accuracy     : {accuracy*100:.2f}%")
print(f"\n  Detailed Report:")
print(classification_report(y_test, y_pred,
      target_names=['Low', 'Medium', 'High']))
print("📁 All charts saved in: results/ folder")
print("🎉 Evaluation complete! Ready for Stage 5: Web App")
print("=" * 55)