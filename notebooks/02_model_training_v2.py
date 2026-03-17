# ============================================================
#  ENHANCED MODEL TRAINING v2
#  Includes: Cross-Validation, Regression Models, Feature Importance
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_squared_error, r2_score)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib, os

os.makedirs('../models', exist_ok=True)
os.makedirs('../results', exist_ok=True)

print("=" * 60)
print("ENHANCED MODEL TRAINING v2")
print("=" * 60)

# ── STEP 1: Load & Preprocess Enhanced Dataset ───────────────
df = pd.read_csv('../data/gaming_dataset_v2.csv')
print(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# Encode categoricals
le_gender = LabelEncoder()
le_degree = LabelEncoder()
df['gender']         = le_gender.fit_transform(df['gender'])
df['degree_program'] = le_degree.fit_transform(df['degree_program'])
df = pd.get_dummies(df, columns=['game_type'], prefix='game')
addiction_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['addiction_level'] = df['addiction_level'].map(addiction_map)
df.drop(columns=['student_id'], inplace=True)

# Scale
cols_to_scale = ['age','year_of_study','years_gaming','sessions_per_week',
                 'session_hours','total_hours_week','sleep_hours','gpa',
                 'attention_score','memory_score','reaction_time']
scaler = MinMaxScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
joblib.dump(scaler, '../models/scaler.pkl')

# ── STEP 2: Classification — Predict Addiction Level ─────────
print("\n" + "=" * 60)
print("PART A: CLASSIFICATION — Predicting Addiction Level")
print("=" * 60)

# Features: exclude cognitive scores (not known at prediction time)
clf_features = [c for c in df.columns if c not in
                ['addiction_level','attention_score','memory_score','reaction_time']]
X_clf = df[clf_features]
y_clf = df['addiction_level']

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Models to compare
classifiers = {
    "Decision Tree"        : DecisionTreeClassifier(random_state=42),
    "Random Forest"        : RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting"    : GradientBoostingClassifier(n_estimators=200, random_state=42),
    "Support Vector Machine": SVC(kernel='rbf', probability=True, random_state=42)
}

clf_results = {}
print(f"\n{'Model':<28} {'Test Acc':>10} {'CV Mean':>10} {'CV Std':>10}")
print("-" * 62)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    cv_scores = cross_val_score(clf, X_clf, y_clf, cv=cv, scoring='accuracy')
    clf_results[name] = {
        'model': clf, 'test_acc': test_acc,
        'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
        'cv_scores': cv_scores
    }
    print(f"{name:<28} {test_acc*100:>9.2f}% {cv_scores.mean()*100:>9.2f}% {cv_scores.std()*100:>9.2f}%")

best_clf_name = max(clf_results, key=lambda x: clf_results[x]['cv_mean'])
best_clf = clf_results[best_clf_name]['model']
joblib.dump(best_clf, '../models/best_model.pkl')
joblib.dump(clf_features, '../models/feature_names.pkl')
print(f"\n✅ Best Classifier: {best_clf_name} "
      f"(CV: {clf_results[best_clf_name]['cv_mean']*100:.2f}%)")

# ── STEP 3: Regression — Predict Attention & Memory Scores ───
print("\n" + "=" * 60)
print("PART B: REGRESSION — Predicting Cognitive Scores")
print("=" * 60)

reg_features = [c for c in df.columns if c not in
                ['addiction_level','attention_score','memory_score','reaction_time']]
X_reg = df[reg_features]

for target_name in ['attention_score', 'memory_score']:
    y_reg = df[target_name]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    reg_model = RandomForestRegressor(n_estimators=200, random_state=42)
    reg_model.fit(Xr_train, yr_train)
    y_pred_reg = reg_model.predict(Xr_test)
    rmse = np.sqrt(mean_squared_error(yr_test, y_pred_reg))
    r2   = r2_score(yr_test, y_pred_reg)
    joblib.dump(reg_model, f'../models/reg_{target_name}.pkl')
    print(f"\n  {target_name}:")
    print(f"    R² Score : {r2:.4f}  (1.0 = perfect)")
    print(f"    RMSE     : {rmse:.4f} (lower is better)")
    print(f"    ✅ Saved : models/reg_{target_name}.pkl")

# ── CHART A: Cross-Validation Comparison ─────────────────────
print("\n📊 Generating charts...")
fig, ax = plt.subplots(figsize=(10, 5))
names    = list(clf_results.keys())
cv_means = [clf_results[n]['cv_mean']*100 for n in names]
cv_stds  = [clf_results[n]['cv_std']*100  for n in names]
colors   = ['#e74c3c' if n == best_clf_name else '#3498db' for n in names]

bars = ax.bar(names, cv_means, yerr=cv_stds, capsize=6,
              color=colors, edgecolor='white', linewidth=1.5, width=0.5)
ax.set_ylabel('Cross-Validation Accuracy (%)', fontsize=11)
ax.set_title('5-Fold Cross-Validation — Model Comparison\n'
             '(Error bars show standard deviation across folds)', fontsize=12)
ax.set_ylim(0, 105)
for bar, mean, std in zip(bars, cv_means, cv_stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
            f'{mean:.1f}%', ha='center', fontsize=10, fontweight='bold')

best_patch = mpatches.Patch(color='#e74c3c', label=f'Best: {best_clf_name}')
ax.legend(handles=[best_patch], fontsize=10)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('../results/chartA_cross_validation.png', dpi=150)
plt.close()
print("   ✅ Saved: results/chartA_cross_validation.png")

# ── CHART B: Feature Importance (Top 15) ─────────────────────
if hasattr(best_clf, 'feature_importances_'):
    importances = pd.Series(best_clf.feature_importances_,
                            index=clf_features).sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors_fi = ['#e74c3c' if v >= importances.quantile(0.75) else
                 '#f39c12' if v >= importances.quantile(0.50) else '#3498db'
                 for v in importances.values]
    importances.plot(kind='barh', ax=ax, color=colors_fi)
    ax.set_title('Top 15 Features — Addiction Level Prediction\n'
                 '(Red = High importance, Blue = Lower importance)', fontsize=12)
    ax.set_xlabel('Feature Importance Score')
    plt.tight_layout()
    plt.savefig('../results/chartB_feature_importance.png', dpi=150)
    plt.close()
    print("   ✅ Saved: results/chartB_feature_importance.png")

# ── CHART C: CV Score Distribution (Box Plot) ────────────────
fig, ax = plt.subplots(figsize=(9, 5))
cv_data = [clf_results[n]['cv_scores']*100 for n in names]
bp = ax.boxplot(cv_data, labels=names, patch_artist=True, notch=False)
palette = ['#e74c3c','#2ecc71','#3498db','#9b59b6']
for patch, color in zip(bp['boxes'], palette):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Accuracy per Fold (%)', fontsize=11)
ax.set_title('Cross-Validation Score Distribution per Model\n'
             '(Box shows spread across 5 folds)', fontsize=12)
ax.set_ylim(50, 105)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('../results/chartC_cv_boxplot.png', dpi=150)
plt.close()
print("   ✅ Saved: results/chartC_cv_boxplot.png")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"  Best Classifier : {best_clf_name}")
print(f"  CV Accuracy     : {clf_results[best_clf_name]['cv_mean']*100:.2f}% "
      f"± {clf_results[best_clf_name]['cv_std']*100:.2f}%")
print(f"  Test Accuracy   : {clf_results[best_clf_name]['test_acc']*100:.2f}%")
print(f"\n  Models saved    : models/")
print(f"  Charts saved    : results/")
print("=" * 60)