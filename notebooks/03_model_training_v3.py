# ============================================================
#  ENHANCED MODEL TRAINING v3 — Maximum Accuracy Push
#  Adds: XGBoost, LightGBM, Extra Trees, AdaBoost, MLP,
#        Voting Ensemble, Stacking Classifier + GridSearch
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, GridSearchCV)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier, AdaBoostClassifier,
                               VotingClassifier, StackingClassifier,
                               RandomForestRegressor)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_squared_error, r2_score, confusion_matrix)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib, os, time

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARNING] XGBoost not installed - skipping")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("[WARNING] LightGBM not installed - skipping")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '../models')
RESULTS_DIR = os.path.join(BASE_DIR, '../results')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("  ENHANCED MODEL TRAINING v3 — Maximum Accuracy Push")
print("  8+ Models | Hyperparameter Tuning | Ensemble Methods")
print("=" * 70)

# ── STEP 1: Load & Preprocess ────────────────────────────────
DATA_PATH = os.path.join(BASE_DIR, '../data/gaming_dataset_v2.csv')
df = pd.read_csv(DATA_PATH)
print(f"\n[OK] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")

le_gender = LabelEncoder()
le_degree = LabelEncoder()
df['gender']         = le_gender.fit_transform(df['gender'])
df['degree_program'] = le_degree.fit_transform(df['degree_program'])
df = pd.get_dummies(df, columns=['game_type'], prefix='game')
addiction_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['addiction_level'] = df['addiction_level'].map(addiction_map)
df.drop(columns=['student_id'], inplace=True)

cols_to_scale = ['age','year_of_study','years_gaming','sessions_per_week',
                 'session_hours','total_hours_week','sleep_hours','gpa',
                 'attention_score','memory_score','reaction_time']
scaler = MinMaxScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

# ── STEP 2: Classification ───────────────────────────────────
print("\n" + "=" * 70)
print("  PART A: CLASSIFICATION — 8+ Model Comparison")
print("=" * 70)

clf_features = [c for c in df.columns if c not in
                ['addiction_level','attention_score','memory_score','reaction_time']]
X_clf = df[clf_features]
y_clf = df['addiction_level']

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Define all classifiers ───────────────────────────────────
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
    "MLP Neural Net": MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                                     max_iter=500, random_state=42,
                                     early_stopping=True, validation_fraction=0.15),
}

if HAS_XGB:
    classifiers["XGBoost"] = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=42, verbosity=0
    )

if HAS_LGBM:
    classifiers["LightGBM"] = LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    )

# ── Train & Evaluate All Models ──────────────────────────────
clf_results = {}
print(f"\n{'Model':<22} {'Test Acc':>10} {'CV Mean':>10} {'CV Std':>10}  {'Time':>8}")
print("-" * 68)

for name, clf in classifiers.items():
    start = time.time()
    clf.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    cv_scores = cross_val_score(clf, X_clf, y_clf, cv=cv, scoring='accuracy')
    elapsed = time.time() - start
    
    clf_results[name] = {
        'model': clf, 'test_acc': test_acc,
        'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
        'cv_scores': cv_scores, 'time': elapsed
    }
    print(f"{name:<22} {test_acc*100:>9.2f}% {cv_scores.mean()*100:>9.2f}% "
          f"{cv_scores.std()*100:>9.2f}%  {elapsed:>7.1f}s")

# ── STEP 3: Hyperparameter Tuning on Top 3 Models ────────────
print("\n" + "=" * 70)
print("  PART B: HYPERPARAMETER TUNING (GridSearchCV)")
print("=" * 70)

# Sort by CV mean and pick top 3
top3 = sorted(clf_results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)[:3]
print(f"\n  Tuning top 3 models: {', '.join([n for n,_ in top3])}")

tuned_results = {}

# Define tuning grids
param_grids = {
    "Random Forest": {
        'n_estimators': [200, 300, 400],
        'max_depth': [None, 10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': [None, 'balanced']
    },
    "Gradient Boosting": {
        'n_estimators': [200, 300, 400],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 1.0],
        'min_samples_split': [2, 5]
    },
    "Extra Trees": {
        'n_estimators': [200, 300, 400],
        'max_depth': [None, 10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': [None, 'balanced']
    },
    "XGBoost": {
        'n_estimators': [200, 300, 400],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 1.5]
    },
    "LightGBM": {
        'n_estimators': [200, 300, 400],
        'max_depth': [4, 6, 8, -1],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'num_leaves': [31, 50, 80]
    },
    "SVM (RBF)": {
        'C': [0.5, 1.0, 5.0, 10.0],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf']
    },
    "MLP Neural Net": {
        'hidden_layer_sizes': [(128,64,32), (256,128,64), (64,32)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.005, 0.01]
    },
    "AdaBoost": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.5, 1.0]
    },
    "Decision Tree": {
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced']
    }
}

for name, res in top3:
    print(f"\n  [Tuning] Tuning: {name}...")
    
    grid = param_grids.get(name, None)
    if grid is None:
        print(f"    No grid defined — keeping default")
        tuned_results[name] = res
        continue
    
    base_model = res['model'].__class__
    
    # Reconstruct base with random_state
    if name == "XGBoost":
        base = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',
                             random_state=42, verbosity=0)
    elif name == "LightGBM":
        base = LGBMClassifier(random_state=42, verbose=-1)
    elif name == "MLP Neural Net":
        base = MLPClassifier(max_iter=500, random_state=42,
                             early_stopping=True, validation_fraction=0.15)
    elif name == "AdaBoost":
        base = AdaBoostClassifier(random_state=42)
    else:
        base = base_model(random_state=42)
    
    gs = GridSearchCV(base, grid, cv=cv, scoring='accuracy',
                      n_jobs=1, verbose=0, refit=True)
    gs.fit(X_clf, y_clf)
    
    best_model = gs.best_estimator_
    best_cv = gs.best_score_
    best_model.fit(X_train, y_train)
    best_test_acc = accuracy_score(y_test, best_model.predict(X_test))
    
    # Re-run full CV with best estimator for std
    cv_scores_tuned = cross_val_score(best_model, X_clf, y_clf, cv=cv, scoring='accuracy')
    
    tuned_results[name] = {
        'model': best_model,
        'test_acc': best_test_acc,
        'cv_mean': cv_scores_tuned.mean(),
        'cv_std': cv_scores_tuned.std(),
        'cv_scores': cv_scores_tuned,
        'best_params': gs.best_params_
    }
    
    improvement = (cv_scores_tuned.mean() - res['cv_mean']) * 100
    print(f"    Best params: {gs.best_params_}")
    print(f"    CV: {res['cv_mean']*100:.2f}% -> {cv_scores_tuned.mean()*100:.2f}% "
          f"({'+' if improvement > 0 else '-'}{abs(improvement):.2f}%)")
    print(f"    Test: {best_test_acc*100:.2f}%")

# ── STEP 4: Build Ensemble Models ────────────────────────────
print("\n" + "=" * 70)
print("  PART C: ENSEMBLE METHODS")
print("=" * 70)

# Voting Ensemble (Soft voting with top 3 tuned models)
print("\n  [Voting] Building Soft Voting Ensemble...")
voting_estimators = []
for name, res in tuned_results.items():
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    voting_estimators.append((safe_name, res['model']))

voting_clf = VotingClassifier(estimators=voting_estimators, voting='soft', n_jobs=1)
voting_cv = cross_val_score(voting_clf, X_clf, y_clf, cv=cv, scoring='accuracy')
voting_clf.fit(X_train, y_train)
voting_test = accuracy_score(y_test, voting_clf.predict(X_test))

clf_results["Voting Ensemble"] = {
    'model': voting_clf, 'test_acc': voting_test,
    'cv_mean': voting_cv.mean(), 'cv_std': voting_cv.std(),
    'cv_scores': voting_cv
}
print(f"    Voting Ensemble CV: {voting_cv.mean()*100:.2f}% +/- {voting_cv.std()*100:.2f}%")
print(f"    Voting Ensemble Test: {voting_test*100:.2f}%")

# Stacking Ensemble
print("\n  [Stacking] Building Stacking Ensemble...")
stack_estimators = []
for name, res in tuned_results.items():
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    stack_estimators.append((safe_name, res['model']))

stacking_clf = StackingClassifier(
    estimators=stack_estimators,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5, n_jobs=1
)
stacking_cv = cross_val_score(stacking_clf, X_clf, y_clf, cv=cv, scoring='accuracy')
stacking_clf.fit(X_train, y_train)
stacking_test = accuracy_score(y_test, stacking_clf.predict(X_test))

clf_results["Stacking Ensemble"] = {
    'model': stacking_clf, 'test_acc': stacking_test,
    'cv_mean': stacking_cv.mean(), 'cv_std': stacking_cv.std(),
    'cv_scores': stacking_cv
}
print(f"    Stacking Ensemble CV: {stacking_cv.mean()*100:.2f}% +/- {stacking_cv.std()*100:.2f}%")
print(f"    Stacking Ensemble Test: {stacking_test*100:.2f}%")

# ── STEP 5: Final Comparison & Save Best ─────────────────────
print("\n" + "=" * 70)
print("  FINAL RESULTS — ALL MODELS RANKED BY CV ACCURACY")
print("=" * 70)

# Merge tuned results back into clf_results
for name, res in tuned_results.items():
    clf_results[name + " (Tuned)"] = res

# Sort all results
all_sorted = sorted(clf_results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)

print(f"\n{'Rank':<6} {'Model':<28} {'Test Acc':>10} {'CV Mean':>10} {'CV Std':>10}")
print("-" * 70)
for i, (name, res) in enumerate(all_sorted):
    marker = " [BEST]" if i == 0 else ""
    print(f"  {i+1:<4} {name:<28} {res['test_acc']*100:>9.2f}% "
          f"{res['cv_mean']*100:>9.2f}% {res['cv_std']*100:>9.2f}%{marker}")

# Save the best model
best_name, best_res = all_sorted[0]
best_model = best_res['model']
joblib.dump(best_model, os.path.join(MODEL_DIR, 'best_model.pkl'))
joblib.dump(clf_features, os.path.join(MODEL_DIR, 'feature_names.pkl'))

print(f"\n  [Best Model] Best Model: {best_name}")
print(f"     CV Accuracy: {best_res['cv_mean']*100:.2f}% +/- {best_res['cv_std']*100:.2f}%")
print(f"     Test Accuracy: {best_res['test_acc']*100:.2f}%")
print(f"     Saved to: models/best_model.pkl")

# Classification report for best model
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
print(f"\n  Classification Report ({best_name}):")
print(classification_report(y_test, y_pred, target_names=['Low','Medium','High']))

# ── STEP 6: Regression Models ────────────────────────────────
print("\n" + "=" * 70)
print("  PART D: REGRESSION — Cognitive Score Prediction")
print("=" * 70)

reg_features = [c for c in df.columns if c not in
                ['addiction_level','attention_score','memory_score','reaction_time']]
X_reg = df[reg_features]

for target_name in ['attention_score', 'memory_score']:
    y_reg = df[target_name]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    reg_model = RandomForestRegressor(n_estimators=300, random_state=42)
    reg_model.fit(Xr_train, yr_train)
    y_pred_reg = reg_model.predict(Xr_test)
    rmse = np.sqrt(mean_squared_error(yr_test, y_pred_reg))
    r2   = r2_score(yr_test, y_pred_reg)
    joblib.dump(reg_model, os.path.join(MODEL_DIR, f'reg_{target_name}.pkl'))
    print(f"\n  {target_name}:")
    print(f"    R² Score : {r2:.4f}")
    print(f"    RMSE     : {rmse:.4f}")

# ── STEP 7: Generate Updated Charts ─────────────────────────
print("\n[Plotting] Generating updated charts...")

# Chart A: Full Model Comparison
fig, ax = plt.subplots(figsize=(14, 6))
names = [n for n, _ in all_sorted[:12]]  # Top 12
cv_means = [clf_results[n]['cv_mean']*100 if n in clf_results else all_sorted[i][1]['cv_mean']*100 
            for i, n in enumerate(names)]
cv_stds = [clf_results[n]['cv_std']*100 if n in clf_results else all_sorted[i][1]['cv_std']*100 
           for i, n in enumerate(names)]

# Use all_sorted directly
cv_means = [all_sorted[i][1]['cv_mean']*100 for i in range(min(12, len(all_sorted)))]
cv_stds = [all_sorted[i][1]['cv_std']*100 for i in range(min(12, len(all_sorted)))]
names = [all_sorted[i][0] for i in range(min(12, len(all_sorted)))]

colors_bar = ['#e74c3c' if i == 0 else '#2ecc71' if i < 3 else '#3498db' for i in range(len(names))]

bars = ax.bar(range(len(names)), cv_means, yerr=cv_stds, capsize=4,
              color=colors_bar, edgecolor='white', linewidth=1.2, width=0.6)
ax.set_ylabel('Cross-Validation Accuracy (%)', fontsize=11)
ax.set_title(f'Enhanced Model Comparison — {len(all_sorted)} Models Evaluated\n'
             f'(Best: {names[0]} at {cv_means[0]:.1f}% CV)', fontsize=12, fontweight='bold')
ax.set_ylim(50, 105)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)

for i, (bar, mean) in enumerate(zip(bars, cv_means)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + cv_stds[i] + 0.5,
            f'{mean:.1f}%', ha='center', fontsize=8, fontweight='bold')

best_patch = mpatches.Patch(color='#e74c3c', label=f'Best: {names[0]}')
top3_patch = mpatches.Patch(color='#2ecc71', label='Top 3')
ax.legend(handles=[best_patch, top3_patch], fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'chartA_cross_validation.png'), dpi=150)
plt.close()
print("   [OK] Saved: results/chartA_cross_validation.png")

# Chart B: Feature Importance
if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_,
                            index=clf_features).sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors_fi = ['#e74c3c' if v >= importances.quantile(0.75) else
                 '#f39c12' if v >= importances.quantile(0.50) else '#3498db'
                 for v in importances.values]
    importances.plot(kind='barh', ax=ax, color=colors_fi)
    ax.set_title(f'Top 15 Features — {best_name}\n'
                 f'(Red = High importance, Blue = Lower importance)', fontsize=12)
    ax.set_xlabel('Feature Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'chartB_feature_importance.png'), dpi=150)
    plt.close()
    print("   [OK] Saved: results/chartB_feature_importance.png")

# Chart C: CV Distribution
fig, ax = plt.subplots(figsize=(14, 5))
top_models = all_sorted[:8]
cv_data = [r['cv_scores']*100 for _, r in top_models]
bp = ax.boxplot(cv_data, labels=[n[:18] for n, _ in top_models],
                patch_artist=True, notch=False)
palette = ['#e74c3c','#2ecc71','#3498db','#9b59b6',
           '#f39c12','#1abc9c','#e67e22','#34495e']
for patch, color in zip(bp['boxes'], palette[:len(bp['boxes'])]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Accuracy per Fold (%)', fontsize=11)
ax.set_title('Cross-Validation Distribution — Top 8 Models', fontsize=12, fontweight='bold')
ax.set_ylim(50, 105)
plt.xticks(rotation=25, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'chartC_cv_boxplot.png'), dpi=150)
plt.close()
print("   [OK] Saved: results/chartC_cv_boxplot.png")

print("\n" + "=" * 70)
print("  TRAINING v3 COMPLETE")
print("=" * 70)
print(f"  [Best] Best: {best_name} - CV: {best_res['cv_mean']*100:.2f}%")
print(f"  Models saved: models/")
print(f"  Charts saved: results/")
print("=" * 70)
