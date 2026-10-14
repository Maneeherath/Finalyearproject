# ============================================================
#  REAL-DATA IGD RISK MODEL  (replaces the synthetic-data models)
#  Dataset: Manchanayake (2021), Mendeley Data, doi:10.17632/8r2jgm6ygh.1 (CC BY 4.0)
#           322 Sri Lankan Advanced Level students who game, with the full IGDS9-SF.
#  Target:  IGDS9-SF total score (9-45), predicted from gaming habits, sleep and
#           psychosocial answers that the app can collect.
#  Risk tiers (<=16 Low, 17-31 Medium, >=32 High) are derived from the predicted
#  score and the cross-validated error, so the app gets probabilities for fusion.
#
#  Evaluation: repeated 5-fold cross-validation (10 repeats) for every model,
#  same folds for all models, compared against a "predict the mean" baseline.
#  Run:  python notebooks/07_train_real_igd_model.py
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import json
import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor,
                              StackingRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import (cohen_kappa_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, BASE)
import igd_features as F  # noqa: E402

MODEL_DIR = os.path.join(BASE, 'models')
RESULTS_DIR = os.path.join(BASE, 'results')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
SEED = 42
N_SPLITS, N_REPEATS = 5, 10

# ── 1. Data ──────────────────────────────────────────────────────────────────
raw = F.load_dataset()
X = F.features_from_dataset(raw)
y = raw['IGD Total'].astype(float).values
tiers = np.array([F.tier_of(v) for v in y])
dsm_status = (raw['Criteria Endorsed'] >= 5).astype(int).values      # DSM-5 style: >=5 criteria endorsed
print(f"Dataset: {len(X)} gamers, {X.shape[1]} features | IGDS9-SF mean {y.mean():.1f} (SD {y.std(ddof=1):.1f})")
print(f"Tiers: Low {np.sum(tiers == 0)}, Medium {np.sum(tiers == 1)}, High {np.sum(tiers == 2)} | "
      f"score >=21: {np.sum(y >= 21)} | DSM-5 status: {dsm_status.sum()}")


# ── 2. Candidate models ──────────────────────────────────────────────────────
def scaled(model):
    return make_pipeline(StandardScaler(), model)


MODELS = {
    'Mean baseline': DummyRegressor(),
    'Ridge Regression': scaled(Ridge(alpha=10.0)),
    'Elastic Net': scaled(ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)),
    'SVR (RBF)': scaled(SVR(C=10.0, epsilon=1.0)),
    'k-Nearest Neighbours': scaled(KNeighborsRegressor(n_neighbors=15)),
    'Random Forest': RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=1),
    'Extra Trees': ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2,
                                                   subsample=0.8, random_state=SEED),
    'XGBoost': XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=2, subsample=0.8,
                            colsample_bytree=0.8, random_state=SEED, verbosity=0),
    'LightGBM': LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=15,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1),
}
MODELS['Stacking Ensemble'] = StackingRegressor(
    estimators=[('ridge', MODELS['Ridge Regression']), ('svr', MODELS['SVR (RBF)']),
                ('rf', MODELS['Random Forest']), ('gb', MODELS['Gradient Boosting'])],
    final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=1)


def derived_metrics(y_true, pred, sigma):
    """Classification-style metrics from a predicted score (tiers, cut-offs)."""
    t_true = np.array([F.tier_of(v) for v in y_true])
    proba = F.tier_probabilities(pred, sigma)
    t_pred = proba.argmax(axis=1)
    out = {
        'tier_accuracy': float(np.mean(t_true == t_pred)),
        'tier_macro_f1': float(f1_score(t_true, t_pred, average='macro', labels=[0, 1, 2], zero_division=0)),
        'tier_weighted_kappa': float(cohen_kappa_score(t_true, t_pred, labels=[0, 1, 2], weights='quadratic')),
    }
    for cut in (21, 32):
        yy = (y_true >= cut).astype(int)
        out[f'auc_ge{cut}'] = float(roc_auc_score(yy, pred)) if 0 < yy.sum() < len(yy) else float('nan')
    return out


# ── 3. Repeated 5-fold cross-validation (identical folds for every model) ────
rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
splits = list(rkf.split(X))
rows, fold_scores, oof = [], {}, {}
for name, model in MODELS.items():
    mae, rmse, r2, rho, derived = [], [], [], [], []
    oof_sum, oof_n = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in splits:
        model.fit(X.iloc[tr], y[tr])
        p = model.predict(X.iloc[te])
        mae.append(mean_absolute_error(y[te], p))
        rmse.append(np.sqrt(mean_squared_error(y[te], p)))
        r2.append(r2_score(y[te], p))
        rho.append(spearmanr(y[te], p)[0] if np.std(p) > 0 else 0.0)
        oof_sum[te] += p
        oof_n[te] += 1
    oof[name] = oof_sum / oof_n          # each student's average out-of-fold prediction over the 10 repeats
    sigma = float(np.mean(rmse))
    d = derived_metrics(y, oof[name], sigma)
    fold_scores[name] = r2
    rows.append({'model': name, 'MAE': np.mean(mae), 'MAE_sd': np.std(mae), 'RMSE': sigma, 'R2': np.mean(r2),
                 'R2_sd': np.std(r2), 'Spearman': np.nanmean(rho), **d})
    print(f"  {name:22s} MAE {np.mean(mae):5.2f} ± {np.std(mae):.2f} | R² {np.mean(r2):+.3f} ± {np.std(r2):.3f} | "
          f"ρ {np.nanmean(rho):.3f} | AUC≥21 {d['auc_ge21']:.3f} | κ {d['tier_weighted_kappa']:.3f}")

res = pd.DataFrame(rows).sort_values('MAE').reset_index(drop=True)
best_name = res.loc[res['model'] != 'Mean baseline', 'model'].iloc[0]
best_row = res[res['model'] == best_name].iloc[0]
base_row = res[res['model'] == 'Mean baseline'].iloc[0]
print(f"\nBest model by MAE: {best_name} (MAE {best_row['MAE']:.2f} vs baseline {base_row['MAE']:.2f})")

# ── 4. Is the improvement over the baseline reliable? (paired across the 50 folds) ──
from scipy.stats import wilcoxon  # noqa: E402
w = wilcoxon(fold_scores[best_name], fold_scores['Mean baseline'])
print(f"Wilcoxon signed-rank on fold R² vs baseline: p = {w.pvalue:.2e}")

# ── 5. Research-only comparison: does adding gaming motives (MOGQ) help? ─────
motives = raw[['Escape', 'Coping']].astype(float)
X_mot = pd.concat([X, motives.fillna(motives.median())], axis=1)
mot_r2 = []
best_model = MODELS[best_name]
for tr, te in splits:
    best_model.fit(X_mot.iloc[tr], y[tr])
    mot_r2.append(r2_score(y[te], best_model.predict(X_mot.iloc[te])))
print(f"{best_name} + escape/coping motives: R² {np.mean(mot_r2):.3f} (vs {best_row['R2']:.3f} without)")

# ── 6. Final model on all data + permutation importance (on held-out folds) ──
final = MODELS[best_name]
importances = []
for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X):
    final.fit(X.iloc[tr], y[tr])
    pi = permutation_importance(final, X.iloc[te], y[te], n_repeats=20, random_state=SEED,
                                scoring='neg_mean_absolute_error')
    importances.append(pi.importances_mean)
imp = pd.Series(np.mean(importances, axis=0), index=F.FEATURES).sort_values(ascending=False)
final.fit(X, y)

# ── 7. Behaviour clusters (replaces the synthetic K-means archetypes) ────────
CLUSTER_FEATURES = ['weekday_hours', 'weekend_hours', 'sleep_hours', 'sleep_quality', 'mode_online_multi']
Xc = X[CLUSTER_FEATURES]
c_scaler = StandardScaler().fit(Xc)
Xcs = c_scaler.transform(Xc)
sil = {k: silhouette_score(Xcs, KMeans(k, n_init=20, random_state=SEED).fit_predict(Xcs)) for k in range(2, 7)}
k_best = max(sil, key=sil.get)
kmeans = KMeans(k_best, n_init=20, random_state=SEED).fit(Xcs)
labels = kmeans.labels_
overall = X[CLUSTER_FEATURES].mean()


def describe_cluster(c):
    m = X[labels == c][CLUSTER_FEATURES].mean()
    weekly = 5 * m['weekday_hours'] + 2 * m['weekend_hours']
    load = "Heavy" if weekly > 1.4 * (5 * overall['weekday_hours'] + 2 * overall['weekend_hours']) else \
           "Light" if weekly < 0.75 * (5 * overall['weekday_hours'] + 2 * overall['weekend_hours']) else "Moderate"
    style = "online multiplayer" if m['mode_online_multi'] > 0.6 else "offline / solo" if m['mode_online_multi'] < 0.3 else "mixed-mode"
    sleep = "short-sleep" if m['sleep_hours'] < overall['sleep_hours'] - 0.4 or m['sleep_quality'] < overall['sleep_quality'] - 1 \
        else "well-rested" if m['sleep_hours'] > overall['sleep_hours'] + 0.4 else ""
    name = " ".join(w for w in [load, sleep, style, "gamers"] if w).capitalize()
    desc = (f"About {weekly:.0f} h/week of gaming ({m['weekday_hours']:.1f} h weekdays, {m['weekend_hours']:.1f} h weekends), "
            f"{m['sleep_hours']:.1f} h sleep (quality {m['sleep_quality']:.1f}/10), "
            f"{m['mode_online_multi'] * 100:.0f}% mainly online multiplayer.")
    return {'name': name, 'description': desc, 'n': int(np.sum(labels == c)),
            'mean_igds': float(y[labels == c].mean()), 'weekly_hours': float(weekly)}


profiles = {int(c): describe_cluster(c) for c in range(k_best)}
print(f"\nClusters (k = {k_best}, silhouette {sil[k_best]:.3f}):")
for c, p in profiles.items():
    print(f"  {c}: {p['name']:45s} n={p['n']:3d}  mean IGDS9-SF {p['mean_igds']:.1f}")

# ── 8. Reference profiles for the radar chart (real group means) ─────────────
RADAR = ['weekday_hours', 'weekend_hours', 'sleep_hours', 'sleep_quality', 'self_esteem', 'few_friends']
radar_ref = {
    'low_risk': X[y <= F.TIER_LOW_MAX][RADAR].mean().to_dict(),
    'elevated': X[y >= 21][RADAR].mean().to_dict(),
    'ranges': {f: [float(X[f].min()), float(X[f].max())] for f in RADAR},
    'n_low': int(np.sum(y <= F.TIER_LOW_MAX)), 'n_elevated': int(np.sum(y >= 21)),
}

# ── 9. Save model + metadata ─────────────────────────────────────────────────
meta = {
    'model_name': best_name,
    'features': F.FEATURES,
    'sigma': float(best_row['RMSE']),
    'cv': {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in best_row.items()},
    'baseline_cv': {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in base_row.items()},
    'wilcoxon_p_vs_baseline': float(w.pvalue),
    'motives_r2': float(np.mean(mot_r2)),
    'cv_protocol': f'{N_REPEATS}x repeated {N_SPLITS}-fold CV, seed {SEED}',
    'n': int(len(y)), 'target_mean': float(y.mean()), 'target_sd': float(y.std(ddof=1)),
    'tiers': {'low_max': F.TIER_LOW_MAX, 'high_min': F.TIER_HIGH_MIN,
              'counts': [int(np.sum(tiers == t)) for t in range(3)]},
    'feature_importance': {k: float(v) for k, v in imp.items()},
    'feature_medians': X.median().to_dict(),
    'radar_reference': radar_ref,
    'dataset': 'Manchanayake, M. (2021). Mendeley Data, V1. doi:10.17632/8r2jgm6ygh.1 (CC BY 4.0)',
}
joblib.dump({'model': final, 'meta': meta}, os.path.join(MODEL_DIR, 'igd_risk_model.pkl'))
joblib.dump({'scaler': c_scaler, 'kmeans': kmeans, 'features': CLUSTER_FEATURES, 'profiles': profiles,
             'silhouette': float(sil[k_best])}, os.path.join(MODEL_DIR, 'igd_clusters.pkl'))
res.to_csv(os.path.join(RESULTS_DIR, 'real_model_comparison.csv'), index=False)
with open(os.path.join(RESULTS_DIR, 'real_model_results.json'), 'w', encoding='utf-8') as fh:
    json.dump({'models': res.round(4).to_dict('records'), 'best': meta['cv'], 'baseline': meta['baseline_cv'],
               'wilcoxon_p_vs_baseline': meta['wilcoxon_p_vs_baseline'], 'motives_r2': meta['motives_r2'],
               'feature_importance': meta['feature_importance'], 'clusters': profiles,
               'cluster_silhouette': sil, 'protocol': meta['cv_protocol'], 'dataset': meta['dataset'],
               'n': meta['n']}, fh, indent=2, default=float)

# ── 10. Charts ───────────────────────────────────────────────────────────────
plot = res.sort_values('R2')
fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#95a5a6' if m == 'Mean baseline' else '#e74c3c' if m == best_name else '#3498db' for m in plot['model']]
ax.barh(plot['model'], plot['R2'], xerr=plot['R2_sd'], color=colors, capsize=3)
for i, (r, m) in enumerate(zip(plot['R2'], plot['MAE'])):
    ax.text(max(r, 0) + 0.01, i, f"R² {r:.3f} · MAE {m:.2f}", va='center', fontsize=8)
ax.axvline(0, color='grey', lw=0.8)
ax.set_xlabel('Cross-validated R² (10 × 5-fold), error bars = SD across folds')
ax.set_title(f'IGDS9-SF score prediction on real data (n = {len(y)}) — best: {best_name}', fontweight='bold', fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'real1_model_comparison.png'), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(y, oof[best_name], c=[['#2ecc71', '#f39c12', '#e74c3c'][t] for t in tiers],
           s=22, alpha=0.75, edgecolor='k', linewidth=0.3)
ax.plot([9, 45], [9, 45], color='grey', ls='--', lw=1)
for c in (F.TIER_LOW_MAX + 0.5, F.TIER_HIGH_MIN - 0.5):
    ax.axvline(c, color='grey', lw=0.6, ls=':')
    ax.axhline(c, color='grey', lw=0.6, ls=':')
ax.set_xlabel('Actual IGDS9-SF score (questionnaire)')
ax.set_ylabel('Predicted score (out-of-fold)')
ax.set_title(f"{best_name}: Spearman ρ = {best_row['Spearman']:.2f}, MAE = {best_row['MAE']:.2f}", fontweight='bold', fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'real2_predicted_vs_actual.png'), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
top = imp.head(12)[::-1]
ax.barh([F.FEATURE_LABELS[f] for f in top.index], top.values, color=['#e74c3c' if v > 0 else '#bdc3c7' for v in top.values])
ax.set_xlabel('Permutation importance (increase in MAE when shuffled, held-out folds)')
ax.set_title(f'What drives the predicted IGDS9-SF score — {best_name}', fontweight='bold', fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'real3_feature_importance.png'), dpi=150)
plt.close(fig)

t_pred = F.tier_probabilities(oof[best_name], meta['sigma']).argmax(axis=1)
cm = confusion_matrix(tiers, t_pred, labels=[0, 1, 2])
fig, ax = plt.subplots(figsize=(4.8, 4.2))
ax.imshow(cm, cmap='Blues')
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
ax.set_xticks(range(3), ['Low', 'Medium', 'High'])
ax.set_yticks(range(3), ['Low', 'Medium', 'High'])
ax.set_xlabel('Predicted tier')
ax.set_ylabel(f'Actual tier (≤{F.TIER_LOW_MAX} / {F.TIER_LOW_MAX + 1}–{F.TIER_HIGH_MIN - 1} / ≥{F.TIER_HIGH_MIN})')
ax.set_title(f"Risk tiers: accuracy {best_row['tier_accuracy'] * 100:.1f}%, κw = {best_row['tier_weighted_kappa']:.2f}",
             fontweight='bold', fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'real4_tier_confusion.png'), dpi=150)
plt.close(fig)

print(f"\nSaved models/igd_risk_model.pkl, models/igd_clusters.pkl and results/real*.png / real_model_results.json")
