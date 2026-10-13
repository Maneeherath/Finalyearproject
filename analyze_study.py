# ============================================================
#  VALIDATION STUDY ANALYSIS
#  Reads data/study/study_responses.csv (written by Study Mode in the app)
#  and produces the tables and charts for the Results / Testing chapters:
#    1. Sample description
#    2. IGDS9-SF descriptives + internal consistency (Cronbach's alpha)
#    3. Criterion validity: system risk vs IGDS9-SF
#         - Spearman rho (threshold-free, primary)
#         - fused system vs behaviour model alone (does fusion help on real users?)
#         - external validation of the real-data model: predicted vs actual IGDS9-SF score
#         - quadratic-weighted Cohen's kappa on 3 risk tiers
#         - ROC AUC against IGDS9-SF disorder cut-offs (32 primary; 21 and 36 sensitivity)
#    4. Usability: SUS mean, SD, 95% CI
#    5. Cognitive battery measures vs IGDS9-SF
#  Usage:  python analyze_study.py [--csv PATH] [--out DIR]
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))

RISK_LABELS = ["Low Risk", "Medium Risk", "High Risk"]

# IGDS9-SF cut-offs reported in the literature (no single universally accepted value):
#   >16 risky gaming and >21 disordered (Brazilian validation; Monacis et al., 2016 also used 21),
#   32 = clinically validated cut-off (Qin et al., 2020), 36 = authors' proposed cut-off (Pontes & Griffiths, 2015)
PRIMARY_CUTOFF = 32
SENSITIVITY_CUTOFFS = [21, 36]
# 3-tier mapping used only for the kappa / confusion-matrix comparison with Low/Medium/High
TIER_LOW_MAX, TIER_HIGH_MIN = 16, 32

COGNITIVE_MEASURES = {
    'nback_dprime': "2-Back d′ (working memory)",
    'nback_accuracy': "2-Back accuracy (%)",
    'ts_switch_cost_ms': "Switch cost (ms)",
    'ts_accuracy': "Task-switching accuracy (%)",
    'mr_accuracy': "Mental-rotation accuracy (%)",
    'mr_slope_ms_per_deg': "Rotation rate (ms/°)",
    'gng_commission_rate': "Go/No-Go commission errors (%)",
    'cog_go_rt_ms': "Go reaction time (ms)",
    'eye_fatigue_score': "Eye fatigue score (0–100)",
}


def igds_tier(total):
    return 0 if total <= TIER_LOW_MAX else 2 if total >= TIER_HIGH_MIN else 1


def cronbach_alpha(items):
    items = np.asarray(items, dtype=float)
    k = items.shape[1]
    item_var = items.var(axis=0, ddof=1).sum()
    total_var = items.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - item_var / total_var)) if total_var > 0 else float('nan')


def spearman(x, y):
    x, y = pd.Series(x, dtype=float), pd.Series(y, dtype=float)
    ok = x.notna() & y.notna()
    if ok.sum() < 3 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return {'rho': None, 'p': None, 'n': int(ok.sum())}
    rho, p = stats.spearmanr(x[ok], y[ok])
    return {'rho': round(float(rho), 3), 'p': round(float(p), 4), 'n': int(ok.sum())}


def risk_score(df, prefix):
    """Continuous risk score in [0, 2]: expected risk tier under the model's probabilities."""
    return df[f'{prefix}_p_medium'] + 2 * df[f'{prefix}_p_high']


def mean_ci(x, conf=0.95):
    x = pd.Series(x, dtype=float).dropna()
    if len(x) < 2:
        return {'mean': round(float(x.mean()), 2) if len(x) else None, 'sd': None, 'ci95': None, 'n': len(x)}
    m, se = x.mean(), stats.sem(x)
    lo, hi = stats.t.interval(conf, len(x) - 1, loc=m, scale=se) if se > 0 else (m, m)
    return {'mean': round(float(m), 2), 'sd': round(float(x.std(ddof=1)), 2),
            'ci95': [round(float(lo), 2), round(float(hi), 2)], 'n': len(x)}


def auc_at(df, cutoff, score_col):
    y = (df['igds_total'] >= cutoff).astype(int)
    if y.nunique() < 2:
        return {'cutoff': cutoff, 'auc': None, 'positives': int(y.sum()), 'note': 'only one class present'}
    return {'cutoff': cutoff, 'auc': round(float(roc_auc_score(y, df[score_col])), 3), 'positives': int(y.sum())}


def fmt(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def analyse(df):
    igds_items = df[[f'igds_q{i}' for i in range(1, 10)]]
    df = df.copy()
    df['fused_score'] = risk_score(df, 'sys_fused')
    df['behaviour_score'] = risk_score(df, 'sys_behaviour')
    df['igds_tier'] = df['igds_total'].apply(igds_tier)
    rf_idx = df['sys_behaviour_pred'].map({l: i for i, l in enumerate(RISK_LABELS)})

    res = {'n': len(df)}
    res['sample'] = {
        'age': mean_ci(df['input_age']),
        'gender': df['input_gender'].value_counts().to_dict(),
        'weekly_gaming_hours': mean_ci(df['input_total_hours_week']),
        'sleep_hours': mean_ci(df['input_sleep_hours']),
        'eye_check_completed': int(df['eye_fatigue_score'].notna().sum()),
    }
    res['igds9sf'] = {
        'total': mean_ci(df['igds_total']),
        'cronbach_alpha': round(cronbach_alpha(igds_items), 3) if len(df) >= 3 else None,
        'prevalence': {f'>= {c}': int((df['igds_total'] >= c).sum()) for c in [21, 32, 36]},
        'dsm5_5plus_very_often': int((df['igds_criteria_very_often'] >= 5).sum()),
        'tiers': {RISK_LABELS[t]: int((df['igds_tier'] == t).sum()) for t in range(3)},
    }
    res['criterion_validity'] = {
        'spearman_fused_score_vs_igds': spearman(df['fused_score'], df['igds_total']),
        'spearman_behaviour_score_vs_igds': spearman(df['behaviour_score'], df['igds_total']),
        'spearman_fused_level_vs_igds': spearman(df['sys_fused_idx'], df['igds_total']),
        'spearman_predicted_vs_actual_igds': spearman(df['sys_pred_igds'], df['igds_total']),
        'mae_predicted_vs_actual_igds': round(float((df['sys_pred_igds'] - df['igds_total']).abs().mean()), 2),
        'weighted_kappa_fused': (round(float(cohen_kappa_score(df['igds_tier'], df['sys_fused_idx'],
                                                               labels=[0, 1, 2], weights='quadratic')), 3)
                                 if len(df) >= 2 else None),
        'weighted_kappa_behaviour': (round(float(cohen_kappa_score(df['igds_tier'], rf_idx,
                                                                  labels=[0, 1, 2], weights='quadratic')), 3)
                                    if len(df) >= 2 else None),
        'exact_agreement_fused_pct': round(float((df['igds_tier'] == df['sys_fused_idx']).mean() * 100), 1),
        'auc_fused': [auc_at(df, c, 'fused_score') for c in [PRIMARY_CUTOFF] + SENSITIVITY_CUTOFFS],
        'auc_behaviour': [auc_at(df, c, 'behaviour_score') for c in [PRIMARY_CUTOFF] + SENSITIVITY_CUTOFFS],
    }
    sus = mean_ci(df['sus_score'])
    sus['median'] = round(float(df['sus_score'].median()), 1)
    sus['pct_above_68'] = round(float((df['sus_score'] > 68).mean() * 100), 1)
    res['usability_sus'] = sus
    res['cognitive_vs_igds'] = {label: spearman(df[col], df['igds_total'])
                                for col, label in COGNITIVE_MEASURES.items() if col in df}
    return df, res


def make_charts(df, res, out):
    paths = {}

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    ax.scatter(df['igds_total'], df['fused_score'], c=[colors[i] for i in df['sys_fused_idx']], s=40, alpha=0.8,
               edgecolor='k', linewidth=0.4)
    ax.axvline(PRIMARY_CUTOFF, color='grey', linestyle='--', linewidth=0.8)
    ax.text(PRIMARY_CUTOFF + 0.3, 1.9, f'IGDS9-SF cut-off {PRIMARY_CUTOFF}', fontsize=7, color='grey')
    r = res['criterion_validity']['spearman_fused_score_vs_igds']
    ax.set_title(f"System risk vs IGDS9-SF (Spearman ρ = {fmt(r['rho'])}, n = {r['n']})", fontsize=9, fontweight='bold')
    ax.set_xlabel('IGDS9-SF total (9–45)', fontsize=8)
    ax.set_ylabel('Fused risk score (0 = Low, 2 = High)', fontsize=8)
    ax.set_ylim(-0.05, 2.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    paths['scatter'] = os.path.join(out, 'study1_risk_vs_igds.png')
    fig.savefig(paths['scatter'], dpi=150)
    plt.close(fig)

    cm = confusion_matrix(df['igds_tier'], df['sys_fused_idx'], labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap='Blues')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    short = ['Low', 'Medium', 'High']
    ax.set_xticks(range(3), short)
    ax.set_yticks(range(3), short)
    ax.set_xlabel('System (fused) risk level', fontsize=8)
    ax.set_ylabel(f'IGDS9-SF tier (≤{TIER_LOW_MAX} / {TIER_LOW_MAX+1}–{TIER_HIGH_MIN-1} / ≥{TIER_HIGH_MIN})', fontsize=8)
    k = res['criterion_validity']['weighted_kappa_fused']
    ax.set_title(f"Agreement (quadratic-weighted κ = {fmt(k)})", fontsize=9, fontweight='bold')
    plt.tight_layout()
    paths['confusion'] = os.path.join(out, 'study2_agreement_matrix.png')
    fig.savefig(paths['confusion'], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(df['sus_score'], bins=np.arange(0, 105, 5), color='#3498db', edgecolor='white')
    ax.axvline(68, color='#e67e22', linestyle='--', linewidth=1)
    ax.text(68.5, ax.get_ylim()[1] * 0.9, 'average (68)', fontsize=7, color='#e67e22')
    s = res['usability_sus']
    ax.set_title(f"SUS usability scores (mean = {fmt(s['mean'], 1)}, n = {s['n']})", fontsize=9, fontweight='bold')
    ax.set_xlabel('SUS score (0–100)', fontsize=8)
    ax.set_ylabel('Participants', fontsize=8)
    plt.tight_layout()
    paths['sus'] = os.path.join(out, 'study3_sus_distribution.png')
    fig.savefig(paths['sus'], dpi=150)
    plt.close(fig)

    items = [(k, v['rho']) for k, v in res['cognitive_vs_igds'].items() if v['rho'] is not None]
    if items:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        names, rhos = zip(*items)
        ax.barh(names, rhos, color=['#e74c3c' if r > 0 else '#3498db' for r in rhos])
        ax.axvline(0, color='grey', linewidth=0.8)
        ax.set_xlim(-1, 1)
        ax.set_xlabel('Spearman ρ with IGDS9-SF total', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title('Cognitive & eye measures vs gaming-disorder symptoms', fontsize=9, fontweight='bold')
        plt.tight_layout()
        paths['cognitive'] = os.path.join(out, 'study4_cognitive_vs_igds.png')
        fig.savefig(paths['cognitive'], dpi=150)
        plt.close(fig)
    return paths


def write_markdown(res, paths, out):
    cv = res['criterion_validity']
    lines = [f"# Validation Study Results (n = {res['n']})", ""]
    lines += ["## Criterion validity: system risk vs IGDS9-SF", "",
              "| Measure | Fused system | Behaviour model alone |", "|---|---|---|",
              f"| Spearman ρ (risk score vs IGDS9-SF total) | {fmt(cv['spearman_fused_score_vs_igds']['rho'])} "
              f"(p = {fmt(cv['spearman_fused_score_vs_igds']['p'], 4)}) | {fmt(cv['spearman_behaviour_score_vs_igds']['rho'])} "
              f"(p = {fmt(cv['spearman_behaviour_score_vs_igds']['p'], 4)}) |",
              f"| Quadratic-weighted κ (3 tiers) | {fmt(cv['weighted_kappa_fused'])} | {fmt(cv['weighted_kappa_behaviour'])} |"]
    for fa, sa in zip(cv['auc_fused'], cv['auc_behaviour']):
        tag = ' (primary)' if fa['cutoff'] == PRIMARY_CUTOFF else ' (sensitivity)'
        lines.append(f"| ROC AUC, IGDS9-SF ≥ {fa['cutoff']}{tag} — {fa['positives']} positive | "
                     f"{fmt(fa['auc'])} | {fmt(sa['auc'])} |")
    pv = cv['spearman_predicted_vs_actual_igds']
    lines += ["", f"Exact 3-tier agreement (fused): {cv['exact_agreement_fused_pct']}%", "",
              "**External validation of the real-data model:** predicted vs questionnaire IGDS9-SF score — "
              f"Spearman ρ = {fmt(pv['rho'])} (p = {fmt(pv['p'], 4)}), MAE = {fmt(cv['mae_predicted_vs_actual_igds'], 2)} points.", ""]

    ig = res['igds9sf']
    lines += ["## IGDS9-SF", "",
              f"- Total: mean {fmt(ig['total']['mean'], 2)}, SD {fmt(ig['total']['sd'], 2)}",
              f"- Cronbach's α: {fmt(ig['cronbach_alpha'])}",
              f"- Scoring ≥ 21 / ≥ 32 / ≥ 36: {ig['prevalence']['>= 21']} / {ig['prevalence']['>= 32']} / {ig['prevalence']['>= 36']}",
              f"- ≥ 5 criteria endorsed 'Very Often' (DSM-5 style): {ig['dsm5_5plus_very_often']}", ""]

    s = res['usability_sus']
    lines += ["## Usability (SUS)", "",
              f"- Mean {fmt(s['mean'], 1)}, SD {fmt(s['sd'], 1)}, 95% CI {s['ci95']}, median {fmt(s['median'], 1)}",
              f"- {s['pct_above_68']}% of participants scored above the industry average of 68", ""]

    lines += ["## Cognitive & eye measures vs IGDS9-SF total", "", "| Measure | Spearman ρ | p | n |", "|---|---|---|---|"]
    for k, v in res['cognitive_vs_igds'].items():
        lines.append(f"| {k} | {fmt(v['rho'])} | {fmt(v['p'], 4)} | {v['n']} |")
    lines += ["", "## Charts", ""] + [f"- `{os.path.relpath(p, BASE)}`" for p in paths.values()]
    path = os.path.join(out, 'study_summary.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def main():
    ap = argparse.ArgumentParser(description="Analyse the validation-study responses.")
    ap.add_argument('--csv', default=os.path.join(BASE, 'data', 'study', 'study_responses.csv'))
    ap.add_argument('--out', default=os.path.join(BASE, 'results', 'study'))
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"No study data yet: {args.csv}\nRun sessions in the app's '📝 Validation Study' page first.")
    df = pd.read_csv(args.csv)
    if len(df) < 2:
        raise SystemExit(f"Only {len(df)} session(s) recorded — at least 2 are needed (aim for 30+ for meaningful statistics).")

    os.makedirs(args.out, exist_ok=True)
    df, res = analyse(df)
    paths = make_charts(df, res, args.out)
    with open(os.path.join(args.out, 'study_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, default=str)
    md = write_markdown(res, paths, args.out)

    cv = res['criterion_validity']
    print(f"Participants: {res['n']}")
    print(f"Spearman rho, fused vs IGDS9-SF:    {fmt(cv['spearman_fused_score_vs_igds']['rho'])}")
    print(f"Spearman rho, behaviour vs IGDS9-SF: {fmt(cv['spearman_behaviour_score_vs_igds']['rho'])}")
    print(f"Weighted kappa (fused):             {fmt(cv['weighted_kappa_fused'])}")
    print(f"SUS mean:                           {fmt(res['usability_sus']['mean'], 1)}")
    if res['n'] < 30:
        print(f"Note: n = {res['n']} is small — report these as preliminary results.")
    print(f"Wrote {md} and {len(paths)} charts to {args.out}")


if __name__ == '__main__':
    main()
