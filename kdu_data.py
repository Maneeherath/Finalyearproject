# ============================================================
#  KDU STUDENT DATA (consented, anonymous)
#  When a student agrees, each completed assessment is saved as one row:
#  their answers, cognitive test results, eye score, the system's
#  prediction, and their own IGDS9-SF questionnaire score. Comparing the
#  predicted score with the questionnaire score is the real-world test of
#  the app on KDU students.
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, roc_auc_score

import igd_features as igdf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data', 'kdu')
DATA_CSV = os.path.join(DATA_DIR, 'kdu_assessments.csv')
MIN_N_FOR_STATS = 5        # below this, only descriptive counts are shown

CONSENT_TEXT = ("I agree that my **anonymous** answers and results (no name, student number or photos) may be used "
                "for this final-year research project. I can take part without agreeing.")

LIKERT_5 = ["Never", "Rarely", "Sometimes", "Often", "Very Often"]

# Pontes, H. M., & Griffiths, M. D. (2015). Measuring DSM-5 internet gaming disorder: Development and
# validation of a short psychometric scale. Computers in Human Behavior, 45, 137-143.
IGDS_INSTRUCTIONS = ("These questions ask about your gaming activity during the past year (i.e., last 12 months). "
                     "By gaming activity we understand any gaming-related activity that has been played either from a "
                     "computer/laptop or from a gaming console or any other kind of device (e.g., mobile phone, tablet, "
                     "etc.) both online and/or offline.")
IGDS_ITEMS = [
    "Do you feel preoccupied with your gaming behavior? (Some examples: Do you think about previous gaming "
    "activity or anticipate the next gaming session? Do you think gaming has become the dominant activity in "
    "your daily life?)",
    "Do you feel more irritability, anxiety or even sadness when you try to either reduce or stop your gaming activity?",
    "Do you feel the need to spend increasing amounts of time engaged in gaming in order to achieve satisfaction or pleasure?",
    "Do you systematically fail when trying to control or cease your gaming activity?",
    "Have you lost interests in previous hobbies and other entertainment activities as a result of your engagement with the game?",
    "Have you continued your gaming activity despite knowing it was causing problems between you and other people?",
    "Have you deceived any of your family members, therapists, or others about the amount of your gaming activity?",
    "Do you play in order to temporarily escape or relieve a negative mood (e.g., helplessness, guilt, anxiety)?",
    "Have you jeopardized or lost an important relationship, job or an educational or career opportunity "
    "because of your gaming activity?",
]


def load():
    return pd.read_csv(DATA_CSV) if os.path.exists(DATA_CSV) else pd.DataFrame()


def next_id():
    df = load()
    return f"K{len(df) + 1:03d}"


def save_row(row):
    """Insert or replace the row with this student's id (a student who changes answers updates their own row)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df = load()
    if not df.empty and 'student_id' in df:
        df = df[df['student_id'] != row['student_id']]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    tmp = DATA_CSV + '.tmp'
    df.to_csv(tmp, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(tmp, DATA_CSV)


def build_row(student_id, inputs, battery, eye_score, assessment, igds_answers):
    row = {'student_id': student_id, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
    row.update({f'in_{k}': v for k, v in inputs.items()})
    row['in_weekly_hours'] = assessment['total_hours']
    row.update({
        'nback_accuracy': battery['nback']['accuracy'], 'nback_dprime': battery['nback']['d_prime'],
        'ts_accuracy': battery['task_switching']['accuracy'], 'ts_switch_cost_ms': battery['task_switching']['switch_cost'],
        'mr_accuracy': battery['mental_rotation']['accuracy'], 'mr_slope_ms_per_deg': battery['mental_rotation']['slope_ms_per_deg'],
        'gng_commission_rate': battery['go_nogo']['commission_rate'], 'gng_go_rt_ms': battery['go_nogo']['mean_go_rt'],
        'eye_fatigue_score': eye_score if eye_score is not None else '',
        'pred_igds': round(assessment['pred_igds'], 2),
        'behaviour_risk': assessment['rf_pred'], 'fused_risk': assessment['fused_pred'],
        'fused_idx': assessment['fused_idx'], 'profile': assessment['archetype'],
    })
    for name, key in (('behaviour', 'rf_proba'), ('fused', 'fused_proba')):
        for i, p in enumerate(assessment[key]):
            row[f'{name}_p_{["low", "medium", "high"][i]}'] = round(float(p), 4)
    row.update({f'igds_q{i + 1}': v for i, v in enumerate(igds_answers)})
    row['igds_total'] = int(sum(igds_answers))
    row['igds_criteria_very_often'] = int(sum(1 for v in igds_answers if v == 5))
    return row


def _spearman(x, y):
    if len(x) < 3 or pd.Series(x).nunique() < 2 or pd.Series(y).nunique() < 2:
        return None, None
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def validation_summary(df):
    """How well the app's predictions match the students' own IGDS9-SF scores."""
    y = df['igds_total'].astype(float)
    pred = df['pred_igds'].astype(float)
    fused_score = df['fused_p_medium'] + 2 * df['fused_p_high']          # expected tier, 0 (Low) .. 2 (High)
    beh_score = df['behaviour_p_medium'] + 2 * df['behaviour_p_high']
    true_tier = y.apply(igdf.tier_of)
    rho_pred, p_pred = _spearman(pred, y)
    rho_fused, p_fused = _spearman(fused_score, y)
    rho_beh, p_beh = _spearman(beh_score, y)
    out = {
        'n': len(df), 'mae': float((pred - y).abs().mean()),
        'rho_pred': rho_pred, 'p_pred': p_pred, 'rho_fused': rho_fused, 'p_fused': p_fused,
        'rho_behaviour': rho_beh, 'p_behaviour': p_beh,
        'tier_agreement': float((true_tier == df['fused_idx']).mean()),
        'kappa': (float(cohen_kappa_score(true_tier, df['fused_idx'], labels=[0, 1, 2], weights='quadratic'))
                  if len(df) >= 2 else None),
        'auc21': None, 'n_ge21': int((y >= 21).sum()),
        'igds_mean': float(y.mean()), 'pred_mean': float(pred.mean()),
    }
    yy = (y >= 21).astype(int)
    if 0 < yy.sum() < len(yy):
        out['auc21'] = float(roc_auc_score(yy, pred))
    return out


def validation_chart(df):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    ax.scatter(df['igds_total'], df['pred_igds'], c=[colors[int(i)] for i in df['fused_idx']], s=40, alpha=0.8,
               edgecolor='k', linewidth=0.4)
    ax.plot([9, 45], [9, 45], color='grey', ls='--', lw=1)
    for c in (igdf.TIER_LOW_MAX + 0.5, igdf.TIER_HIGH_MIN - 0.5):
        ax.axvline(c, color='#bdc3c7', lw=0.7, ls=':')
    ax.set_xlim(8, 46)
    ax.set_ylim(8, 46)
    ax.set_xlabel("Student's own IGDS9-SF score (questionnaire)", fontsize=8)
    ax.set_ylabel("App's predicted score", fontsize=8)
    ax.set_title("KDU students: predicted vs actual (colour = app's fused risk)", fontsize=9, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def fmt(v, nd=2):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"
