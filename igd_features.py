# ============================================================
#  SHARED FEATURE DEFINITIONS FOR THE REAL-DATA IGD RISK MODEL
#  Used by BOTH the training script (notebooks/07_train_real_igd_model.py)
#  and the app, so a student's answers are encoded exactly the way the
#  training data was.
#
#  Training data: Manchanayake, M. (2021). Findings from a study investigating
#  the prevalence and associations of Internet Gaming Disorder among an
#  Advanced Level student population from Colombo, Sri Lanka. Mendeley Data,
#  V1. https://doi.org/10.17632/8r2jgm6ygh.1  (CC BY 4.0)
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import os

import numpy as np
import pandas as pd
from scipy.stats import norm

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, 'data', 'external', 'srilanka_igd', 'IGD_Database.xlsx')

# IGDS9-SF risk tiers (same cut-offs as Study Mode / analyze_study.py):
#   <=16 Low, 17-31 Medium, >=32 High  (>16 risky gaming; 32 = clinically validated cut-off, Qin et al., 2020)
TIER_LOW_MAX, TIER_HIGH_MIN = 16, 32
RISK_LABELS = ["Low Risk", "Medium Risk", "High Risk"]

# The questionnaire recorded ranges; each range is represented by its midpoint
HOURS_MID = {'1 or less': 0.5, '2 to 3': 2.5, '4 to 5': 4.5, '6 to 7': 6.5, '8 to 10': 9.0, '11 or more': 12.0}
SLEEP_MID = {'less than 4': 3.5, '4 to 5': 4.5, '6 to 7': 6.5, '8 or more': 8.5}
START_AGE_MID = {'less than 5': 4.0, '5 to 7': 6.0, '8 to 10': 9.0, '11 to 13': 12.0, '14 to 16': 15.0}
SPORTS_LEVEL = {'Minor': 1, 'Moderate': 2, 'Major': 3}          # blank = no sports involvement

GENRE_GROUP = {
    'BR': 'Battle royale',
    'Non BR Shooter': 'Shooter',
    'Action-Adv.': 'Action / strategy / story', 'Strategy': 'Action / strategy / story',
    'Fighting': 'Action / strategy / story', 'Story': 'Action / strategy / story',
    'Simulator': 'Action / strategy / story',
    'Puzzle/Arcade': 'Casual (puzzle / platform)', 'Platform': 'Casual (puzzle / platform)',
    'Sports': 'Sports / racing', 'Racing': 'Sports / racing',
}
GENRES = ['Battle royale', 'Shooter', 'Action / strategy / story', 'Sports / racing', 'Casual (puzzle / platform)']
MODES = ['Online multiplayer', 'Online single-player', 'Offline single-player']
DEVICES = ['Mobile', 'PC', 'Console', 'Tablet']

FEATURES = [
    'weekday_hours', 'weekend_hours', 'weekly_hours', 'sleep_hours', 'sleep_quality', 'start_age',
    'is_male', 'mode_online_multi', 'mode_online_single',
    'genre_br', 'genre_shooter', 'genre_action', 'genre_sports',
    'device_mobile', 'device_pc', 'self_esteem', 'few_friends', 'sports_level',
]
FEATURE_LABELS = {
    'weekday_hours': 'Weekday gaming (h/day)', 'weekend_hours': 'Weekend gaming (h/day)',
    'weekly_hours': 'Weekly gaming (h/week)', 'sleep_hours': 'Sleep (h/night)', 'sleep_quality': 'Sleep quality (1-10)',
    'start_age': 'Age started gaming', 'is_male': 'Male', 'mode_online_multi': 'Online multiplayer',
    'mode_online_single': 'Online single-player', 'genre_br': 'Battle royale', 'genre_shooter': 'Shooter',
    'genre_action': 'Action/strategy/story', 'genre_sports': 'Sports/racing', 'device_mobile': 'Mobile device',
    'device_pc': 'PC', 'self_esteem': 'Self-esteem (1-5)', 'few_friends': 'Few friends (1-5)',
    'sports_level': 'Sports involvement (0-3)',
}


def load_dataset(path=DATA_PATH):
    """Gamers who completed the IGDS9-SF, with the two out-of-range MOGQ entries set to missing."""
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    df = df[df['IGD Total'].notna()].copy()
    mogq = [f'MOGQ {i}' for i in range(1, 28)]
    df[mogq] = df[mogq].where((df[mogq] >= 1) & (df[mogq] <= 5))
    return df


def features_from_dataset(df):
    age = df['Age'].astype(float)
    start = df['Start Age'].map(START_AGE_MID)
    start = start.fillna(age - 0.5)                       # 'Recently' -> started within the last year
    genre = df['Genre'].map(GENRE_GROUP)
    out = pd.DataFrame({
        'weekday_hours': df['Weekday Hours'].map(HOURS_MID),
        'weekend_hours': df['Weekend Hours'].map(HOURS_MID),
        'sleep_hours': df['Sleep Hours /d'].map(SLEEP_MID),
        'sleep_quality': df['Sleep Quality'].astype(float),
        'start_age': start,
        'is_male': (df['Sex'] == 'M').astype(int),
        'mode_online_multi': (df['Game Type'] == 'Online Multiplayer').astype(int),
        'mode_online_single': (df['Game Type'] == 'Online Single player').astype(int),
        'genre_br': (genre == 'Battle royale').astype(int),
        'genre_shooter': (genre == 'Shooter').astype(int),
        'genre_action': (genre == 'Action / strategy / story').astype(int),
        'genre_sports': (genre == 'Sports / racing').astype(int),
        'device_mobile': (df['Device'] == 'Mobile').astype(int),
        'device_pc': (df['Device'] == 'PC').astype(int),
        'self_esteem': df['Self-esteem'].astype(float),
        'few_friends': df['No/Few Friends'].astype(float),
        'sports_level': df['Sports'].map(SPORTS_LEVEL).fillna(0),
    }, index=df.index)
    out['weekly_hours'] = 5 * out['weekday_hours'] + 2 * out['weekend_hours']
    return out[FEATURES]


def features_from_inputs(inp):
    """inp: dict of the app's Home-page answers -> one-row feature frame."""
    genre, mode, device = inp['genre'], inp['mode'], inp['device']
    row = {
        'weekday_hours': inp['weekday_hours'], 'weekend_hours': inp['weekend_hours'],
        'weekly_hours': 5 * inp['weekday_hours'] + 2 * inp['weekend_hours'],
        'sleep_hours': inp['sleep_hours'], 'sleep_quality': inp['sleep_quality'],
        'start_age': inp['start_age'], 'is_male': int(inp['gender'] == 'Male'),
        'mode_online_multi': int(mode == 'Online multiplayer'), 'mode_online_single': int(mode == 'Online single-player'),
        'genre_br': int(genre == 'Battle royale'), 'genre_shooter': int(genre == 'Shooter'),
        'genre_action': int(genre == 'Action / strategy / story'), 'genre_sports': int(genre == 'Sports / racing'),
        'device_mobile': int(device == 'Mobile'), 'device_pc': int(device == 'PC'),
        'self_esteem': inp['self_esteem'], 'few_friends': inp['few_friends'], 'sports_level': inp['sports_level'],
    }
    return pd.DataFrame([row])[FEATURES]


def tier_of(score):
    return 0 if score <= TIER_LOW_MAX else 2 if score >= TIER_HIGH_MIN else 1


def tier_probabilities(pred_score, sigma):
    """P(Low/Medium/High) assuming the true IGDS9-SF score ~ Normal(prediction, cross-validated RMSE)."""
    pred = np.atleast_1d(np.asarray(pred_score, dtype=float))
    p_low = norm.cdf((TIER_LOW_MAX + 0.5 - pred) / sigma)
    p_high = 1 - norm.cdf((TIER_HIGH_MIN - 0.5 - pred) / sigma)
    probs = np.column_stack([p_low, 1 - p_low - p_high, p_high])
    return probs if np.ndim(pred_score) else probs[0]
