# Unit tests: feature encoding, risk tiers and tier probabilities (igd_features.py)
import numpy as np
import pytest

import igd_features as F

BASE = dict(age=21, gender="Male", start_age=10, sports="Minor", sports_level=1, weekday_hours=2.0, weekend_hours=3.0,
            mode=F.MODES[0], genre=F.GENRES[0], device=F.DEVICES[0], sleep_hours=7.0, sleep_quality=7,
            self_esteem=4, few_friends=2)


def test_features_have_model_columns_in_order():
    X = F.features_from_inputs(BASE)
    assert list(X.columns) == F.FEATURES and X.shape == (1, 18)


def test_weekly_hours_formula():
    X = F.features_from_inputs({**BASE, 'weekday_hours': 3.5, 'weekend_hours': 6.0})
    assert X['weekly_hours'].iloc[0] == pytest.approx(5 * 3.5 + 2 * 6.0)


@pytest.mark.parametrize('mode,col', [('Online multiplayer', 'mode_online_multi'), ('Online single-player', 'mode_online_single')])
def test_mode_one_hot(mode, col):
    X = F.features_from_inputs({**BASE, 'mode': mode})
    others = [c for c in ('mode_online_multi', 'mode_online_single') if c != col]
    assert X[col].iloc[0] == 1 and all(X[c].iloc[0] == 0 for c in others)


def test_offline_mode_sets_no_mode_indicator():
    X = F.features_from_inputs({**BASE, 'mode': 'Offline single-player'})
    assert X['mode_online_multi'].iloc[0] == 0 and X['mode_online_single'].iloc[0] == 0


@pytest.mark.parametrize('genre,col', [('Battle royale', 'genre_br'), ('Shooter', 'genre_shooter'),
                                       ('Action / strategy / story', 'genre_action'), ('Sports / racing', 'genre_sports')])
def test_genre_one_hot(genre, col):
    X = F.features_from_inputs({**BASE, 'genre': genre})
    genre_cols = ['genre_br', 'genre_shooter', 'genre_action', 'genre_sports']
    assert X[col].iloc[0] == 1 and sum(X[c].iloc[0] for c in genre_cols) == 1


def test_casual_genre_sets_no_genre_indicator():
    X = F.features_from_inputs({**BASE, 'genre': 'Casual (puzzle / platform)'})
    assert sum(X[c].iloc[0] for c in ['genre_br', 'genre_shooter', 'genre_action', 'genre_sports']) == 0


@pytest.mark.parametrize('device,mobile,pc', [('Mobile', 1, 0), ('PC', 0, 1), ('Console', 0, 0), ('Tablet', 0, 0)])
def test_device_one_hot(device, mobile, pc):
    X = F.features_from_inputs({**BASE, 'device': device})
    assert (X['device_mobile'].iloc[0], X['device_pc'].iloc[0]) == (mobile, pc)


@pytest.mark.parametrize('gender,flag', [('Male', 1), ('Female', 0)])
def test_gender_flag(gender, flag):
    assert F.features_from_inputs({**BASE, 'gender': gender})['is_male'].iloc[0] == flag


@pytest.mark.parametrize('score,tier', [(9, 0), (16, 0), (17, 1), (31, 1), (32, 2), (45, 2)])
def test_tier_boundaries(score, tier):
    assert F.tier_of(score) == tier


@pytest.mark.parametrize('pred', [9, 12.5, 16.5, 20, 25, 31.5, 38, 45])
def test_tier_probabilities_sum_to_one(pred):
    p = F.tier_probabilities(pred, 5.13)
    assert p.shape == (3,) and p.sum() == pytest.approx(1.0) and (p >= 0).all()


def test_tier_probabilities_monotonic_in_score():
    preds = np.linspace(9, 45, 50)
    p = F.tier_probabilities(preds, 5.13)
    assert p.shape == (50, 3)
    assert np.all(np.diff(p[:, 2]) > 0) and np.all(np.diff(p[:, 0]) < 0)


def test_tier_probabilities_centre_of_band_is_most_likely():
    assert np.argmax(F.tier_probabilities(12, 5.13)) == 0
    assert np.argmax(F.tier_probabilities(24, 5.13)) == 1
    assert np.argmax(F.tier_probabilities(40, 5.13)) == 2


def test_dataset_loads_with_expected_size_and_range():
    raw = F.load_dataset()
    X = F.features_from_dataset(raw)
    y = raw['IGD Total'].astype(float)
    assert len(raw) == 322 and X.shape == (322, 18)
    assert y.min() >= 9 and y.max() <= 45
    assert not X.isna().any().any()
