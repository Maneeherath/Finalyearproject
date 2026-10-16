# Unit and integration tests: eye analysis (eye_analysis.py) and study measures (eye_gaming_study.py)
import glob
import os

import cv2
import numpy as np
import pandas as pd
import pytest

import eye_analysis as ea

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTOS = os.path.join(ROOT, 'demo_run', 'eye_test')


@pytest.fixture(scope='module')
def analyzer():
    return ea.EyeAnalyzer()


@pytest.mark.parametrize('score,level', [(0, 0), (24.9, 0), (25, 1), (49.9, 1), (50, 2), (74.9, 2), (75, 3), (100, 3)])
def test_level_boundaries(score, level):
    assert ea.level_index(score) == level and ea.level_of(score) == ea.LEVEL_NAMES[level]


def test_calibration_maps_threshold_to_half_and_is_monotonic():
    t = 0.25
    assert ea.calibrate(t, t) == pytest.approx(0.5)
    assert ea.calibrate(0, t) == pytest.approx(0) and ea.calibrate(1, t) == pytest.approx(1)
    xs = np.linspace(0, 1, 101)
    assert np.all(np.diff(ea.calibrate(xs, t)) > 0)


def test_prep_crop_shape_and_range():
    crop = (np.random.default_rng(0).random((90, 70, 3)) * 255).astype('uint8')
    out = ea.prep_crop(crop, 48)
    assert out.shape == (48, 48) and out.dtype == np.uint8


def test_blank_image_reports_no_eyes(analyzer):
    r = analyzer.analyze(np.full((480, 640, 3), 128, np.uint8))
    assert r['fatigue_score'] is None and r['reason'] == 'no_eyes'


def test_scores_are_probabilities(analyzer):
    files = sorted(glob.glob(os.path.join(PHOTOS, '*', '*.jpg')))
    if not files:
        pytest.skip('test photos not available')
    for f in files:
        r = analyzer.analyze(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB))
        assert all(0 <= p <= 1 for p in r['p_closed'])
        if r['fatigue_score'] is not None:
            assert 0 <= r['fatigue_score'] <= 100


def test_open_and_closed_photos_are_separated(analyzer):
    """Real-world photos: most scorable photos must be classified correctly."""
    correct = total = 0
    for label in ('open', 'closed'):
        for f in sorted(glob.glob(os.path.join(PHOTOS, label, '*.jpg'))):
            r = analyzer.analyze(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB))
            if r['fatigue_score'] is None:
                continue
            total += 1
            correct += (r['fatigue_score'] >= 50) == (label == 'closed')
    if total == 0:
        pytest.skip('test photos not available')
    assert correct / total >= 0.75


# ── eye-gaming study measures ────────────────────────────────────────────────
import eye_gaming_study as egs  # noqa: E402


def series(seconds=60, fps=30, blink_every=2.0, blink_frames=4, open_ear=0.30, closed_ear=0.08):
    period = int(blink_every * fps)
    return [[k / fps, 1, closed_ear if k % period in range(period // 2, period // 2 + blink_frames) else open_ear, 0.1]
            for k in range(int(seconds * fps))]


def test_blink_rate_and_perclos_from_known_series():
    f = egs.compute_features(series())
    assert f['blinks'] == 30 and f['blink_rate_per_min'] == pytest.approx(30.0, abs=0.5)
    assert f['perclos_pct'] == pytest.approx(100 * 4 / 60, abs=0.2) and f['quality'] == 'ok'


def test_long_closures_are_not_blinks_but_count_in_perclos():
    fr = series(blink_frames=30)                     # 1 s closures
    f = egs.compute_features(fr)
    assert f['blinks'] == 0 and f['perclos_pct'] > 40


def test_single_dropped_frame_does_not_lose_a_blink():
    fr = series()
    for k in range(len(fr)):
        if k % 60 == 32:                             # drop one frame in the middle of every blink
            fr[k] = [fr[k][0], 0, None, None]
    assert egs.compute_features(fr)['blinks'] == 30


def test_quality_flag_when_face_not_visible():
    fr = series()
    for k in range(0, len(fr), 2):
        fr[k] = [fr[k][0], 0, None, None]
    assert egs.compute_features(fr)['quality'].startswith('low')


def test_too_few_frames_returns_none():
    assert egs.compute_features([[0.0, 1, 0.3, 0.1]]) is None


@pytest.mark.parametrize('ids,expected', [([], 'G001'), (['G001', 'G002'], 'G003'), (['G009', 'G004'], 'G010')])
def test_next_participant_id(ids, expected):
    df = pd.DataFrame({'participant_id': ids}) if ids else pd.DataFrame()
    assert egs.next_id(df) == expected
