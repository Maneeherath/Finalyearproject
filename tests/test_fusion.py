# Unit tests: late-fusion engine (fusion_engine.py)
import numpy as np
import pytest

import fusion_engine as fe


def test_behaviour_vote_normalises_and_reports_max_confidence():
    p, c = fe.behaviour_vote([2, 6, 2])
    assert p.sum() == pytest.approx(1.0) and c == pytest.approx(0.6)


@pytest.mark.parametrize('vote,arg_lo,arg_hi,cap', [
    (fe.cognitive_inhibition_vote, 0, 100, fe.COGNITIVE_LITERATURE_WEIGHT),
    (fe.eye_fatigue_vote, 0, 100, fe.EYE_FATIGUE_LITERATURE_WEIGHT)])
def test_prior_confidence_is_capped_and_zero_at_midpoint(vote, arg_lo, arg_hi, cap):
    assert vote(arg_lo)[1] == pytest.approx(cap)
    assert vote(arg_hi)[1] == pytest.approx(cap)
    assert vote((arg_lo + arg_hi) / 2)[1] == pytest.approx(0.0)


def test_wellbeing_vote_cap_and_direction():
    p_best, c_best = fe.wellbeing_vote(0, 0, 3)
    p_worst, c_worst = fe.wellbeing_vote(6, 6, 9)
    assert c_best == pytest.approx(fe.WELLBEING_LITERATURE_WEIGHT) == pytest.approx(c_worst)
    assert p_worst[2] > p_best[2] and p_worst[0] < p_best[0]


def test_one_clear_wellbeing_problem_is_not_diluted():
    lonely_only, c_lonely = fe.wellbeing_vote(0, 0, 9)          # strongly lonely, mood and anxiety fine
    all_bad, c_all = fe.wellbeing_vote(6, 6, 9)
    assert np.allclose(lonely_only, all_bad) and c_lonely == pytest.approx(c_all)
    assert lonely_only[2] > fe.wellbeing_vote(0, 0, 3)[0][2]


def test_cognitive_weight_matches_published_effect_size():
    d = fe.INHIBITION_EFFECT_D
    assert fe.COGNITIVE_LITERATURE_WEIGHT == pytest.approx(d / np.sqrt(d ** 2 + 4))
    assert fe.COGNITIVE_LITERATURE_WEIGHT == pytest.approx(0.27, abs=0.005)


def test_prior_lean_increases_risk_monotonically():
    highs = [fe.cognitive_inhibition_vote(v)[0][2] for v in range(0, 101, 10)]
    assert np.all(np.diff(highs) > 0)
    for v in range(0, 101, 25):
        assert fe.cognitive_inhibition_vote(v)[0].sum() == pytest.approx(1.0)


def test_fused_distribution_is_valid_and_weights_sum_to_one():
    votes = [fe.behaviour_vote([0.1, 0.6, 0.3]), fe.cognitive_inhibition_vote(75), fe.eye_fatigue_vote(20),
             fe.wellbeing_vote(3, 3, 6)]
    fused, w = fe.fuse(votes)
    assert fused.sum() == pytest.approx(1.0) and (fused > 0).all()
    assert sum(w.values()) == pytest.approx(1.0) and list(w) == fe.VOTE_NAMES


def test_zero_confidence_vote_has_no_effect():
    beh = fe.behaviour_vote([0.2, 0.5, 0.3])
    alone, _ = fe.fuse([beh])
    with_missing, _ = fe.fuse([beh, (np.ones(3) / 3, 0.0), (np.array([0.9, 0.05, 0.05]), 0.0)])
    assert np.allclose(alone, with_missing, atol=1e-6)


def test_single_vote_returns_that_vote():
    p = np.array([0.2, 0.5, 0.3])
    fused, _ = fe.fuse([(p, 0.5)])
    assert np.allclose(fused, p)


def test_confident_behaviour_vote_is_not_overridden_by_priors():
    beh = fe.behaviour_vote([0.05, 0.9, 0.05])
    fused, _ = fe.fuse([beh, fe.cognitive_inhibition_vote(100), fe.eye_fatigue_vote(100), fe.wellbeing_vote(6, 6, 9)])
    assert np.argmax(fused) == 1


def test_consistent_priors_raise_high_probability():
    beh = fe.behaviour_vote([0.15, 0.6, 0.25])
    base, _ = fe.fuse([beh])
    worse, _ = fe.fuse([beh, fe.cognitive_inhibition_vote(100), fe.eye_fatigue_vote(100), fe.wellbeing_vote(6, 6, 9)])
    better, _ = fe.fuse([beh, fe.cognitive_inhibition_vote(0), fe.eye_fatigue_vote(0), fe.wellbeing_vote(0, 0, 3)])
    assert worse[2] > base[2] > better[2]
