# Unit tests: cognitive test sequences and scoring (research_cognitive_tests.py)
import random

import numpy as np
import pytest

import research_cognitive_tests as rct


def test_d_prime_perfect_and_chance():
    assert rct.d_prime(4, 0, 0, 6) > 2
    assert rct.d_prime(2, 2, 3, 3) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize('seed', range(20))
def test_nback_sequence_has_exact_targets(seed):
    random.seed(seed)
    seq, targets = rct._make_nback_sequence()
    assert len(seq) == rct.NB_TRIALS and sum(targets) == rct.NB_TARGETS
    for i in range(rct.NB_N, len(seq)):
        assert (seq[i] == seq[i - rct.NB_N]) == targets[i]          # no accidental matches


def test_nback_scoring_known_log():
    log = [{'target': True, 'said': True, 'rt': 800}] * 3 + [{'target': True, 'said': False, 'rt': 900}] + \
          [{'target': False, 'said': False, 'rt': 700}] * 5 + [{'target': False, 'said': True, 'rt': 600}]
    r = rct._score_nback(log)
    assert r['accuracy'] == 80 and r['hit_rate'] == 75 and r['false_alarm_rate'] == 17
    assert r['mean_rt'] == round(np.mean([t['rt'] for t in log]))


@pytest.mark.parametrize('seed', range(20))
def test_gonogo_sequence(seed):
    random.seed(seed)
    seq = rct._make_gng_sequence()
    assert len(seq) == rct.GNG_TRIALS and seq.count(False) == rct.GNG_NOGO and seq[:2] == [True, True]


def test_gonogo_scoring_known_log():
    log = [{'go': True, 'responded': True, 'rt': 400}] * 11 + [{'go': True, 'responded': False, 'rt': None}] + \
          [{'go': False, 'responded': True, 'rt': 300}] + [{'go': False, 'responded': False, 'rt': None}] * 3
    r = rct._score_gng(log)
    assert r['commission_errors'] == 1 and r['commission_rate'] == 25 and r['omission_errors'] == 1
    assert r['go_accuracy'] == round(11 / 12 * 100) and r['mean_go_rt'] == 400
    assert r['balanced_accuracy'] == round((11 / 12 + 0.75) / 2 * 100)


@pytest.mark.parametrize('seed', range(20))
def test_task_switching_trials_follow_aabb(seed):
    random.seed(seed)
    trials = rct._make_ts_trials()
    tasks = [t['task'] for t in trials]
    assert len(trials) == rct.TS_TRIALS
    switches = [tasks[i] != tasks[i - 1] for i in range(1, len(tasks))]
    assert sum(switches) == 6 and len(switches) - sum(switches) == 6
    for t in trials:
        digit = int(next(c for c in t['stimulus'] if c.isdigit()))
        letter = next(c for c in t['stimulus'] if c.isalpha())
        expected = ('left' if digit % 2 == 0 else 'right') if t['task'] == 'number' else \
                   ('left' if letter in rct.TS_CONSONANTS else 'right')
        assert t['answer'] == expected


def test_task_switching_switch_cost():
    log = [{'correct': True, 'rt': 900, 'switch': False}] + \
          [{'correct': True, 'rt': 800, 'switch': False}] * 6 + [{'correct': True, 'rt': 1100, 'switch': True}] * 6
    assert rct._score_ts(log)['switch_cost'] == 300


@pytest.mark.parametrize('seed', range(20))
def test_mental_rotation_trials(seed):
    random.seed(seed)
    trials = rct._make_mr_trials()
    assert len(trials) == len(rct.MR_ANGLES) * rct.MR_REPEATS == 12
    for a in rct.MR_ANGLES:                       # every angle once normal and once mirrored
        assert sorted(t['mirrored'] for t in trials if t['angle'] == a) == [False, True]


def test_mental_rotation_slope_positive_for_rotation_effect():
    log = [{'angle': a, 'correct': True, 'rt': 600 + 4 * min(a, 360 - a)} for a in rct.MR_ANGLES]
    r = rct._score_mr(log)
    assert r['accuracy'] == 100 and r['slope_ms_per_deg'] == pytest.approx(4.0, abs=0.01)


def test_domain_scores():
    battery = {'nback': {'accuracy': 90}, 'go_nogo': {'balanced_accuracy': 80}, 'task_switching': {'accuracy': 70}}
    assert rct.domain_scores(battery) == (90, 75)
