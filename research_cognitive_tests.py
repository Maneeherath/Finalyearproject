# ============================================================
#  RESEARCH-BASED COGNITIVE BATTERY
#  Four paradigms taken from the published cognitive-psychology
#  and video-gaming literature (see REFERENCES below):
#    1. 2-Back Working Memory Task      (Kirchner, 1958; Colzato et al., 2013)
#    2. Task-Switching (alternating runs)(Rogers & Monsell, 1995; Green et al., 2012)
#    3. Mental Rotation of letters      (Cooper & Shepard, 1973; Feng et al., 2007)
#    4. Go/No-Go Response Inhibition    (Donders, 1868; Ding et al., 2014)
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import random
import time

import numpy as np
import streamlit as st
from scipy.stats import norm


REFERENCES = {
    "2-Back Working Memory": [
        "Kirchner, W. K. (1958). Age differences in short-term retention of rapidly changing information. "
        "Journal of Experimental Psychology, 55(4), 352-358.",
        "Owen, A. M., McMillan, K. M., Laird, A. R., & Bullmore, E. (2005). N-back working memory paradigm: "
        "A meta-analysis of normative functional neuroimaging studies. Human Brain Mapping, 25(1), 46-59.",
        "Colzato, L. S., van den Wildenberg, W. P. M., Zmigrod, S., & Hommel, B. (2013). Action video gaming and "
        "cognitive control: playing first person shooter games is associated with improvement in working memory "
        "but not action inhibition. Psychological Research, 77(2), 234-239.",
        "Hautus, M. J. (1995). Corrections for extreme proportions and their biasing effects on estimated values "
        "of d'. Behavior Research Methods, Instruments, & Computers, 27(1), 46-51.",
    ],
    "Go/No-Go Response Inhibition": [
        "Donders, F. C. (1868/1969). On the speed of mental processes. Acta Psychologica, 30, 412-431.",
        "Ding, W., Sun, J., Sun, Y., Chen, X., Zhou, Y., Zhuang, Z., Li, L., Zhang, Y., Xu, J., & Du, Y. (2014). "
        "Trait impulsivity and impaired prefrontal impulse inhibition function in adolescents with internet gaming "
        "addiction revealed by a Go/No-Go fMRI study. Behavioral and Brain Functions, 10, 20.",
        "Littel, M., van den Berg, I., Luijten, M., van Rooij, A. J., Keemink, L., & Franken, I. H. A. (2012). "
        "Error processing and response inhibition in excessive computer game players: an event-related potential "
        "study. Addiction Biology, 17(5), 934-947.",
    ],
    "Task-Switching": [
        "Rogers, R. D., & Monsell, S. (1995). Predictable switch between simple cognitive tasks. "
        "Journal of Experimental Psychology: General, 124(2), 207-231.",
        "Monsell, S. (2003). Task switching. Trends in Cognitive Sciences, 7(3), 134-140.",
        "Green, C. S., Sugarman, M. A., Medford, K., Klobusicky, E., & Bavelier, D. (2012). The effect of action "
        "video game experience on task-switching. Computers in Human Behavior, 28(3), 984-994.",
    ],
    "Mental Rotation": [
        "Shepard, R. N., & Metzler, J. (1971). Mental rotation of three-dimensional objects. Science, 171(3972), 701-703.",
        "Cooper, L. A., & Shepard, R. N. (1973). Chronometric studies of the rotation of mental images. "
        "In W. G. Chase (Ed.), Visual Information Processing (pp. 75-176). Academic Press.",
        "Feng, J., Spence, I., & Pratt, J. (2007). Playing an action video game reduces gender differences in "
        "spatial cognition. Psychological Science, 18(10), 850-855.",
    ],
}

# ── Protocol constants ────────────────────────────────────────────────────────
NB_TRIALS, NB_TARGETS, NB_N = 12, 4, 2          # short version: 10 scored trials, 4 matches
NB_LETTERS = ['B', 'F', 'K', 'M', 'R', 'T']

GNG_TRIALS, GNG_NOGO = 16, 4          # 25% No-Go keeps "Go" the prepotent response
GNG_WINDOW, GNG_ISI = 1.0, 0.5        # seconds: response window / blank inter-trial interval
GNG_TICK = 0.05                       # loop tick: timing resolution of onsets/omissions and click pickup

TS_TRIALS = 13                        # trial 1 is neither switch nor repeat -> 6 switch + 6 repeat
TS_VOWELS, TS_CONSONANTS = ['A', 'E', 'I', 'U'], ['G', 'K', 'M', 'R']

MR_LETTERS = ['R', 'F', 'G', 'J', 'L', 'P']   # asymmetric letters (normal vs mirror is distinguishable)
MR_ANGLES = [0, 60, 120, 180, 240, 300]

RESET_KEYS = ['nb_seq', 'nb_targets', 'nb_idx', 'nb_start', 'nb_log', 'nb_result',
              'gng_seq', 'gng_idx', 'gng_phase', 'gng_t0', 'gng_log', 'gng_result',
              'ts_trials', 'ts_idx', 'ts_start', 'ts_log', 'ts_result',
              'mr_trials', 'mr_idx', 'mr_start', 'mr_log', 'mr_result']


def init_state():
    for k, d in [('nb_seq', []), ('nb_targets', []), ('nb_idx', 0), ('nb_start', 0.0), ('nb_log', []), ('nb_result', None),
                 ('gng_seq', []), ('gng_idx', 0), ('gng_phase', 'start'), ('gng_t0', 0.0), ('gng_log', []), ('gng_result', None),
                 ('ts_trials', []), ('ts_idx', 0), ('ts_start', 0.0), ('ts_log', []), ('ts_result', None),
                 ('mr_trials', []), ('mr_idx', 0), ('mr_start', 0.0), ('mr_log', []), ('mr_result', None)]:
        if k not in st.session_state:
            st.session_state[k] = d


def _elapsed_ms(t0):
    return round((time.time() - t0) * 1000)


def _mean(xs):
    return round(float(np.mean(xs))) if xs else 0


def d_prime(hits, misses, fas, crs):
    # Log-linear correction (Hautus, 1995) keeps z finite when a rate is 0% or 100%
    hr = (hits + 0.5) / (hits + misses + 1)
    far = (fas + 0.5) / (fas + crs + 1)
    return float(norm.ppf(hr) - norm.ppf(far))


# ════════════════════════════════════════════════════════════
#  TEST 1: 2-BACK WORKING MEMORY
# ════════════════════════════════════════════════════════════
def _make_nback_sequence():
    target_pos = set(random.sample(range(NB_N, NB_TRIALS), NB_TARGETS))
    seq, targets = [], []
    for i in range(NB_TRIALS):
        if i in target_pos:
            seq.append(seq[i - NB_N])
            targets.append(True)
        else:
            avoid = seq[i - NB_N] if i >= NB_N else None
            seq.append(random.choice([l for l in NB_LETTERS if l != avoid]))
            targets.append(False)
    return seq, targets


def _score_nback(log):
    hits = sum(1 for t in log if t['target'] and t['said'])
    misses = sum(1 for t in log if t['target'] and not t['said'])
    fas = sum(1 for t in log if not t['target'] and t['said'])
    crs = sum(1 for t in log if not t['target'] and not t['said'])
    return {
        'accuracy': round((hits + crs) / len(log) * 100),
        'hit_rate': round(hits / max(1, hits + misses) * 100),
        'false_alarm_rate': round(fas / max(1, fas + crs) * 100),
        'd_prime': round(d_prime(hits, misses, fas, crs), 2),
        'mean_rt': _mean([t['rt'] for t in log]),
    }


def render_nback():
    ss = st.session_state
    st.markdown(f"Letters appear one at a time. Press **Match** if the current letter is the same as the one shown "
                f"**{NB_N} letters earlier**, otherwise press **No Match**. Measures working-memory *updating*.")

    if ss['nb_result'] is None:
        if not ss['nb_seq']:
            if st.button("▶️ Start 2-Back Task", key="nb_go"):
                ss['nb_seq'], ss['nb_targets'] = _make_nback_sequence()
                ss['nb_idx'], ss['nb_log'], ss['nb_start'] = 0, [], time.time()
                st.rerun()
        else:
            i = ss['nb_idx']
            st.markdown(f"<h1 style='text-align: center; font-size: 4rem;'>{ss['nb_seq'][i]}</h1>", unsafe_allow_html=True)
            st.write(f"**Trial {i+1}/{NB_TRIALS}**")

            if i < NB_N:
                st.caption("Memorize this letter — there is nothing to compare it with yet.")
                if st.button("Next ➡️", key=f"nb_next_{i}"):
                    ss['nb_idx'] += 1
                    ss['nb_start'] = time.time()
                    st.rerun()
            else:
                c1, c2 = st.columns(2)
                said = None
                if c1.button("✅ Match", key=f"nb_y_{i}", use_container_width=True):
                    said = True
                if c2.button("❌ No Match", key=f"nb_n_{i}", use_container_width=True):
                    said = False
                if said is not None:
                    ss['nb_log'].append({'target': ss['nb_targets'][i], 'said': said, 'rt': _elapsed_ms(ss['nb_start'])})
                    ss['nb_idx'] += 1
                    ss['nb_start'] = time.time()
                    if ss['nb_idx'] >= NB_TRIALS:
                        ss['nb_result'] = _score_nback(ss['nb_log'])
                    st.rerun()

    if ss['nb_result'] is not None:
        r = ss['nb_result']
        c1, c2, c3 = st.columns(3)
        c1.metric("Working-Memory Accuracy", f"{r['accuracy']}%")
        c2.metric("Sensitivity (d′)", f"{r['d_prime']:.2f}", "Good" if r['d_prime'] >= 1.5 else "Low")
        c3.metric("Mean Decision Time", f"{r['mean_rt']} ms")
        st.caption(f"Hit rate {r['hit_rate']}% · False-alarm rate {r['false_alarm_rate']}%")


# ════════════════════════════════════════════════════════════
#  TEST 4: GO/NO-GO RESPONSE INHIBITION
# ════════════════════════════════════════════════════════════
def _make_gng_sequence():
    # First two trials are always Go so the prepotent response is established before any No-Go
    rest = [False] * GNG_NOGO + [True] * (GNG_TRIALS - 2 - GNG_NOGO)
    random.shuffle(rest)
    return [True, True] + rest


def _score_gng(log):
    go = [t for t in log if t['go']]
    nogo = [t for t in log if not t['go']]
    hits = [t for t in go if t['responded']]
    commissions = sum(1 for t in nogo if t['responded'])
    go_acc = len(hits) / len(go)
    nogo_acc = 1 - commissions / len(nogo)
    return {
        'go_accuracy': round(go_acc * 100),
        'omission_errors': len(go) - len(hits),
        'commission_errors': commissions,
        'commission_rate': round(commissions / len(nogo) * 100),
        'mean_go_rt': _mean([t['rt'] for t in hits]),
        'balanced_accuracy': round((go_acc + nogo_acc) / 2 * 100),
    }


def _gng_record(responded, rt):
    ss = st.session_state
    ss['gng_log'].append({'go': ss['gng_seq'][ss['gng_idx']], 'responded': responded, 'rt': rt})
    ss['gng_idx'] += 1
    ss['gng_phase'], ss['gng_t0'] = 'isi', time.time()
    if ss['gng_idx'] >= GNG_TRIALS:
        ss['gng_phase'] = 'done'
        ss['gng_result'] = _score_gng(ss['gng_log'])


def _gng_press():
    # Button callback: runs at the start of the rerun the click triggers, so RT is taken at click time.
    # Presses during the blank interval (anticipations) or after the window are ignored.
    ss = st.session_state
    if ss['gng_phase'] == 'stim' and time.time() - ss['gng_t0'] < GNG_WINDOW:
        _gng_record(responded=True, rt=_elapsed_ms(ss['gng_t0']))


def _gng_run_trials(stim_slot, trial_slot):
    # Timed trial loop inside the script run. A PRESS click interrupts this run at the next
    # Streamlit call, so the stimulus is redrawn every tick to keep click pickup within ~GNG_TICK.
    ss = st.session_state
    while ss['gng_phase'] in ('isi', 'stim'):
        now = time.time()
        if ss['gng_phase'] == 'isi' and now - ss['gng_t0'] >= GNG_ISI:
            ss['gng_phase'], ss['gng_t0'] = 'stim', now
        elif ss['gng_phase'] == 'stim' and now - ss['gng_t0'] >= GNG_WINDOW:
            # Withheld response: omission on Go, correct rejection on No-Go
            _gng_record(responded=False, rt=None)
            continue

        i = ss['gng_idx']
        if ss['gng_phase'] == 'stim':
            symbol = "🟢" if ss['gng_seq'][i] else "🔴"
        else:
            symbol = "<span style='opacity: 0.5;'>+</span>"
        stim_slot.markdown(f"<div style='text-align: center; font-size: 5rem; height: 8rem; line-height: 8rem;'>{symbol}</div>",
                           unsafe_allow_html=True)
        trial_slot.markdown(f"**Trial {i+1}/{GNG_TRIALS}**")
        time.sleep(GNG_TICK)
    st.rerun()   # task finished: rerun so results and the battery summary refresh


def render_go_nogo():
    ss = st.session_state
    st.markdown("Press **PRESS** as fast as you can when a 🟢 **green** circle appears. "
                "**Do NOT press** when a 🔴 **red** circle appears — just wait. Measures impulse (response) inhibition, "
                "the function found to be impaired in internet-gaming addiction (Ding et al., 2014).")

    if ss['gng_result'] is None:
        if ss['gng_phase'] == 'start':
            if st.button("▶️ Start Go/No-Go Task", key="gng_go"):
                ss['gng_seq'] = _make_gng_sequence()
                ss['gng_idx'], ss['gng_log'] = 0, []
                ss['gng_phase'], ss['gng_t0'] = 'isi', time.time()
                st.rerun()
        else:
            stim_slot, trial_slot = st.empty(), st.empty()
            st.button("🔘 PRESS", key="gng_press", use_container_width=True, type="primary", on_click=_gng_press)
            st.caption(f"Each circle stays for {GNG_WINDOW:.1f}s. Not pressing in time counts as a withheld response.")
            _gng_run_trials(stim_slot, trial_slot)

    if ss['gng_result'] is not None:
        r = ss['gng_result']
        c1, c2, c3 = st.columns(3)
        c1.metric("Go Accuracy", f"{r['go_accuracy']}%")
        impulsive = r['commission_rate'] >= 40
        c2.metric("Commission Errors (No-Go)", f"{r['commission_errors']}/{GNG_NOGO}",
                  "Impulsive" if impulsive else "Controlled", delta_color="inverse" if impulsive else "normal")
        c3.metric("Mean Go RT", f"{r['mean_go_rt']} ms")
        st.caption(f"Inhibition (balanced) accuracy {r['balanced_accuracy']}% · Omission errors {r['omission_errors']}")


# ════════════════════════════════════════════════════════════
#  TEST 2: TASK-SWITCHING (ALTERNATING RUNS, AABB)
# ════════════════════════════════════════════════════════════
def _make_ts_trials():
    pattern = ['number', 'number', 'letter', 'letter']
    offset = random.randrange(4)
    trials = []
    for i in range(TS_TRIALS):
        digit = random.randint(2, 9)
        letter = random.choice(TS_VOWELS + TS_CONSONANTS)
        pair = f"{digit}{letter}" if random.random() < 0.5 else f"{letter}{digit}"
        task = pattern[(i + offset) % 4]
        if task == 'number':
            answer = 'left' if digit % 2 == 0 else 'right'
        else:
            answer = 'left' if letter in TS_CONSONANTS else 'right'
        trials.append({'task': task, 'stimulus': pair, 'answer': answer})
    return trials


def _score_ts(log):
    scored = log[1:]   # trial 1 has no preceding task
    rep_rt = [t['rt'] for t in scored if t['correct'] and not t['switch']]
    sw_rt = [t['rt'] for t in scored if t['correct'] and t['switch']]
    rep, sw = _mean(rep_rt), _mean(sw_rt)
    return {
        'accuracy': round(sum(t['correct'] for t in log) / len(log) * 100),
        'repeat_rt': rep,
        'switch_rt': sw,
        'switch_cost': sw - rep if rep_rt and sw_rt else 0,
    }


def render_task_switching():
    ss = st.session_state
    st.markdown("Each trial shows a digit–letter pair. The coloured cue tells you which task to do. "
                "The task changes every **two** trials. Use the same two buttons for both tasks:  \n"
                "🔢 **NUMBER** → ⬅️ Even / Odd ➡️   ·   🔤 **LETTER** → ⬅️ Consonant / Vowel ➡️  \n"
                "The *switch cost* (slower answers right after the task changes) measures cognitive flexibility.")

    if ss['ts_result'] is None:
        if not ss['ts_trials']:
            if st.button("▶️ Start Task-Switching Test", key="ts_go"):
                ss['ts_trials'] = _make_ts_trials()
                ss['ts_idx'], ss['ts_log'], ss['ts_start'] = 0, [], time.time()
                st.rerun()
        else:
            i = ss['ts_idx']
            trial = ss['ts_trials'][i]
            if trial['task'] == 'number':
                cue, bg = "🔢 NUMBER — is the digit EVEN or ODD?", "#2563eb"
            else:
                cue, bg = "🔤 LETTER — is the letter a CONSONANT or VOWEL?", "#ea580c"
            st.markdown(f"<div style='background: {bg}; color: white; border-radius: 10px; padding: 0.4rem; "
                        f"text-align: center; font-weight: 700;'>{cue}</div>"
                        f"<h1 style='text-align: center; font-size: 4rem; letter-spacing: 6px;'>{trial['stimulus']}</h1>",
                        unsafe_allow_html=True)
            st.write(f"**Trial {i+1}/{TS_TRIALS}**")

            c1, c2 = st.columns(2)
            choice = None
            if c1.button("⬅️ EVEN · CONSONANT", key=f"ts_l_{i}", use_container_width=True):
                choice = 'left'
            if c2.button("ODD · VOWEL ➡️", key=f"ts_r_{i}", use_container_width=True):
                choice = 'right'
            if choice is not None:
                ss['ts_log'].append({
                    'task': trial['task'],
                    'switch': i > 0 and trial['task'] != ss['ts_trials'][i-1]['task'],
                    'correct': choice == trial['answer'],
                    'rt': _elapsed_ms(ss['ts_start']),
                })
                ss['ts_idx'] += 1
                ss['ts_start'] = time.time()
                if ss['ts_idx'] >= TS_TRIALS:
                    ss['ts_result'] = _score_ts(ss['ts_log'])
                st.rerun()

    if ss['ts_result'] is not None:
        r = ss['ts_result']
        c1, c2, c3 = st.columns(3)
        c1.metric("Task-Switch Accuracy", f"{r['accuracy']}%")
        c2.metric("Switch Cost", f"{r['switch_cost']} ms", "Flexible" if r['switch_cost'] < 300 else "Rigid",
                  delta_color="normal" if r['switch_cost'] < 300 else "inverse")
        c3.metric("Repeat / Switch RT", f"{r['repeat_rt']} / {r['switch_rt']} ms")


# ════════════════════════════════════════════════════════════
#  TEST 3: MENTAL ROTATION (COOPER & SHEPARD LETTER PARADIGM)
# ════════════════════════════════════════════════════════════
def _make_mr_trials():
    angles = list(MR_ANGLES)                      # one trial per angle
    mirrored = [True] * 3 + [False] * 3
    random.shuffle(angles)
    random.shuffle(mirrored)
    return [{'letter': random.choice(MR_LETTERS), 'angle': a, 'mirrored': m} for a, m in zip(angles, mirrored)]


def _score_mr(log):
    correct = [t for t in log if t['correct']]
    disparity = [min(t['angle'], 360 - t['angle']) for t in correct]
    rts = [t['rt'] for t in correct]
    # Linear RT-by-angle slope is the signature of mental rotation (Shepard & Metzler, 1971)
    slope = float(np.polyfit(disparity, rts, 1)[0]) if len(set(disparity)) >= 2 else 0.0
    return {
        'accuracy': round(len(correct) / len(log) * 100),
        'mean_rt': _mean(rts),
        'slope_ms_per_deg': round(slope, 2),
    }


def render_mental_rotation():
    ss = st.session_state
    st.markdown("A letter appears rotated. Decide whether it is a **normal** letter or a **mirror image**, "
                "turning it upright in your mind. Measures visuospatial processing — the ability that action-video-game "
                "play has been shown to improve (Feng et al., 2007).")

    if ss['mr_result'] is None:
        if not ss['mr_trials']:
            if st.button("▶️ Start Mental Rotation Test", key="mr_go"):
                ss['mr_trials'] = _make_mr_trials()
                ss['mr_idx'], ss['mr_log'], ss['mr_start'] = 0, [], time.time()
                st.rerun()
        else:
            i = ss['mr_idx']
            trial = ss['mr_trials'][i]
            flip = -1 if trial['mirrored'] else 1
            st.markdown(f"<div style='display: flex; justify-content: center; align-items: center; height: 9rem;'>"
                        f"<span style='display: inline-block; font: 700 6rem Arial, Helvetica, sans-serif; "
                        f"transform: rotate({trial['angle']}deg) scaleX({flip});'>{trial['letter']}</span></div>",
                        unsafe_allow_html=True)
            st.write(f"**Trial {i+1}/{len(ss['mr_trials'])}**")

            c1, c2 = st.columns(2)
            said_mirror = None
            if c1.button("✅ Normal", key=f"mr_n_{i}", use_container_width=True):
                said_mirror = False
            if c2.button("🔁 Mirrored", key=f"mr_m_{i}", use_container_width=True):
                said_mirror = True
            if said_mirror is not None:
                ss['mr_log'].append({'angle': trial['angle'], 'correct': said_mirror == trial['mirrored'],
                                     'rt': _elapsed_ms(ss['mr_start'])})
                ss['mr_idx'] += 1
                ss['mr_start'] = time.time()
                if ss['mr_idx'] >= len(ss['mr_trials']):
                    ss['mr_result'] = _score_mr(ss['mr_log'])
                st.rerun()

    if ss['mr_result'] is not None:
        r = ss['mr_result']
        c1, c2, c3 = st.columns(3)
        c1.metric("Rotation Accuracy", f"{r['accuracy']}%")
        c2.metric("Mean RT (correct)", f"{r['mean_rt']} ms")
        c3.metric("Rotation Rate", f"{r['slope_ms_per_deg']} ms/°")


# ════════════════════════════════════════════════════════════
#  PAGE SECTION + AGGREGATION
# ════════════════════════════════════════════════════════════
def render_battery():
    init_state()

    with st.expander("📚 Research Basis & References"):
        for test_name, refs in REFERENCES.items():
            st.markdown(f"**{test_name}**")
            for ref in refs:
                st.markdown(f"- {ref}")

    with st.expander("🧩 Test 1 — 2-Back Task (Working-Memory Updating)", expanded=True):
        render_nback()
    with st.expander("🔀 Test 2 — Task-Switching (Cognitive Flexibility)", expanded=True):
        render_task_switching()
    with st.expander("🔄 Test 3 — Mental Rotation (Visuospatial Processing)", expanded=True):
        render_mental_rotation()
    # Go/No-Go runs a timed loop that holds the script, so it goes last on the page
    with st.expander("🛑 Test 4 — Go/No-Go Task (Response Inhibition)", expanded=True):
        render_go_nogo()


def battery_results():
    """Completed battery results, keyed by test."""
    ss = st.session_state
    out = {}
    for name, key in [('nback', 'nb_result'), ('task_switching', 'ts_result'),
                      ('mental_rotation', 'mr_result'), ('go_nogo', 'gng_result')]:
        if ss.get(key) is not None:
            out[name] = ss[key]
    return out


def domain_scores(battery):
    """Map battery results onto the model's memory / attention domains (percent scores)."""
    memory = battery['nback']['accuracy']
    attention = round((battery['go_nogo']['balanced_accuracy'] + battery['task_switching']['accuracy']) / 2)
    # Mental rotation is a visuospatial measure; the trained models have no matching feature,
    # so it is reported separately rather than forced into memory or attention.
    return memory, attention
