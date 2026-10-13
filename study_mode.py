# ============================================================
#  VALIDATION STUDY MODE
#  Runs the full-system test with real participants and records
#  everything needed to compare the system's risk output against
#  a validated self-report scale and to measure usability:
#    - IGDS9-SF  (Pontes & Griffiths, 2015)  -> criterion measure
#    - SUS       (Brooke, 1996)              -> usability
#  Each completed session is appended to data/study/study_responses.csv
#  and analysed by analyze_study.py.
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import csv
import os
import time

import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.join(BASE, 'data', 'study')
STUDY_CSV = os.path.join(STUDY_DIR, 'study_responses.csv')

LIKERT_5 = ["Never", "Rarely", "Sometimes", "Often", "Very Often"]

# Pontes, H. M., & Griffiths, M. D. (2015). Measuring DSM-5 internet gaming disorder: Development and
# validation of a short psychometric scale. Computers in Human Behavior, 45, 137-143.
IGDS_INSTRUCTIONS = ("These questions will ask you about your gaming activity during the past year "
                     "(i.e., last 12 months). By gaming activity we understand any gaming-related activity that "
                     "has been played either from a computer/laptop or from a gaming console or any other kind of "
                     "device (e.g., mobile phone, tablet, etc.) both online and/or offline.")
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

# Brooke, J. (1996). SUS: A "quick and dirty" usability scale. In P. W. Jordan et al. (Eds.),
# Usability Evaluation in Industry (pp. 189-194). Taylor & Francis.
SUS_ANCHORS = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
SUS_ITEMS = [
    "I think that I would like to use this system frequently.",
    "I found the system unnecessarily complex.",
    "I thought the system was easy to use.",
    "I think that I would need the support of a technical person to be able to use this system.",
    "I found the various functions in this system were well integrated.",
    "I thought there was too much inconsistency in this system.",
    "I would imagine that most people would learn to use this system very quickly.",
    "I found the system very cumbersome to use.",
    "I felt very confident using the system.",
    "I needed to learn a lot of things before I could get going with this system.",
]

INFO_SHEET = """
**Study:** Computational Modeling of Cognitive Performance & Gaming Behavior (KDU, BSc Computer Science final-year project)

**What you will do (about 15 minutes):**
1. Answer a 9-question gaming questionnaire (IGDS9-SF).
2. Complete four short cognitive tests in the app.
3. Optionally take one webcam photo for the eye-fatigue check (the photo is analysed in memory and **not saved**).
4. Fill in the gaming & lifestyle survey and view your result.
5. Rate how easy the app was to use (SUS).

**Your data:** Only an anonymous participant code is stored — no name, student number, email or images.
Results are used only for this research project. You can stop at any time without giving a reason, and
unsaved answers are discarded. This tool is a research prototype, **not** a medical diagnosis.
"""

STUDY_KEYS = ['study_consent', 'study_pid', 'study_igds', 'study_sus', 'study_comment', 'study_saved_pid']


def sus_score(responses):
    """responses: ten 1-5 ratings in item order -> 0-100 SUS score (Brooke, 1996)."""
    total = sum((r - 1) if i % 2 == 0 else (5 - r) for i, r in enumerate(responses))
    return total * 2.5


def igds_total(responses):
    """responses: nine 1-5 ratings -> total score (9-45) and number of criteria endorsed 'Very Often'."""
    return sum(responses), sum(1 for r in responses if r == 5)


def _next_participant_id():
    n = 0
    if os.path.exists(STUDY_CSV):
        with open(STUDY_CSV, newline='', encoding='utf-8') as f:
            n = sum(1 for _ in csv.DictReader(f))
    return f"P{n + 1:03d}"


def _append_row(row):
    os.makedirs(STUDY_DIR, exist_ok=True)
    exists = os.path.exists(STUDY_CSV)
    fields = list(row.keys())
    if exists:
        with open(STUDY_CSV, newline='', encoding='utf-8') as f:
            header = next(csv.reader(f), [])
        # Keep the existing column order; add any new columns at the end
        fields = header + [k for k in row if k not in header]
        if fields != header:
            with open(STUDY_CSV, newline='', encoding='utf-8') as f:
                old_rows = list(csv.DictReader(f))
            with open(STUDY_CSV, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(old_rows)
    with open(STUDY_CSV, 'a' if exists else 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)


def build_row(ss):
    """Flatten one completed session into a CSV row."""
    a = ss['last_assessment']
    cr = ss['cognitive_results']
    b = cr['battery']
    igds = ss['study_igds']
    sus = ss['study_sus']
    igds_sum, igds_criteria = igds_total(igds)

    row = {
        'participant_id': ss['study_pid'],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    row.update({f'input_{k}': v for k, v in a['inputs'].items()})
    row.update({
        'input_total_hours_week': a['total_hours'], 'input_sleep_hours': a['sleep_hours'],
        # System outputs (behaviour = real-data IGDS9-SF model)
        'sys_behaviour_pred': a['rf_pred'], 'sys_behaviour_conf': round(a['rf_conf'], 4),
        'sys_cognitive_pred': a['cog_pred'], 'sys_eye_pred': a['eye_pred'],
        'sys_fused_pred': a['fused_pred'], 'sys_fused_idx': a['fused_idx'],
        'sys_pred_igds': round(a['pred_igds'], 2), 'sys_archetype': a['archetype'],
        'sys_w_behaviour': round(a['rf_w'], 4), 'sys_w_cognitive': round(a['cog_w'], 4), 'sys_w_eye': round(a['eye_w'], 4),
        'sys_used_measured_eye': a['used_measured_eye'],
        # Cognitive battery
        'cog_memory_pct': cr['memory_score_pct'], 'cog_attention_pct': cr['attention_score_pct'],
        'cog_go_rt_ms': cr['reaction_time_ms'],
        'nback_accuracy': b['nback']['accuracy'], 'nback_dprime': b['nback']['d_prime'],
        'nback_hit_rate': b['nback']['hit_rate'], 'nback_fa_rate': b['nback']['false_alarm_rate'],
        'ts_accuracy': b['task_switching']['accuracy'], 'ts_switch_cost_ms': b['task_switching']['switch_cost'],
        'ts_repeat_rt_ms': b['task_switching']['repeat_rt'], 'ts_switch_rt_ms': b['task_switching']['switch_rt'],
        'mr_accuracy': b['mental_rotation']['accuracy'], 'mr_mean_rt_ms': b['mental_rotation']['mean_rt'],
        'mr_slope_ms_per_deg': b['mental_rotation']['slope_ms_per_deg'],
        'gng_go_accuracy': b['go_nogo']['go_accuracy'], 'gng_commission_rate': b['go_nogo']['commission_rate'],
        'gng_omission_errors': b['go_nogo']['omission_errors'], 'gng_balanced_accuracy': b['go_nogo']['balanced_accuracy'],
        # Eye fatigue
        'eye_fatigue_score': ss.get('eye_fatigue_score', ''), 'eye_fatigue_level': ss.get('eye_fatigue_level', ''),
        'eye_source': ss.get('eye_fatigue_source', ''),
        # Criterion measure & usability
        'igds_total': igds_sum, 'igds_criteria_very_often': igds_criteria,
        'sus_score': sus_score(sus),
        'comment': ss.get('study_comment', '').replace('\n', ' ').strip(),
    })
    for i, p in enumerate(a['fused_proba']):
        row[f'sys_fused_p_{["low", "medium", "high"][i]}'] = round(float(p), 4)
    for key, name in (('rf', 'behaviour'), ('cog', 'cog'), ('eye', 'eye')):
        for i, p in enumerate(a[f'{key}_proba']):
            row[f'sys_{name}_p_{["low", "medium", "high"][i]}'] = round(float(p), 4)
    row.update({f'igds_q{i+1}': v for i, v in enumerate(igds)})
    row.update({f'sus_q{i+1}': v for i, v in enumerate(sus)})
    return row


def _likert_form(form_key, items, anchors):
    answers = []
    for i, text in enumerate(items):
        choice = st.radio(f"**{i+1}.** {text}", anchors, index=None, horizontal=True, key=f"{form_key}_{i}")
        answers.append(None if choice is None else anchors.index(choice) + 1)
    return answers


def _step(done, label):
    st.markdown(f"{'✅' if done else '⬜'} {label}")


def render_study_page(reset_participant_keys):
    ss = st.session_state
    st.title("📝 Validation Study Mode")
    st.markdown("Runs the full-system test with real students. Each session compares the system's risk level with the "
                "validated **IGDS9-SF** questionnaire and records the **SUS** usability score.")

    n_saved = 0
    if os.path.exists(STUDY_CSV):
        with open(STUDY_CSV, newline='', encoding='utf-8') as f:
            n_saved = sum(1 for _ in csv.DictReader(f))
    st.caption(f"Sessions saved so far: **{n_saved}** · File: `data/study/study_responses.csv`")

    if ss.get('study_saved_pid'):
        st.success(f"✅ Session **{ss['study_saved_pid']}** saved. Thank the participant before resetting the app.")
        if st.button("➡️ Start next participant", type="primary"):
            reset_participant_keys()
            st.rerun()
        return

    # ── Step 0: information sheet & consent ───────────────────────────────
    if not ss.get('study_consent'):
        with st.container(border=True):
            st.subheader("Participant information & consent")
            st.markdown(INFO_SHEET)
            agree = st.checkbox("I have read the information above, I am 18 or older, and I agree to take part.")
            if st.button("Begin session", type="primary", disabled=not agree):
                ss['study_consent'] = True
                ss['study_pid'] = _next_participant_id()
                st.rerun()
        return

    a = ss.get('last_assessment')
    igds_done = ss.get('study_igds') is not None
    battery_done = 'cognitive_results' in ss
    eye_done = 'eye_fatigue_score' in ss
    assess_done = a is not None and a.get('used_measured_cognition')
    sus_done = ss.get('study_sus') is not None

    st.subheader(f"Participant {ss['study_pid']}")
    _step(igds_done, "**Step 1** — IGDS9-SF questionnaire (below, before any results are shown)")
    _step(battery_done, "**Step 2** — Complete all four tests on **🧠 Cognitive & Biometric Tests**")
    _step(eye_done, "**Step 3** *(optional)* — Eye check on **👁️ Eye Fatigue Analysis** (single scan or a session log)")
    _step(assess_done, "**Step 4** — Fill in the survey and run the assessment on **🏠 Home & Multimodal Prediction**")
    _step(sus_done, "**Step 5** — Usability questionnaire (below)")
    if a is not None and not a.get('used_measured_cognition'):
        st.warning("The assessment was run before the cognitive tests were finished. Re-run it on the Home page "
                   "so the prediction uses this participant's measured scores.")
    st.divider()

    # ── Step 1: IGDS9-SF ──────────────────────────────────────────────────
    if not igds_done:
        st.subheader("Step 1 · Internet Gaming Disorder Scale – Short-Form (IGDS9-SF)")
        st.info(IGDS_INSTRUCTIONS)
        with st.form("igds_form"):
            answers = _likert_form("igds", IGDS_ITEMS, LIKERT_5)
            if st.form_submit_button("Save questionnaire", type="primary"):
                if None in answers:
                    st.error("Please answer all nine questions.")
                else:
                    ss['study_igds'] = answers
                    st.rerun()
        return
    st.caption("IGDS9-SF recorded. (Scores are kept hidden from the participant.)")

    if not (battery_done and assess_done):
        st.info("Continue with Steps 2–4 using the navigation menu, then come back here for the final questionnaire.")
        return

    # ── Step 5: SUS ───────────────────────────────────────────────────────
    if not sus_done:
        st.subheader("Step 5 · System Usability Scale (SUS)")
        st.markdown("Please rate how much you agree with each statement about the app you just used.")
        with st.form("sus_form"):
            answers = _likert_form("sus", SUS_ITEMS, SUS_ANCHORS)
            comment = st.text_area("Any comments or suggestions? (optional)")
            if st.form_submit_button("Save usability questionnaire", type="primary"):
                if None in answers:
                    st.error("Please rate all ten statements.")
                else:
                    ss['study_sus'] = answers
                    ss['study_comment'] = comment
                    st.rerun()
        return

    # ── Save ──────────────────────────────────────────────────────────────
    st.subheader("Save session")
    st.markdown("All steps are complete. Saving appends one anonymous row to the study file.")
    if st.button("💾 Save participant session", type="primary"):
        _append_row(build_row(ss))
        ss['study_saved_pid'] = ss['study_pid']
        st.rerun()


def render_export():
    """Researcher-only download of the collected responses."""
    if os.path.exists(STUDY_CSV):
        with open(STUDY_CSV, 'rb') as f:
            st.download_button("⬇️ Download study_responses.csv", f.read(), file_name="study_responses.csv",
                               mime="text/csv")
