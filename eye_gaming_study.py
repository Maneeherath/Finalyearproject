# ============================================================
#  EYE–GAMING STUDY (researcher tool)
#  Research question: do eye signs a webcam can measure differ between heavy
#  and light gamers?
#  Per participant (with consent):
#    1. Questionnaire: gaming hours, years gaming, IGDS9-SF (Pontes & Griffiths,
#       2015), and confounders (sleep last night, screen time today, time since
#       last gaming, glasses/contacts).
#    2. Standardised webcam recording while reading a neutral text. The video is
#       analysed in the participant's browser (egs_component/: Google MediaPipe
#       Face Landmarker -> eye aspect ratio, EAR; Soukupova & Cech, 2016); only
#       per-frame numbers reach the app - no video or images are sent or stored.
#       Measures: blink rate (Patel et al., 1991; Tsubota & Nakamori, 1993),
#       PERCLOS (Wierwille et al., 1994), eye openness (median EAR).
#    3. One anonymous row is saved to data/eye_gaming_study/responses.csv (git-ignored).
#       Online, results need the STUDY_PASSWORD secret, and the CSV must be
#       downloaded after sessions (the cloud disk is wiped on restarts).
#  Results: Spearman correlations of each eye measure with weekly gaming hours
#  and IGDS9-SF, heavy vs light gamer comparison (Mann-Whitney U).
#  Requires ethics approval before use with real participants.
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import hmac
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data', 'eye_gaming_study')
DATA_CSV = os.path.join(DATA_DIR, 'responses.csv')
HEAVY_HOURS = 21                         # >= 3 h/day on average counts as a heavy gamer for the group comparison
MIN_N_FOR_STATS = 5
BLINK_MIN_S, BLINK_MAX_S = 0.05, 0.5     # a closure this long is a blink; longer closures count only in PERCLOS
MAX_GAP_S = 0.15                         # face briefly not tracked for up to this long: keep counting the blink
# Eye aspect ratio (EAR) differs between people (eye shape), so a frame counts as "closed" relative to the
# participant's own open-eye level: EAR below 60% of their median EAR in the recording.
CLOSED_RATIO = 0.6
MEASURE_METHOD = "browser MediaPipe Face Landmarker, EAR < 60% of own median"

# Browser recorder: MediaPipe runs on the participant's device; only per-frame numbers come back
_eye_recorder = components.declare_component("eye_recorder", path=os.path.join(BASE, "egs_component"))

LIKERT_5 = ["Never", "Rarely", "Sometimes", "Often", "Very Often"]
# Pontes, H. M., & Griffiths, M. D. (2015). Computers in Human Behavior, 45, 137-143.
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
LAST_GAMING = ["Less than 1 hour ago", "1–3 hours ago", "3–12 hours ago", "More than 12 hours ago", "I don't game"]
EYEWEAR = ["None", "Glasses (wearing now)", "Contact lenses (wearing now)"]

READING_TEXT = (
    "Sri Lanka has grown tea for more than one hundred and fifty years. The first commercial plantations were "
    "started in the central hills, where the cool climate and regular rain suit the tea bush. Today tea is grown "
    "in several regions, from the low country near the coast to estates more than two thousand metres above sea "
    "level. Each region gives its tea a slightly different taste. High-grown teas are usually light and bright, "
    "while low-grown teas are stronger and darker. Most of the work on an estate still depends on skilled people: "
    "the youngest leaves are picked by hand, then withered, rolled, dried and sorted in the factory. The finished "
    "tea is sold at auction in Colombo and exported to many countries around the world."
)

INFO_SHEET = """
**Study:** *Do eye signs measured by a webcam differ between heavy and light gamers?* — KDU BSc Computer Science
final-year research project (P R M K Herath).

**What you will do (about 5 minutes):**
1. Answer questions about your gaming, sleep and screen use, including a 9-question gaming questionnaire.
2. Sit in front of the webcam and read a short text on screen for about one minute.

**Your data:** the webcam video is analysed **inside your own browser** and is **never sent or saved** — only
numbers such as blinks per minute are kept, together with your answers, under an anonymous code (no name, student
number or images). You can stop at any time, and you can ask for your record to be deleted later using your code.
"""

RESET_KEYS = ['egs_consent', 'egs_answers', 'egs_features', 'egs_saved_id', 'egs_done_at']


# ════════════════════════════════════════════════════════════
#  RECORDING AND EYE MEASURES
# ════════════════════════════════════════════════════════════
def compute_features(frames):
    """frames: list of [t_seconds, face_ok (0/1), EAR or None, MediaPipe blink score or None] in time order."""
    if len(frames) < 2:
        return None
    duration = frames[-1][0] - frames[0][0]
    usable = [f for f in frames if f[1] and f[2] is not None]
    usable_share = len(usable) / len(frames)
    baseline = float(np.median([f[2] for f in usable])) if usable else None
    blinks, closed_since, closed_frames, last_seen = 0, None, 0, None
    for t, ok, e, _ in frames:
        if not ok or e is None:
            # A single dropped frame mid-blink is tolerated; a longer loss ends an unfinished closure uncounted
            if last_seen is not None and t - last_seen > MAX_GAP_S:
                closed_since = None
            continue
        last_seen = t
        if e < CLOSED_RATIO * baseline:
            closed_frames += 1
            closed_since = t if closed_since is None else closed_since
        else:
            if closed_since is not None and BLINK_MIN_S <= t - closed_since <= BLINK_MAX_S:
                blinks += 1
            closed_since = None
    usable_minutes = duration * usable_share / 60
    blink_scores = [f[3] for f in usable if f[3] is not None]
    fps = len(frames) / duration if duration > 0 else 0
    return {
        'rec_seconds': round(duration, 1), 'rec_fps': round(fps, 1),
        'face_visible_pct': round(usable_share * 100, 1),
        'blinks': blinks,
        'blink_rate_per_min': round(blinks / usable_minutes, 1) if usable_minutes > 0 else None,
        'perclos_pct': round(closed_frames / len(usable) * 100, 1) if usable else None,
        'median_ear': round(baseline, 3) if baseline else None,
        'mean_blink_score_pct': round(float(np.mean(blink_scores)) * 100, 1) if blink_scores else None,
        'measure_method': MEASURE_METHOD,
        'quality': 'ok' if usable_share >= 0.7 and fps >= 8 else
                   ('low: face not visible enough' if usable_share < 0.7 else 'low: too few frames per second'),
    }


def is_local():
    """True when the app is opened on the same computer (http://localhost), i.e. not the public online version."""
    try:
        host = (st.context.headers.get('Host') or '').split(':')[0]
    except Exception:
        return True
    return host in ('localhost', '127.0.0.1', '')


def researcher_unlocked():
    """Results and records are open locally; online they need the STUDY_PASSWORD secret."""
    if is_local():
        return True
    try:
        secret = st.secrets.get('STUDY_PASSWORD')
    except Exception:
        secret = None
    if not secret:
        st.info("🔒 Results are hidden on the online version. The app owner can set **STUDY_PASSWORD** under "
                "*Manage app → Settings → Secrets* to view them here.")
        return False
    if st.session_state.get('egs_unlocked'):
        return True
    pw = st.text_input("Researcher password", type="password", key="egs_pw")
    if pw and hmac.compare_digest(pw, str(secret)):
        st.session_state['egs_unlocked'] = True
        st.rerun()
    elif pw:
        st.error("Wrong password.")
    return False


# ════════════════════════════════════════════════════════════
#  STORAGE
# ════════════════════════════════════════════════════════════
def load():
    return pd.read_csv(DATA_CSV) if os.path.exists(DATA_CSV) else pd.DataFrame()


def save(row):
    os.makedirs(DATA_DIR, exist_ok=True)
    df = pd.concat([load(), pd.DataFrame([row])], ignore_index=True)
    tmp = DATA_CSV + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, DATA_CSV)


def next_id(df):
    if df.empty or 'participant_id' not in df:
        return "G001"
    return f"G{int(df['participant_id'].str[1:].astype(int).max()) + 1:03d}"


def delete(pid):
    df = load()
    if not df.empty:
        df = df[df['participant_id'] != pid]
        df.to_csv(DATA_CSV, index=False)


def restore(uploaded_df):
    """Merge a previously downloaded responses.csv back in (the online server's disk is wiped on restarts)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df = pd.concat([load(), uploaded_df], ignore_index=True).drop_duplicates('participant_id', keep='first')
    df.sort_values('participant_id').to_csv(DATA_CSV, index=False)
    return len(df)


# ════════════════════════════════════════════════════════════
#  ANALYSIS
# ════════════════════════════════════════════════════════════
EYE_MEASURES = {'blink_rate_per_min': 'Blink rate (per min)', 'perclos_pct': 'PERCLOS (%)',
                'median_ear': 'Eye openness (median EAR)', 'mean_blink_score_pct': 'Eye-closure score (MediaPipe, %)'}
OUTCOMES = {'weekly_gaming_hours': 'Weekly gaming hours', 'igds_total': 'IGDS9-SF score',
            'sleep_last_night': 'Sleep last night (h)'}


def spearman(x, y):
    d = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(d) < 3 or d['x'].nunique() < 2 or d['y'].nunique() < 2:
        return None, None, len(d)
    r, p = stats.spearmanr(d['x'], d['y'])
    return float(r), float(p), len(d)


def correlation_table(df):
    rows = []
    for m, m_label in EYE_MEASURES.items():
        row = {'Eye measure': m_label}
        for o, o_label in OUTCOMES.items():
            r, p, n = spearman(df[m], df[o])
            row[o_label] = '—' if r is None else f"ρ = {r:+.2f} (p = {p:.3f}, n = {n})"
        rows.append(row)
    return pd.DataFrame(rows)


def group_comparison(df, measure='blink_rate_per_min'):
    heavy = df.loc[df['weekly_gaming_hours'] >= HEAVY_HOURS, measure].dropna()
    light = df.loc[df['weekly_gaming_hours'] < HEAVY_HOURS, measure].dropna()
    if len(heavy) < 2 or len(light) < 2:
        return None
    u, p = stats.mannwhitneyu(heavy, light, alternative='two-sided')
    return {'n_heavy': len(heavy), 'n_light': len(light), 'median_heavy': float(heavy.median()),
            'median_light': float(light.median()), 'U': float(u), 'p': float(p)}


def scatter_chart(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (m, label) in zip(axes, [('blink_rate_per_min', 'Blink rate (per min)'), ('perclos_pct', 'PERCLOS (%)')]):
        d = df[['weekly_gaming_hours', m]].dropna()
        ax.scatter(d['weekly_gaming_hours'], d[m], s=40, color='#3498db', edgecolor='k', linewidth=0.4)
        if len(d) >= 3 and d['weekly_gaming_hours'].nunique() > 1:
            k, b = np.polyfit(d['weekly_gaming_hours'], d[m], 1)
            xs = np.linspace(d['weekly_gaming_hours'].min(), d['weekly_gaming_hours'].max(), 20)
            ax.plot(xs, k * xs + b, color='#e74c3c', lw=1.2, ls='--')
        ax.axvline(HEAVY_HOURS, color='grey', lw=0.8, ls=':')
        ax.set_xlabel('Weekly gaming hours (self-report)', fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle('Webcam eye measures vs gaming time', fontsize=10, fontweight='bold')
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════
#  PAGE
# ════════════════════════════════════════════════════════════
def _questionnaire():
    with st.form("egs_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 18, 60, 21)
        gender = c1.selectbox("Gender", ["Male", "Female", "Prefer not to say"])
        weekday = c2.slider("Gaming on a typical weekday (h/day)", 0.0, 12.0, 1.0, step=0.5)
        weekend = c2.slider("Gaming on a typical weekend day (h/day)", 0.0, 14.0, 2.0, step=0.5)
        years = c2.number_input("Years of regular gaming", 0, 30, 5)
        sleep = c3.slider("Sleep last night (h)", 0.0, 12.0, 7.0, step=0.5)
        screen = c3.slider("Screen time today so far (h)", 0.0, 16.0, 3.0, step=0.5)
        last = c3.selectbox("Your last gaming session ended…", LAST_GAMING)
        eyewear = c1.selectbox("Eyewear right now", EYEWEAR)
        st.markdown("**Gaming questionnaire (IGDS9-SF)** — about your gaming in the past 12 months:")
        igds = [st.radio(f"**{i + 1}.** {q}", LIKERT_5, index=None, horizontal=True, key=f"egs_igds_{i}")
                for i, q in enumerate(IGDS_ITEMS)]
        if st.form_submit_button("Save answers", type="primary"):
            if None in igds:
                st.error("Please answer all nine gaming questions.")
            else:
                st.session_state['egs_answers'] = {
                    'age': age, 'gender': gender, 'weekday_hours': weekday, 'weekend_hours': weekend,
                    'weekly_gaming_hours': round(5 * weekday + 2 * weekend, 1), 'years_gaming': years,
                    'sleep_last_night': sleep, 'screen_time_today': screen, 'last_gaming': last, 'eyewear': eyewear,
                    **{f'igds_q{i + 1}': LIKERT_5.index(a) + 1 for i, a in enumerate(igds)},
                    'igds_total': sum(LIKERT_5.index(a) + 1 for a in igds),
                }
                st.rerun()


def _session():
    ss = st.session_state
    if ss.get('egs_saved_id'):
        st.success(f"✅ Saved as **{ss['egs_saved_id']}**. Give this code to the participant in case they want "
                   "their record deleted later.")
        if st.button("➡️ Next participant", type="primary"):
            for k in RESET_KEYS + [k for k in list(ss.keys()) if k.startswith('egs_igds_')]:
                ss.pop(k, None)
            ss['egs_rec_n'] = ss.get('egs_rec_n', 0) + 1
            st.rerun()
        return

    if not ss.get('egs_consent'):
        st.markdown(INFO_SHEET)
        a = st.checkbox("The participant has read the information above.")
        b = st.checkbox("The participant is 18 or older.")
        c = st.checkbox("The participant agrees to take part and to the in-browser (unsaved) webcam analysis.")
        if st.button("Start session", type="primary", disabled=not (a and b and c)):
            ss['egs_consent'] = True
            st.rerun()
        return

    if 'egs_answers' not in ss:
        st.markdown("**Step 1 of 2 — questionnaire**")
        _questionnaire()
        return

    st.markdown("**Step 2 of 2 — eye recording**")
    if 'egs_features' not in ss:
        seconds = st.select_slider("Recording length", options=[30, 60, 90], value=60, format_func=lambda s: f"{s} s")
        st.info("Ask the participant to sit about an arm's length from the webcam in good light, start the camera, "
                "then press **Start recording** and **read the text below at a normal pace** until the recording "
                "ends. They should blink naturally.")
        # A new key per attempt gives a fresh recorder (never the previous participant's recording)
        value = _eye_recorder(seconds=seconds, key=f"egs_rec_{ss.get('egs_rec_n', 0)}_{seconds}", default=None)
        st.markdown(f"> {READING_TEXT}")
        # A new result arrives as a new value; ignore one already used (e.g. after "Redo recording")
        if value and value.get('done_at') != ss.get('egs_done_at'):
            ss['egs_done_at'] = value['done_at']
            feats = compute_features(value['frames'])
            if feats:
                ss['egs_features'] = feats
                st.rerun()
            st.error("No usable frames were recorded. Start the camera again and make sure the face is visible.")
        return

    f = ss['egs_features']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Blink rate", f"{f['blink_rate_per_min']}/min" if f['blink_rate_per_min'] is not None else "—")
    c2.metric("PERCLOS", f"{f['perclos_pct']}%" if f['perclos_pct'] is not None else "—")
    c3.metric("Face visible", f"{f['face_visible_pct']}%")
    c4.metric("Frames per second", f['rec_fps'])
    if f['quality'] != 'ok':
        st.warning(f"Recording quality: {f['quality']}. You can redo the recording.")
    b1, b2 = st.columns(2)
    if b1.button("💾 Save participant", type="primary", use_container_width=True):
        pid = next_id(load())
        save({'participant_id': pid, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
              **ss['egs_answers'], **f})
        ss['egs_saved_id'] = pid
        st.rerun()
    if b2.button("🔄 Redo recording", use_container_width=True):
        ss.pop('egs_features', None)
        ss['egs_rec_n'] = ss.get('egs_rec_n', 0) + 1
        st.rerun()


def _results():
    st.subheader("Results so far")
    if not researcher_unlocked():
        return
    if not is_local():
        st.warning("⚠️ Online, saved records are **wiped when the app restarts or is updated**. Download the CSV "
                   "after every session, and restore it below if the list is empty.")
    with st.expander("♻️ Restore records from a downloaded CSV"):
        up = st.file_uploader("responses CSV", type=["csv"], key="egs_restore")
        if up is not None and st.button("Restore / merge"):
            st.success(f"Restored — {restore(pd.read_csv(up))} records in total.")
    df = load()
    if df.empty:
        st.info("No participants yet.")
        return
    ok = df[df['quality'] == 'ok']
    c1, c2, c3 = st.columns(3)
    c1.metric("Participants", len(df))
    c2.metric("Good-quality recordings", len(ok))
    c3.metric(f"Heavy gamers (≥ {HEAVY_HOURS} h/week)", int((ok['weekly_gaming_hours'] >= HEAVY_HOURS).sum()))
    if len(ok) < MIN_N_FOR_STATS:
        st.info(f"Statistics appear once at least {MIN_N_FOR_STATS} good-quality recordings exist (now {len(ok)}).")
    else:
        st.markdown("**Spearman correlations** (good-quality recordings only):")
        st.dataframe(correlation_table(ok), use_container_width=True, hide_index=True)
        g = group_comparison(ok)
        if g:
            st.markdown(f"**Blink rate, heavy vs light gamers:** median {g['median_heavy']:.1f} vs "
                        f"{g['median_light']:.1f} per minute (n = {g['n_heavy']} vs {g['n_light']}; Mann-Whitney "
                        f"U = {g['U']:.0f}, p = {g['p']:.3f}).")
        fig = scatter_chart(ok)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Read with care: with small samples, several tests and possible confounders (sleep, time since "
                   "last gaming, glasses), a single significant result may be chance. Report the sleep column too — "
                   "if eye measures follow sleep more than gaming, sleep is the better explanation. Aim for 30+ people.")
    with st.expander("Records, download and withdrawal"):
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download responses.csv", df.to_csv(index=False).encode('utf-8'),
                           file_name="eye_gaming_study_responses.csv", mime="text/csv")
        pid = st.selectbox("Delete a participant's record (withdrawal)", [""] + df['participant_id'].tolist())
        if pid and st.button(f"Delete {pid}"):
            delete(pid)
            st.rerun()


def render_page():
    st.title("👁️ Eye–Gaming Study")
    st.caption("Researcher tool: tests whether webcam-measurable eye signs differ between heavy and light gamers. "
               "Get ethics approval before using it with real participants. Data: `data/eye_gaming_study/` "
               "(numbers only, git-ignored). The eye measurement runs in the browser (Google MediaPipe Face "
               "Landmarker), so it works both locally and on the online version.")
    with st.container(border=True):
        _session()
    st.divider()
    _results()
