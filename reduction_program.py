# ============================================================
#  GAMING REDUCTION PROGRAM (evidence-based self-help)
#  A guided self-help programme built from the components used in
#  cognitive-behavioural interventions for problematic gaming:
#    - Stepped care by risk level, with referral for High risk
#    - Motivational interviewing: readiness ruler + decisional balance
#    - SMART goal + implementation intentions ("if-then" plans)
#    - Daily self-monitoring with progress feedback
#    - Urge surfing for cravings
#    - Cognitive restructuring of maladaptive gaming cognitions
#    - Behavioural activation (replacement activities) + brief abstinence
#  Progress is stored locally per private nickname in data/program/.
#  This is a self-help tool, not therapy or a diagnosis.
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import datetime
import json
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
PROGRAM_DIR = os.path.join(BASE, 'data', 'program')
RESET_HOURS = 84   # abstinence period used by King et al. (2016)

REFERENCES = [
    "Stevens, M. W. R., King, D. L., Dorstyn, D., & Delfabbro, P. H. (2019). Cognitive-behavioral therapy for "
    "Internet gaming disorder: A systematic review and meta-analysis. Clinical Psychology & Psychotherapy, 26(2), 191-203.",
    "Lindenberg, K., Kindt, S., & Szász-Janocha, C. (2022). Effectiveness of cognitive behavioral therapy-based "
    "intervention in preventing gaming disorder and unspecified Internet use disorder in adolescents: A cluster "
    "randomized clinical trial. JAMA Network Open, 5(2), e2148995.",
    "Miller, W. R., & Rollnick, S. (2013). Motivational Interviewing: Helping People Change (3rd ed.). Guilford Press.",
    "Gollwitzer, P. M., & Sheeran, P. (2006). Implementation intentions and goal achievement: A meta-analysis of "
    "effects and processes. Advances in Experimental Social Psychology, 38, 69-119.",
    "Michie, S., et al. (2013). The behavior change technique taxonomy (v1) of 93 hierarchically clustered techniques. "
    "Annals of Behavioral Medicine, 46(1), 81-95.",
    "Bowen, S., & Marlatt, A. (2009). Surfing the urge: Brief mindfulness-based intervention for college student "
    "smokers. Psychology of Addictive Behaviors, 23(4), 666-671.",
    "King, D. L., & Delfabbro, P. H. (2014). The cognitive psychology of Internet gaming disorder. Clinical "
    "Psychology Review, 34(4), 298-308.",
    "King, D. L., Kaptsis, D., Delfabbro, P. H., & Gradisar, M. (2016). Craving for internet games? Withdrawal "
    "symptoms from an 84-h abstinence from Massively Multiplayer Online gaming. Computers in Human Behavior, 62, 488-494.",
]

# ── Stepped care ──────────────────────────────────────────────────────────────
STEPS = {
    "Low Risk": ("🟢 Level 1 — Stay balanced",
                 "Your gaming currently looks balanced. Keep it that way with light self-monitoring: log your play "
                 "for a week and set one if-then plan for your riskiest moment (e.g., late at night)."),
    "Medium Risk": ("🟡 Level 2 — Guided self-help programme",
                    "Work through every tab below over the next 2–4 weeks: build motivation, set a weekly goal with "
                    "if-then plans, log daily, use urge surfing for cravings and challenge your gaming thinking traps. "
                    "These are the core parts of CBT programmes for problem gaming."),
    "High Risk": ("🔴 Level 3 — Self-help plus professional support",
                  "Your results suggest gaming may be causing real harm. Use this programme, **and** please talk to a "
                  "professional — CBT delivered by a trained therapist has the strongest evidence for gaming problems."),
}
HELP_TEXT = ("**Free, confidential support in Sri Lanka:** National Mental Health Helpline **1926** (24/7, toll-free) · "
             "WhatsApp **075 555 1926** (National Institute of Mental Health). You can also contact your university's "
             "student counselling service.")

# ── Motivational interviewing ─────────────────────────────────────────────────
DECISIONAL_BALANCE = [
    ('pros_gaming', "Good things about gaming the way I do now"),
    ('cons_gaming', "Not-so-good things about gaming the way I do now"),
    ('pros_change', "Good things that could happen if I cut down"),
    ('cons_change', "What might be hard about cutting down"),
]

# ── Implementation intentions ─────────────────────────────────────────────────
TRIGGERS = [
    "I feel bored after lectures", "it is 30 minutes before my bedtime", "I feel stressed about an assignment",
    "friends invite me to play late at night", "I finish my planned gaming time", "I feel lonely or low",
    "I sit at my desk to study", "I wake up and reach for my phone",
]
ACTIONS = [
    "stop playing and put the controller/phone in another room", "go for a 15-minute walk",
    "start one 25-minute study block first", "message a friend instead of playing",
    "do 3 minutes of urge surfing", "read or listen to music until I feel sleepy",
    "tell my friends I'm logging off at the agreed time", "check my gaming log before deciding",
]

# ── Maladaptive gaming cognitions (King & Delfabbro, 2014): four categories ──
THINKING_TRAPS = {
    "Overvaluing game rewards": {
        'examples': ["If I stop now I'll lose my progress / my rank.", "My in-game items are worth more than real things."],
        'reframe': "Game rewards are designed to feel urgent. Ask: will this matter in a week? What real-life goal "
                   "did I give up for it today? Progress will still be there tomorrow.",
    },
    "Rigid rules about gaming": {
        'examples': ["I have to finish the level / daily quest before I can stop.", "I must log in every day or I'll fall behind."],
        'reframe': "This is a rule the game taught you, not a fact. Try breaking it once on purpose and notice what "
                   "actually happens — usually nothing bad.",
    },
    "Gaming to feel good about myself": {
        'examples': ["Gaming is the only thing I'm good at.", "I feel worthless when I'm not winning."],
        'reframe': "Your worth is bigger than one skill. List two non-gaming things you did well this week, and plan one "
                   "activity where you can build another skill.",
    },
    "Gaming for social acceptance": {
        'examples': ["My friends will drop me if I don't play with them.", "People only respect me because of my rank."],
        'reframe': "Real friends will understand a limit. Tell them your stop time in advance, and suggest one offline "
                   "meet-up — friendships rarely depend on one match.",
    },
}

# ── Behavioural activation: alternatives matched to the need gaming meets ─────
ALTERNATIVES = {
    "Excitement / challenge": ["Sport or gym session", "Puzzle or strategy board game", "Learn a coding or design skill"],
    "Relaxation / escape": ["Walk outside without your phone", "Music or a podcast", "Short breathing or stretching routine"],
    "Social connection": ["Meet a friend in person", "Join a university club or society", "Cook or eat with others"],
    "Achievement / progress": ["Finish one assignment section", "Track a fitness goal", "Build a small personal project"],
    "Boredom": ["Plan a mentally engaging task (reading, a course, a hobby)", "Volunteer or help someone", "Tidy or reorganise your space"],
}

URGE_STEPS = [
    ("Notice", "Stop and notice the urge to play. Say to yourself: \"I'm having an urge to game.\""),
    ("Locate", "Where do you feel it in your body — hands, chest, stomach? Just observe it, without acting."),
    ("Breathe", "Take five slow breaths. Breathe in for 4 counts, out for 6. Let the urge be there."),
    ("Ride the wave", "Urges rise, peak and fall like a wave. Watch it change in strength without fighting or obeying it."),
    ("Choose", "The urge has passed its peak. Now choose what you will actually do next — pick from your plan."),
]


# ════════════════════════════════════════════════════════════
#  STORAGE
# ════════════════════════════════════════════════════════════
def _profile_path(code):
    return os.path.join(PROGRAM_DIR, f"{code}.json")


def load_profile(code):
    path = _profile_path(code)
    if os.path.exists(path):
        with open(path, encoding='utf-8-sig') as f:    # tolerate a BOM if the file was edited by hand
            return json.load(f)
    return {'code': code, 'created': datetime.date.today().isoformat(), 'motivation': {}, 'goal': {},
            'plans': [], 'log': [], 'urges': [], 'thoughts': [], 'reset': {}}


def save_profile(profile):
    os.makedirs(PROGRAM_DIR, exist_ok=True)
    tmp = _profile_path(profile['code']) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2)
    os.replace(tmp, _profile_path(profile['code']))


# ════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════
def _tab_motivation(p):
    st.markdown("Change starts with your own reasons. These two questions come from **motivational interviewing** "
                "(Miller & Rollnick, 2013).")
    m = p['motivation']
    with st.form("motivation_form"):
        c1, c2 = st.columns(2)
        importance = c1.slider("How **important** is it to you to cut down on gaming? (0–10)", 0, 10, m.get('importance', 5))
        confidence = c2.slider("How **confident** are you that you could cut down? (0–10)", 0, 10, m.get('confidence', 5))
        st.markdown("**Decisional balance** — be honest, there are no wrong answers:")
        cols = st.columns(2)
        answers = {key: cols[i % 2].text_area(label, m.get(key, ''), height=90)
                   for i, (key, label) in enumerate(DECISIONAL_BALANCE)}
        if st.form_submit_button("Save", type="primary"):
            p['motivation'] = dict(importance=importance, confidence=confidence, updated=datetime.date.today().isoformat(),
                                   **answers)
            save_profile(p)
            st.rerun()

    if m:
        imp, conf = m['importance'], m['confidence']
        st.info(f"You rated importance **{imp}** — why not lower, like {max(imp - 3, 0)}? Your answer to that is your "
                f"own reason for change. Write it at the top of your goal.")
        if conf < imp:
            st.caption(f"Confidence ({conf}) is lower than importance ({imp}). The **Goals & If-Then Plans** tab helps "
                       f"most here: specific plans for your hardest moments raise confidence.")


def _tab_goals(p):
    st.markdown("Specific goals plus **if-then plans** for your trigger moments help people act on their intentions — "
                "a medium-to-large effect (d = 0.65) across 94 studies (Gollwitzer & Sheeran, 2006).")
    a = st.session_state.get('last_assessment')
    g = p['goal']
    baseline_default = float(g.get('baseline_hours_week') or (a['total_hours'] if a else 14.0))

    with st.form("goal_form"):
        c1, c2, c3 = st.columns(3)
        baseline = c1.number_input("Current gaming (hours/week)", 0.0, 100.0, baseline_default, step=0.5)
        target = c2.number_input("Target for this week (hours/week)", 0.0, 100.0,
                                 float(g.get('target_hours_week') or round(baseline_default * 0.75 * 2) / 2), step=0.5,
                                 help="Suggestion: about 25% less than now. Smaller steps are easier to keep.")
        cutoff = c3.selectbox("Stop gaming before bed by", ["30 min", "60 min", "90 min"],
                              index=["30 min", "60 min", "90 min"].index(g.get('bed_cutoff', "60 min")),
                              help="Late-night gaming cuts sleep, a key link between gaming and poorer attention.")
        free_days = st.multiselect("Gaming-free days", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                                   g.get('free_days', []))
        reason = st.text_input("My main reason for this goal", g.get('reason', ''))
        if st.form_submit_button("Save goal", type="primary"):
            p['goal'] = dict(baseline_hours_week=baseline, target_hours_week=target, bed_cutoff=cutoff,
                             free_days=free_days, reason=reason, set_on=datetime.date.today().isoformat())
            save_profile(p)
            st.rerun()

    if g:
        daily = g['target_hours_week'] * 60 / max(1, 7 - len(g['free_days']))
        st.success(f"🎯 Goal: **{g['target_hours_week']:g} h/week** (≈ {daily:.0f} min on gaming days), "
                   f"no gaming {g['bed_cutoff']} before bed"
                   + (f", gaming-free on {', '.join(g['free_days'])}" if g['free_days'] else "")
                   + (f". Reason: *{g['reason']}*" if g.get('reason') else "."))

    st.subheader("If-then plans")
    for i, plan in enumerate(p['plans']):
        c1, c2 = st.columns([6, 1])
        c1.markdown(f"**If** {plan['if']}, **then I will** {plan['then']}.")
        if c2.button("Remove", key=f"rm_plan_{i}"):
            p['plans'].pop(i)
            save_profile(p)
            st.rerun()
    if len(p['plans']) < 5:
        with st.form("plan_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            trig = c1.selectbox("If…", TRIGGERS + ["(write my own)"])
            trig_own = c1.text_input("My own trigger", label_visibility="collapsed", placeholder="My own trigger")
            act = c2.selectbox("then I will…", ACTIONS + ["(write my own)"])
            act_own = c2.text_input("My own action", label_visibility="collapsed", placeholder="My own action")
            if st.form_submit_button("Add plan"):
                when = trig_own.strip() if trig == "(write my own)" else trig
                then = act_own.strip() if act == "(write my own)" else act
                if when and then:
                    p['plans'].append({'if': when, 'then': then})
                    save_profile(p)
                    st.rerun()
                st.error("Write your own trigger and action, or pick from the lists.")


def _tab_log(p):
    st.markdown("Logging every day is one of the most effective behaviour-change techniques (**self-monitoring**, "
                "Michie et al., 2013): it shows patterns you don't notice in the moment.")
    today = datetime.date.today()
    with st.form("log_form"):
        c1, c2, c3 = st.columns(3)
        day = c1.date_input("Date", today, max_value=today)
        minutes = c2.number_input("Minutes gamed", 0, 1440, 0, step=15)
        sleep = c3.number_input("Sleep last night (hours)", 0.0, 14.0, 7.0, step=0.5)
        c4, c5, c6 = st.columns(3)
        craving = c4.slider("Strongest craving today (0–10)", 0, 10, 3)
        mood = c5.select_slider("Mood today", ["😞", "🙁", "😐", "🙂", "😄"], "😐")
        late = c6.checkbox("Played after my bedtime cutoff")
        note = st.text_input("What helped or what triggered gaming today? (optional)")
        if st.form_submit_button("Save today's log", type="primary"):
            entry = dict(date=day.isoformat(), minutes=int(minutes), sleep=sleep, craving=craving,
                         mood=["😞", "🙁", "😐", "🙂", "😄"].index(mood) + 1, late=late, note=note)
            p['log'] = [e for e in p['log'] if e['date'] != entry['date']] + [entry]
            p['log'].sort(key=lambda e: e['date'])
            save_profile(p)
            st.rerun()

    if not p['log']:
        st.info("No entries yet — log today to start your progress chart.")
        return

    df = pd.DataFrame(p['log'])
    df['date'] = pd.to_datetime(df['date'])
    last7 = df[df['date'] > pd.Timestamp(today) - pd.Timedelta(days=7)]
    week_hours = last7['minutes'].sum() / 60
    g = p['goal']
    target = g.get('target_hours_week')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last 7 days", f"{week_hours:.1f} h",
              f"{week_hours - target:+.1f} h vs goal" if target is not None else None, delta_color="inverse")
    c2.metric("Avg craving (7 days)", f"{last7['craving'].mean():.1f}/10" if len(last7) else "—")
    c3.metric("Late-night sessions (7 days)", int(last7['late'].sum()))
    c4.metric("Days logged", len(df))
    if target is not None and len(last7):
        if week_hours <= target:
            st.success("✅ You're within your weekly goal. Keep the streak going!")
        else:
            st.warning("Above your goal this week. Look at the notes on high-minute days — what triggered them? "
                       "Add an if-then plan for that moment.")

    daily_goal = None
    if g:
        daily_goal = g['target_hours_week'] * 60 / max(1, 7 - len(g['free_days']))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    recent = df.tail(28)
    ax.bar(recent['date'], recent['minutes'], color=['#e74c3c' if l else '#3498db' for l in recent['late']], width=0.7,
           label='Minutes gamed (red = past bedtime cutoff)')
    if daily_goal:
        ax.axhline(daily_goal, color='#27ae60', linestyle='--', label=f'Daily goal ({daily_goal:.0f} min)')
    ax.set_ylabel('Minutes', fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(recent['date'], recent['craving'], color='#8e44ad', marker='o', lw=1.5, label='Craving (0–10)')
    ax2.set_ylim(0, 10.5)
    ax2.set_ylabel('Craving', fontsize=8)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper left')
    ax.tick_params(axis='x', labelsize=7, rotation=30)
    ax.set_title('Your daily gaming and cravings', fontsize=9, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _tab_urge(p):
    st.markdown("When a craving hits, **ride it out instead of acting on it**. Urges rise, peak and pass, usually within "
                "minutes. In a randomised trial, students taught this \"urge surfing\" technique acted on their urges "
                "less over the following week (Bowen & Marlatt, 2009).")
    ss = st.session_state
    step = ss.get('urge_step')

    # Callbacks update the step before the page redraws, so each click needs only one run
    def start():
        ss['urge_step'], ss['urge_before'] = 0, ss['urge_before_input']

    def advance():
        ss['urge_step'] = ss['urge_step'] + 1

    if step is None:
        st.slider("How strong is the urge right now? (0–10)", 0, 10, 6, key="urge_before_input")
        st.button("🌊 Start urge surfing", type="primary", on_click=start)
    elif step < len(URGE_STEPS):
        title, text = URGE_STEPS[step]
        st.progress((step + 1) / len(URGE_STEPS), text=f"Step {step + 1} of {len(URGE_STEPS)} · {title}")
        st.markdown(f"### {title}\n{text}")
        st.caption("Take your time — move on when you're ready (about 30 seconds per step).")
        st.button("Next ➡️" if step < len(URGE_STEPS) - 1 else "Finish", type="primary", on_click=advance)
    else:
        after = st.slider("How strong is the urge now? (0–10)", 0, 10, max(ss['urge_before'] - 2, 0), key="urge_after_input")
        if st.button("Save", type="primary"):
            p['urges'].append({'time': datetime.datetime.now().isoformat(timespec='minutes'),
                               'before': ss['urge_before'], 'after': after})
            save_profile(p)
            ss.pop('urge_step', None)
            st.rerun()
        if p['plans']:
            st.markdown("**Your if-then plans:**\n" + "\n".join(f"- If {x['if']}, then I will {x['then']}." for x in p['plans']))

    if p['urges']:
        drops = [u['before'] - u['after'] for u in p['urges']]
        st.caption(f"Urges surfed: **{len(p['urges'])}** · average drop in strength: **{sum(drops) / len(drops):.1f}** points")


def _tab_thoughts(p):
    st.markdown("Problem gaming is kept going by four kinds of unhelpful beliefs (King & Delfabbro, 2014). Changing them "
                "is the \"cognitive\" part of CBT, which shows large reductions in gaming-disorder symptoms "
                "(Hedges' g = 0.92; Stevens et al., 2019).")
    for name, info in THINKING_TRAPS.items():
        with st.expander(f"🧠 {name}"):
            st.markdown("**Sounds like:** " + " · ".join(f"*\"{e}\"*" for e in info['examples']))
            st.markdown(f"**Try instead:** {info['reframe']}")
            with st.form(f"thought_{name}", clear_on_submit=True):
                thought = st.text_input("A thought like this I had recently")
                reframe = st.text_input("A more balanced way to see it")
                if st.form_submit_button("Save to my thought record") and thought.strip():
                    p['thoughts'].append({'category': name, 'thought': thought.strip(), 'reframe': reframe.strip(),
                                          'date': datetime.date.today().isoformat()})
                    save_profile(p)
                    st.rerun()
    if p['thoughts']:
        st.subheader("My thought record")
        st.dataframe(pd.DataFrame(p['thoughts'])[['date', 'category', 'thought', 'reframe']],
                     use_container_width=True, hide_index=True)


def _tab_replace(p):
    st.markdown("Cutting down leaves a gap. Plan what fills it, matched to what gaming gives you (**behavioural activation**).")
    need = st.selectbox("What does gaming mostly give you?", list(ALTERNATIVES))
    st.markdown("\n".join(f"- {a}" for a in ALTERNATIVES[need]))

    st.subheader(f"⏸️ {RESET_HOURS}-hour reset challenge")
    st.markdown(f"A voluntary {RESET_HOURS}-hour (3½-day) break from gaming. In a study of regular online gamers, the "
                "main difficulty during this break was **boredom and wanting mental stimulation**, not severe withdrawal "
                "(King et al., 2016) — so plan engaging alternatives from the list above before you start.")
    r = p['reset']
    now = datetime.datetime.now()
    if r.get('start') and not r.get('done'):
        start = datetime.datetime.fromisoformat(r['start'])
        hours = (now - start).total_seconds() / 3600
        st.progress(min(hours / RESET_HOURS, 1.0), text=f"{hours:.0f} of {RESET_HOURS} hours")
        c1, c2 = st.columns(2)
        if hours >= RESET_HOURS:
            st.balloons()
            reflection = st.text_area("Reset complete! What did you notice? What will you keep doing?")
            if st.button("Save reflection", type="primary"):
                p['reset'].update(done=True, reflection=reflection)
                save_profile(p)
                st.rerun()
        elif c2.button("End challenge early"):
            p['reset'] = {}
            save_profile(p)
            st.rerun()
    else:
        if r.get('done'):
            st.success(f"✅ Reset completed. Your reflection: *{r.get('reflection') or '—'}*")
        if st.button(f"Start a {RESET_HOURS}-hour reset now"):
            p['reset'] = {'start': now.isoformat(timespec='minutes')}
            save_profile(p)
            st.rerun()


# ════════════════════════════════════════════════════════════
#  PAGE
# ════════════════════════════════════════════════════════════
def render_program_page(embedded=False):
    """embedded=True: shown inside the app's Solution step, which already shows the title and risk banner."""
    ss = st.session_state
    if not embedded:
        st.title("🛡️ Gaming Reduction Program")
    st.markdown("A step-by-step self-help programme built from the parts of **cognitive-behavioural therapy (CBT)** "
                "shown to reduce problem gaming (Stevens et al., 2019; Lindenberg et al., 2022).")
    st.caption("This is a self-help tool, not therapy or a medical diagnosis.")

    if not ss.get('program_code'):
        with st.container(border=True):
            st.markdown("**Start or continue your programme.** Choose a private nickname — your progress is saved on "
                        "the computer running this app under that name only (no real name or student number needed).")
            st.caption("On the online version, anyone who types the same nickname sees the same progress, and saved "
                       "progress may be cleared when the server restarts — pick an unusual nickname.")
            code = st.text_input("Nickname (letters, numbers, - or _)", max_chars=30)
            if st.button("Open my programme", type="primary"):
                if re.fullmatch(r'[A-Za-z0-9_-]{3,30}', code or ''):
                    ss['program_code'] = code.lower()
                    st.rerun()
                st.error("Use 3–30 letters, numbers, '-' or '_'.")
        return

    p = load_profile(ss['program_code'])
    c1, c2 = st.columns([4, 1])
    c1.markdown(f"Programme for **{p['code']}** · started {p['created']}")
    if c2.button("Switch user"):
        ss.pop('program_code', None)
        ss.pop('urge_step', None)
        st.rerun()

    a = ss.get('last_assessment')
    risk = a['fused_pred'] if a else None
    if embedded:
        pass                                  # the Solution step already shows the matched programme step
    elif risk is None:
        st.info("Complete the assessment steps to get a programme step matched to your risk level. Until then, all "
                "parts below are available.")
    else:
        title, text = STEPS[risk]
        box = st.error if risk == "High Risk" else st.warning if risk == "Medium Risk" else st.success
        box(f"**{title}** (your latest result: {risk})\n\n{text}")
    if (risk == "High Risk" and not embedded) or (p['log'] and pd.DataFrame(p['log'])['craving'].tail(7).mean() >= 8):
        st.info(HELP_TEXT)

    tabs = st.tabs(["🎯 Motivation", "📅 Goals & If-Then Plans", "📓 Daily Log & Progress",
                    "🌊 Urge Surfing", "🧠 Thinking Traps", "🔄 Replace & Reset"])
    for tab, render in zip(tabs, [_tab_motivation, _tab_goals, _tab_log, _tab_urge, _tab_thoughts, _tab_replace]):
        with tab:
            render(p)

    with st.expander("📚 Research basis"):
        st.markdown("\n".join(f"- {r}" for r in REFERENCES))
    if risk != "High Risk":
        st.caption(HELP_TEXT)
