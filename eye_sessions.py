# ============================================================
#  CONTINUOUS EYE-SESSION DASHBOARD (app side of eye_monitor.py)
#  Starts/stops the background monitor, shows live status, and plots
#  the fatigue timeline of a logged gaming session.
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import glob
import json
import os
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from eye_monitor import SESSION_DIR, STATUS_FILE, STOP_FILE

BASE = os.path.dirname(os.path.abspath(__file__))
MONITOR_SCRIPT = os.path.join(BASE, 'eye_monitor.py')
STALE_AFTER_S = 15     # status older than this means the monitor is no longer running
STARTUP_GRACE_S = 120  # allowance for loading TensorFlow + ResNet50 on a slow laptop


def read_status():
    try:
        with open(STATUS_FILE, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def is_running(status):
    # The monitor rewrites its status every 5 s; a stale file means it was closed or crashed.
    # While it is still loading the ResNet50 model there is no heartbeat yet, so allow longer.
    limit = STARTUP_GRACE_S if status and status.get('starting') else STALE_AFTER_S
    return bool(status and status.get('running') and time.time() - status.get('updated', 0) < limit)


def start_monitor(duration_min, window_s, preview):
    os.makedirs(SESSION_DIR, exist_ok=True)
    cmd = [sys.executable, MONITOR_SCRIPT, '--duration', str(duration_min), '--window', str(window_s)]
    if preview:
        cmd.append('--preview')
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' and not preview else 0
    subprocess.Popen(cmd, cwd=BASE, creationflags=flags,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_monitor():
    os.makedirs(SESSION_DIR, exist_ok=True)
    open(STOP_FILE, 'w').close()


def list_sessions():
    return sorted(glob.glob(os.path.join(SESSION_DIR, 'session_*.csv')), reverse=True)


def session_label(path):
    stamp = os.path.basename(path)[len('session_'):-len('.csv')]
    try:
        return time.strftime('%d %b %Y, %H:%M', time.strptime(stamp, '%Y%m%d_%H%M%S'))
    except ValueError:
        return stamp


def level_for(score):
    return ("Alert" if score < 25 else "Mildly Fatigued" if score < 50
            else "Fatigued" if score < 75 else "Severely Fatigued")


def plot_session(df):
    fig, ax1 = plt.subplots(figsize=(8, 3.4))
    ax1.plot(df['elapsed_min'], df['fatigue_index'], color='#e74c3c', marker='o', lw=2, label='Fatigue index')
    if df['resnet_fatigue'].notna().any():
        ax1.plot(df['elapsed_min'], df['resnet_fatigue'], color='#8e44ad', lw=1, ls='--', label='ResNet50 fatigue')
    ax1.plot(df['elapsed_min'], df['perclos_pct'], color='#f39c12', lw=1, ls=':', label='PERCLOS (%)')
    ax1.set_ylim(0, 100)
    ax1.set_xlabel('Minutes into session', fontsize=8)
    ax1.set_ylabel('Score / %', fontsize=8)
    ax2 = ax1.twinx()
    ax2.bar(df['elapsed_min'], df['blinks_per_min'], width=0.25, color='#3498db', alpha=0.3, label='Blinks / min')
    ax2.set_ylabel('Blinks per minute', fontsize=8)
    lines = ax1.get_legend_handles_labels()
    bars = ax2.get_legend_handles_labels()
    ax1.legend(lines[0] + bars[0], lines[1] + bars[1], fontsize=7, loc='upper left')
    ax1.grid(alpha=0.3)
    ax1.set_title('Eye-fatigue timeline during the gaming session', fontsize=9, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_monitor_tab():
    st.markdown("Run the monitor **while you game**: it reads the webcam in the background (the game can be "
                "full-screen) and logs blink rate, eye closure (PERCLOS) and the ResNet50 fatigue score every "
                "window. Frames are analysed in memory only — **no images are saved**.")

    status = read_status()
    running = is_running(status)

    with st.container(border=True):
        if running:
            started = time.strftime('%H:%M:%S', time.localtime(status['started']))
            if status.get('starting'):
                st.info(f"⏳ Monitor starting (loading the ResNet50 model, ~20 s) — started {started}.")
            else:
                st.success(f"🟢 Monitor running since {started}. Start your game — come back here to stop it.")
            last = status.get('last')
            if last:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Fatigue index", last['fatigue_index'] if last['fatigue_index'] != '' else "—")
                c2.metric("Blinks / min", last['blinks_per_min'])
                c3.metric("PERCLOS", f"{last['perclos_pct']}%")
                c4.metric("Face visible", f"{last['face_visible_pct']}%")
            else:
                st.caption("Waiting for the first window to finish…")
            b1, b2 = st.columns(2)
            if b1.button("🔄 Refresh status", use_container_width=True):
                st.rerun()
            if b2.button("⏹️ Stop monitor", type="primary", use_container_width=True):
                stop_monitor()
                time.sleep(1.5)
                st.rerun()
        else:
            if status and status.get('error'):
                st.error(status['error'])
            st.caption("Close other apps that use the webcam (including the Single Scan tab) before starting.")
            c1, c2, c3 = st.columns(3)
            duration = c1.number_input("Duration (minutes, 0 = until stopped)", 0, 480, 60, step=15)
            window = c2.select_slider("Log every", options=[15, 30, 60, 120], value=30, format_func=lambda s: f"{s} s")
            preview = c3.checkbox("Show preview window", value=False,
                                  help="Opens a small camera window to check the face is detected.")
            if st.button("▶️ Start background monitor", type="primary"):
                start_monitor(duration, window, preview)
                # Wait for the monitor's first status report so the page shows it as running
                for _ in range(20):
                    time.sleep(0.5)
                    if is_running(read_status()):
                        break
                st.rerun()

    sessions = list_sessions()
    if not sessions:
        st.info("No sessions logged yet.")
        return

    st.subheader("📈 Session results")
    path = st.selectbox("Session", sessions, format_func=session_label)
    df = pd.read_csv(path)
    for col in ('fatigue_index', 'resnet_fatigue'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    good = df[df['quality'] == 'ok'].dropna(subset=['fatigue_index'])
    if good.empty:
        st.warning("This session has no usable windows yet (the face was not visible enough).")
        return

    first, last = good['fatigue_index'].iloc[0], good['fatigue_index'].iloc[-1]
    avg, peak = round(good['fatigue_index'].mean(), 1), round(good['fatigue_index'].max(), 1)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Session length", f"{df['elapsed_min'].max():.1f} min")
    c2.metric("Average fatigue", f"{avg}/100", level_for(avg), delta_color="off")
    c3.metric("Peak fatigue", f"{peak}/100")
    c4.metric("Change start → end", f"{last - first:+.1f}", delta_color="off")
    plot_session(good)
    if len(good) < len(df):
        st.caption(f"{len(df) - len(good)} window(s) excluded because the face was not visible enough.")
    with st.expander("Window data"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    if st.button("✅ Use this session's average fatigue in the prediction"):
        st.session_state['eye_fatigue_score'] = avg
        st.session_state['eye_fatigue_level'] = level_for(avg)
        st.session_state['eye_fatigue_source'] = 'session_monitor'
        st.success(f"Eye fatigue {avg}/100 ({level_for(avg)}) synced to the fusion engine.")
