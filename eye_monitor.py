# ============================================================
#  CONTINUOUS EYE-FATIGUE MONITOR (runs in the background while gaming)
#  Reads the webcam several times a second and, for every frame, uses the
#  same eye localisation + eye-state CNN as the app's Eye Check
#  (eye_analysis.py) to decide whether the eyes are open or closed. Per time
#  window it logs:
#    - PERCLOS: share of frames with eyes closed (Wierwille et al., 1994;
#      Dinges & Grace, 1998) - the standard drowsiness measure
#    - Blink rate per minute: screen use suppresses blinking, a known
#      driver of digital eye strain (Patel et al., 1991; Tsubota & Nakamori, 1993)
#    - Mean probability of closed eyes from the CNN
#  and writes one row per window to data/eye_sessions/session_<time>.csv.
#  No images are stored - frames are processed in memory only.
#
#  Usage:  python eye_monitor.py [--duration MIN] [--window SEC] [--preview]
#  Stop:   Ctrl+C, the --duration limit, or the Stop button in the app
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import argparse
import csv
import json
import os
import time

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SESSION_DIR = os.path.join(BASE, 'data', 'eye_sessions')
STATUS_FILE = os.path.join(SESSION_DIR, 'status.json')
STOP_FILE = os.path.join(SESSION_DIR, 'STOP')

# Blinks are short closures; longer closures still count toward PERCLOS
BLINK_MIN_S, BLINK_MAX_S = 0.05, 0.5
PERCLOS_ALERT_LIMIT = 0.15        # PERCLOS at/above this maps to the maximum drowsiness sub-score
NORMAL_BLINK_RATE = 15.0          # blinks/min at rest; strain sub-score grows as the rate drops below this
MIN_FACE_SHARE = 0.3              # windows where the eyes are found less than this are flagged low quality

FIELDS = ['window_start', 'window_end', 'elapsed_min', 'frames', 'face_visible_pct', 'perclos_pct',
          'blinks_per_min', 'mean_closed_pct', 'fatigue_index', 'quality']


def fatigue_index(perclos, blink_rate):
    """0-100 index: PERCLOS (drowsiness) weighted 60%, reduced blinking (screen strain) 40%."""
    perclos_score = min(perclos / PERCLOS_ALERT_LIMIT, 1.0) * 100
    blink_score = float(np.clip((NORMAL_BLINK_RATE - blink_rate) / 10.0, 0, 1)) * 100
    return round(0.6 * perclos_score + 0.4 * blink_score, 1)


def write_status(**status):
    status['updated'] = time.time()
    tmp = STATUS_FILE + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(status, f)
    os.replace(tmp, STATUS_FILE)


def main():
    ap = argparse.ArgumentParser(description="Continuous eye-fatigue monitor for gaming sessions.")
    ap.add_argument('--camera', type=int, default=0, help="webcam index")
    ap.add_argument('--source', help="video file instead of the webcam (for testing)")
    ap.add_argument('--duration', type=float, default=0, help="minutes to run (0 = until stopped)")
    ap.add_argument('--window', type=float, default=30, help="seconds per logged window")
    ap.add_argument('--fps', type=float, default=10, help="frames analysed per second")
    ap.add_argument('--preview', action='store_true', help="show a small preview window")
    args = ap.parse_args()

    os.makedirs(SESSION_DIR, exist_ok=True)
    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)
    # Report immediately: loading TensorFlow and the eye model below takes a while
    write_status(running=True, starting=True, pid=os.getpid(), started=time.time(), last=None)

    if args.source:
        cap = cv2.VideoCapture(args.source)
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
    if not cap.isOpened():
        write_status(running=False, error="Could not open the webcam / video source.")
        raise SystemExit("Could not open the webcam / video source.")

    from eye_analysis import EyeAnalyzer, MIN_CROP
    analyzer = EyeAnalyzer()

    started = time.time()
    session_file = os.path.join(SESSION_DIR, time.strftime('session_%Y%m%d_%H%M%S.csv', time.localtime(started)))
    with open(session_file, 'w', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def new_window(t):
        return {'start': t, 'frames': 0, 'face': 0, 'closed': 0, 'blinks': 0, 'p_closed': []}

    win = new_window(started)
    closed_since = None
    last_status = 0.0
    last_row = None
    write_status(running=True, pid=os.getpid(), session_file=session_file, started=started, last=None)
    print(f"Monitoring... logging to {session_file} (Ctrl+C to stop)")

    try:
        while True:
            t0 = time.time()
            if args.duration and t0 - started >= args.duration * 60:
                break
            if os.path.exists(STOP_FILE):
                break

            ok, frame = cap.read()
            if not ok:
                if args.source:
                    break       # end of test video
                time.sleep(0.2)
                continue

            h, w = frame.shape[:2]
            if w > 640:
                frame = cv2.resize(frame, (640, int(h * 640 / w)))
            win['frames'] += 1

            boxes, method = analyzer.find_eyes(frame)
            big_enough = bool(boxes) and min(x1 - x0 for x0, _, x1, _ in boxes) >= MIN_CROP
            if method == 'face' and big_enough:          # count only frames with a usable face in view
                win['face'] += 1
                p = float(np.mean(analyzer.p_closed(frame, boxes)))
                win['p_closed'].append(p)
                eyes_closed = p >= 0.5
                if eyes_closed:
                    win['closed'] += 1
                    if closed_since is None:
                        closed_since = t0
                else:
                    if closed_since is not None and BLINK_MIN_S <= t0 - closed_since <= BLINK_MAX_S:
                        win['blinks'] += 1
                    closed_since = None
                if args.preview:
                    for x0, y0, x1, y1 in boxes:
                        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255) if eyes_closed else (0, 200, 0), 2)
                    cv2.putText(frame, f"{'closed' if eyes_closed else 'open'} {p:.2f}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if eyes_closed else (0, 200, 0), 2)
            else:
                closed_since = None

            if args.preview:
                cv2.imshow("Eye monitor (press q to stop)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if t0 - win['start'] >= args.window:
                last_row = flush_window(win, t0, started, session_file)
                win = new_window(t0)
            if t0 - last_status >= 5:     # heartbeat so the app can tell the monitor is alive
                write_status(running=True, pid=os.getpid(), session_file=session_file, started=started, last=last_row)
                last_status = t0

            # Keep CPU use low while a game is running
            time.sleep(max(0.0, 1.0 / args.fps - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        if win['frames'] and time.time() - win['start'] >= args.window / 2:
            last_row = flush_window(win, time.time(), started, session_file)
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()
        if os.path.exists(STOP_FILE):
            os.remove(STOP_FILE)
        write_status(running=False, pid=os.getpid(), session_file=session_file, started=started, last=last_row)
        print(f"Stopped. Session saved to {session_file}")


def flush_window(win, now, started, session_file):
    duration_min = max((now - win['start']) / 60, 1e-6)
    face_share = win['face'] / max(win['frames'], 1)
    perclos = win['closed'] / win['face'] if win['face'] else 0.0
    blink_rate = win['blinks'] / duration_min
    row = {
        'window_start': time.strftime('%H:%M:%S', time.localtime(win['start'])),
        'window_end': time.strftime('%H:%M:%S', time.localtime(now)),
        'elapsed_min': round((now - started) / 60, 2),
        'frames': win['frames'],
        'face_visible_pct': round(face_share * 100, 1),
        'perclos_pct': round(perclos * 100, 1),
        'blinks_per_min': round(blink_rate, 1),
        'mean_closed_pct': round(float(np.mean(win['p_closed'])) * 100, 1) if win['p_closed'] else '',
        'fatigue_index': fatigue_index(perclos, blink_rate) if face_share >= MIN_FACE_SHARE else '',
        'quality': 'ok' if face_share >= MIN_FACE_SHARE else 'low (face not visible)',
    }
    with open(session_file, 'a', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
    return row


if __name__ == '__main__':
    main()
