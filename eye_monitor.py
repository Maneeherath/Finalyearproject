# ============================================================
#  CONTINUOUS EYE-FATIGUE MONITOR (runs in the background while gaming)
#  Reads the webcam several times a second, measures:
#    - PERCLOS: share of frames with eyes closed (Wierwille et al., 1994;
#      Dinges & Grace, 1998) - approximated with OpenCV Haar cascades
#    - Blink rate per minute: screen use suppresses blinking, a known
#      driver of digital eye strain (Patel et al., 1991; Tsubota & Nakamori, 1993)
#    - ResNet50 fatigue probability on the eye crop (same model and
#      preprocessing as the app's single-scan check), sampled every few seconds
#  and writes one row per time window to data/eye_sessions/session_<time>.csv.
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
MIN_FACE_SHARE = 0.3              # windows where the face is visible less than this are flagged low quality

FIELDS = ['window_start', 'window_end', 'elapsed_min', 'frames', 'face_visible_pct', 'perclos_pct',
          'blinks_per_min', 'resnet_fatigue', 'resnet_samples', 'fatigue_index', 'quality']


def fatigue_index(perclos, blink_rate, resnet_fatigue):
    """Combine the window's signals into a 0-100 fatigue index.
    ResNet50 keeps the largest weight (as in the app's 70/30 single-scan blend); the two
    time-based signals that a single photo cannot capture share the rest equally."""
    perclos_score = min(perclos / PERCLOS_ALERT_LIMIT, 1.0) * 100
    blink_score = float(np.clip((NORMAL_BLINK_RATE - blink_rate) / 10.0, 0, 1)) * 100
    if resnet_fatigue is None:
        return round(0.5 * perclos_score + 0.5 * blink_score, 1)
    return round(0.6 * resnet_fatigue + 0.2 * perclos_score + 0.2 * blink_score, 1)


def load_resnet():
    from tensorflow.keras.models import load_model
    path = os.path.join(BASE, 'models', 'resnet_eye_fatigue_model.h5')
    return load_model(path, compile=False) if os.path.exists(path) else None


def resnet_fatigue_score(model, eye_rgb):
    # Same preprocessing as the app: 64x64 RGB crop scaled to [0, 1]; model output 1.0 = Alert
    x = np.expand_dims(cv2.resize(eye_rgb, (64, 64)).astype('float32') / 255.0, axis=0)
    p_alert = float(model.predict(x, verbose=0)[0][0])
    return (1.0 - p_alert) * 100


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
    ap.add_argument('--resnet-every', type=float, default=5, help="seconds between ResNet50 samples")
    ap.add_argument('--no-resnet', action='store_true', help="skip the ResNet50 model (lower CPU use)")
    ap.add_argument('--preview', action='store_true', help="show a small preview window")
    args = ap.parse_args()

    os.makedirs(SESSION_DIR, exist_ok=True)
    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)
    # Report immediately: loading TensorFlow/ResNet50 below takes ~20 s
    write_status(running=True, starting=True, pid=os.getpid(), started=time.time(), last=None)

    if args.source:
        cap = cv2.VideoCapture(args.source)
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
    if not cap.isOpened():
        write_status(running=False, error="Could not open the webcam / video source.")
        raise SystemExit("Could not open the webcam / video source.")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    model = None if args.no_resnet else load_resnet()

    started = time.time()
    session_file = os.path.join(SESSION_DIR, time.strftime('session_%Y%m%d_%H%M%S.csv', time.localtime(started)))
    with open(session_file, 'w', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def new_window(t):
        return {'start': t, 'frames': 0, 'face': 0, 'closed': 0, 'blinks': 0, 'resnet': []}

    win = new_window(started)
    closed_since = None
    last_resnet = 0.0
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
            gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            win['frames'] += 1

            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            if len(faces):
                win['face'] += 1
                fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
                # Eyes sit in the upper part of the face box
                roi = gray[fy:fy + int(fh * 0.6), fx:fx + fw]
                eyes = eye_cascade.detectMultiScale(roi, 1.1, 6, minSize=(fw // 10, fw // 10))
                eyes_open = len(eyes) > 0

                if not eyes_open:
                    win['closed'] += 1
                    if closed_since is None:
                        closed_since = t0
                else:
                    if closed_since is not None and BLINK_MIN_S <= t0 - closed_since <= BLINK_MAX_S:
                        win['blinks'] += 1
                    closed_since = None
                    if model is not None and t0 - last_resnet >= args.resnet_every:
                        ex, ey, ew, eh = max(eyes, key=lambda r: r[2] * r[3])
                        eye_bgr = frame[fy + ey:fy + ey + eh, fx + ex:fx + ex + ew]
                        win['resnet'].append(resnet_fatigue_score(model, cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2RGB)))
                        last_resnet = t0

                if args.preview:
                    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 200, 0), 2)
                    cv2.putText(frame, "eyes open" if eyes_open else "eyes closed", (fx, fy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0) if eyes_open else (0, 0, 255), 2)
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
    resnet = float(np.mean(win['resnet'])) if win['resnet'] else None
    row = {
        'window_start': time.strftime('%H:%M:%S', time.localtime(win['start'])),
        'window_end': time.strftime('%H:%M:%S', time.localtime(now)),
        'elapsed_min': round((now - started) / 60, 2),
        'frames': win['frames'],
        'face_visible_pct': round(face_share * 100, 1),
        'perclos_pct': round(perclos * 100, 1),
        'blinks_per_min': round(blink_rate, 1),
        'resnet_fatigue': '' if resnet is None else round(resnet, 1),
        'resnet_samples': len(win['resnet']),
        'fatigue_index': fatigue_index(perclos, blink_rate, resnet) if face_share >= MIN_FACE_SHARE else '',
        'quality': 'ok' if face_share >= MIN_FACE_SHARE else 'low (face not visible)',
    }
    with open(session_file, 'a', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
    return row


if __name__ == '__main__':
    main()
