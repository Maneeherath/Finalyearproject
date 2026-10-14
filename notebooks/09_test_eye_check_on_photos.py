# ============================================================
#  REAL-WORLD TEST OF THE EYE CHECK
#  Runs the app's full eye pipeline (eye_analysis.EyeAnalyzer: face/eye
#  localisation + eye-state CNN) on ordinary colour photos sorted into
#  <folder>/open and <folder>/closed, and reports how often it is right.
#  The MRL test set is infrared eye crops; this checks the photos the app
#  will actually see.
#  Usage: python notebooks/09_test_eye_check_on_photos.py [folder]
#         (default: demo_run/eye_test, see its CREDITS.md for photo sources)
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import glob
import os
import sys

import cv2
import numpy as np
from PIL import Image

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, BASE)
from eye_analysis import EyeAnalyzer  # noqa: E402

folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'demo_run', 'eye_test')
# Optional 2nd argument: model version tag, e.g. "v1" -> models/eye_state_cnn_v1.keras (for comparisons)
tag = sys.argv[2] if len(sys.argv) > 2 else ''
suffix = f'_{tag}' if tag else ''
analyzer = EyeAnalyzer(os.path.join(BASE, 'models', f'eye_state_cnn{suffix}.keras'),
                       os.path.join(BASE, 'models', f'eye_state_cnn{suffix}.json'))
print(f"Model: eye_state_cnn{suffix} (threshold {analyzer.threshold:.2f}, mirror averaging {analyzer.tta})")
rows, tiles = [], []
for truth in ('open', 'closed'):
    row_tiles = []
    for f in sorted(glob.glob(os.path.join(folder, truth, '*.*'))):
        res = analyzer.analyze(np.array(Image.open(f).convert('RGB')))
        score = res['fatigue_score']
        pred = None if score is None else ('closed' if score >= 50 else 'open')
        rows.append((os.path.basename(f), truth, res['method'], res['eyes_found'], score, pred, res['reason']))
        im = cv2.cvtColor(res['annotated_rgb'], cv2.COLOR_RGB2BGR)
        s = 260 / max(im.shape[:2])
        im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)))
        pad = np.full((290, 270, 3), 255, np.uint8)
        pad[5:5 + im.shape[0], 5:5 + im.shape[1]] = im
        label = f"{truth}: {'no eyes' if score is None else f'score {score:.0f}'}"
        ok = pred == truth
        cv2.putText(pad, label, (8, 283), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0) if ok else (0, 0, 255), 1)
        row_tiles.append(pad)
    if row_tiles:
        tiles.append(np.hstack(row_tiles))

print(f"{'photo':16s} {'truth':7s} {'eyes found via':15s} {'n':>2s} {'score':>6s}  prediction")
for name, truth, method, n, score, pred, reason in rows:
    mark = '✓' if pred == truth else ('–' if pred is None else '✗')
    refused = 'refused: eyes too small' if reason == 'too_small' else 'refused: no eyes found'
    print(f"{name:16s} {truth:7s} {method:15s} {n:2d} {('—' if score is None else f'{score:5.1f}'):>6s}  {pred or refused} {mark}")
found = [r for r in rows if r[5] is not None]
correct = sum(r[1] == r[5] for r in found)
print(f"\nEyes found in {len(found)}/{len(rows)} photos; correct open/closed decision in {correct}/{len(found)} of those "
      f"({correct / max(len(found), 1):.0%}); overall {correct}/{len(rows)}.")
open_scores = [r[4] for r in found if r[1] == 'open']
closed_scores = [r[4] for r in found if r[1] == 'closed']
if open_scores and closed_scores:
    print(f"Mean eye fatigue score: open eyes {np.mean(open_scores):.1f}, closed eyes {np.mean(closed_scores):.1f}")
if tiles:
    width = max(t.shape[1] for t in tiles)
    sheet = np.vstack([np.hstack([t, np.full((t.shape[0], width - t.shape[1], 3), 255, np.uint8)]) for t in tiles])
    out = os.path.join(BASE, 'results', f'eye5_realworld_photos{suffix}.jpg')
    cv2.imwrite(out, sheet)
    print(f"Saved {out}")
