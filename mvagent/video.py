"""Frame analysis of one source video (single decode pass, cached as JSON).

rows : [t, motion, cut, brightness]      sampled ~8x per second
faces: [t, face_height, mouth_visible, mouth_open]   sampled ~every 0.75 s
shots: [[start, end], ...] derived from detected hard cuts
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from . import config


def _make_face_fn():
    """MediaPipe Face Mesh (tracks the mouth). Falls back to a Haar cascade if unavailable."""
    try:
        import mediapipe as mp

        fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=4, refine_landmarks=False,
            min_detection_confidence=0.4)

        def fn(bgr):
            r = fm.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            best = [0.0, 0.0, 0.0]
            for lm in (r.multi_face_landmarks or []):
                P = np.array([[p.x, p.y] for p in lm.landmark])
                fh = float(P[:, 1].max() - P[:, 1].min())
                mouth = P[[13, 14, 61, 291, 0, 17]]
                vis = float(((mouth > 0.01) & (mouth < 0.99)).all())
                op = float(abs(P[14, 1] - P[13, 1]) / max(fh, 1e-6))
                if fh > best[0]:
                    best = [fh, vis, op]
            return best

        return fn, "mediapipe"
    except Exception as e:
        print(f"[video] mediapipe unavailable ({type(e).__name__}); using Haar cascade")
        casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        def fn(bgr):
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            faces = casc.detectMultiScale(g, 1.2, 5, minSize=(30, 30))
            if len(faces) == 0:
                return [0.0, 0.0, 0.0]
            h = max(f[3] for f in faces) / g.shape[0]
            return [float(h), 1.0, 0.0]

        return fn, "haar"


def _cache_file(path):
    p = Path(path)
    key = hashlib.md5(f"{p.name}{p.stat().st_size}".encode()).hexdigest()[:10]
    return config.ANALYSIS / f"video_{p.stem}_{key}.json"


def analyze_video(path) -> dict:
    cp = _cache_file(path)
    if cp.exists():
        return json.loads(cp.read_text())
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(fps / 8)))
    face_fn, detector = _make_face_fn()
    rows, faces = [], []
    prev = prevhist = None
    i = n_sample = 0
    last_t = 0.0
    while cap.grab():
        if i % step == 0:
            ok, fr = cap.retrieve()
            if not ok:
                break
            t = i / fps
            last_t = t
            s = cv2.resize(fr, (320, max(1, int(320 * fr.shape[0] / fr.shape[1]))))
            g = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([s], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
            hist = cv2.normalize(hist, hist).flatten()
            motion = float(np.abs(g.astype(np.float32) - prev.astype(np.float32)).mean()) if prev is not None else 0.0
            cut = float(cv2.compareHist(prevhist, hist, cv2.HISTCMP_BHATTACHARYYA)) if prevhist is not None else 0.0
            rows.append([round(t, 3), round(motion, 3), round(cut, 3), round(float(g.mean()), 2)])
            if n_sample % 6 == 0:
                f640 = cv2.resize(fr, (640, max(1, int(640 * fr.shape[0] / fr.shape[1]))))
                fh, vis, op = face_fn(f640)
                faces.append([round(t, 3), round(fh, 4), vis, round(op, 4)])
            prev, prevhist = g, hist
            n_sample += 1
        i += 1
    cap.release()
    duration = max(last_t, i / fps)
    cuts = [r[0] for r in rows if r[2] > 0.5]
    bounds = [0.0] + cuts + [duration]
    shots = [[bounds[k], bounds[k + 1]] for k in range(len(bounds) - 1) if bounds[k + 1] - bounds[k] > 0.4]
    out = {"path": str(path), "fps": fps, "width": W, "height": H, "duration": duration,
           "detector": detector, "rows": rows, "faces": faces, "cuts": cuts, "shots": shots}
    config.ANALYSIS.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(out))
    return out
