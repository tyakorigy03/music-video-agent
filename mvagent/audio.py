"""Song analysis: tempo, beat grid, energy curve, and a rough per-second vocal estimate."""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import config


def _cache_path(path):
    p = Path(path)
    key = hashlib.md5(f"{p.name}{p.stat().st_size}".encode()).hexdigest()[:10]
    return config.ANALYSIS / f"audio_{key}.json"


def analyze_audio(path) -> dict:
    cp = _cache_path(path)
    if cp.exists():
        return json.loads(cp.read_text())
    import librosa

    y, sr = librosa.load(str(path), sr=22050, mono=True)
    dur = len(y) / sr
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    beats = [float(b) for b in beats]
    if len(beats) < 8:  # fall back to a regular grid
        step = 60.0 / max(tempo, 60.0)
        beats = [float(t) for t in np.arange(0, dur, step)]

    # energy per 2 s window, normalised 0..1
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    en = []
    for i in range(int(np.ceil(dur / 2.0))):
        m = (t >= i * 2.0) & (t < (i + 1) * 2.0)
        en.append(float(rms[m].mean()) if m.any() else 0.0)
    en = np.array(en)
    en = (en - en.min()) / (en.max() - en.min() + 1e-9)

    # rough vocal presence per second: harmonic mid-band share + level (heuristic, not a separator)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    H, _ = librosa.decompose.hpss(S, margin=3.0)
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = (f >= 250) & (f <= 3400)
    mid = H[band].sum(0)
    ratio = mid / (S.sum(0) + 1e-9)
    v, lev = [], []
    for i in range(int(np.ceil(dur))):
        m = (t >= i) & (t < i + 1)
        v.append(float(ratio[m].mean()) if m.any() else 0.0)
        lev.append(float(mid[m].mean()) if m.any() else 0.0)
    v, lev = np.array(v), np.array(lev)
    lev = lev / (lev.max() + 1e-9)
    sc = (v - v.min()) / (v.max() - v.min() + 1e-9)
    sc = 0.6 * sc + 0.4 * lev
    vocal = (sc > np.percentile(sc, 30)).astype(float)
    vocal = (np.convolve(vocal, np.ones(3) / 3, "same") > 0.5).astype(int)

    out = {"dur": float(dur), "tempo": tempo, "beats": beats,
           "energy2s": en.tolist(), "vocal": vocal.tolist()}
    config.ANALYSIS.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(out))
    return out
