"""Turn analyses into an edit plan (a list of slots, each mapped to a source clip).

Rules, all learned from the manual version of this project:
* Every cut lands on a beat; every clip is at most MAX_CLIP seconds.
* Song energy drives footage motion (quiet parts calm, loud parts busy).
* While the song has vocals, shots where her mouth is visible are heavily penalised
  (we cannot lip-sync a different song): dancing, wide shots and hands win.
* The intro (up to the first big energy jump) is filled with calm, single-shot,
  wide establishing footage, slowed down when the shot is short, with a fade-in from black.
* Sources alternate, no source window is used twice, near-black footage is avoided.
"""
import hashlib
import random

import numpy as np

from . import config

SHOT_MARGIN = 0.3        # trim this much off both ends of a shot (cut detection is ~0.125 s coarse)
MAX_STRETCH = 2.4        # slowest allowed slow-motion factor in the intro


def stable_seed(name: str) -> int:
    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------- audio side
def energy_at(audio, t0, t1):
    en = audio["energy2s"]
    i0 = min(int(t0 // 2), len(en) - 1)
    i1 = max(i0 + 1, int(np.ceil(t1 / 2)))
    return float(np.mean(en[i0:i1]))


def vocal_frac(audio, t0, t1):
    seg = audio["vocal"][int(t0): max(int(t0) + 1, int(np.ceil(t1)))]
    return float(np.mean(seg)) if seg else 0.0


def make_slots(audio, rng):
    beats = np.array(audio["beats"])
    en2 = np.array(audio["energy2s"])
    dur = audio["dur"]
    forced = [0.0]
    for i in range(1, len(en2)):
        d = en2[i] - en2[i - 1]
        if d > 0.2 or d < -0.25:
            forced.append(float(beats[np.argmin(abs(beats - i * 2.0))]))
    forced.append(dur)
    forced = sorted(set(round(f, 3) for f in forced))

    slots = []
    for a, b in zip(forced[:-1], forced[1:]):
        pts = sorted(set([a] + [float(t) for t in beats if a < t < b] + [b]))
        idx = 0
        while idx < len(pts) - 1:
            rem = len(pts) - 1 - idx
            if rem <= 16:
                end = len(pts) - 1
            else:
                ks = [k for k in (8, 12, 16) if rem - k >= 6]     # never leave a tiny remainder
                end = min(idx + rng.choice(ks), len(pts) - 1)
            while pts[end] - pts[idx] > config.MAX_CLIP and end > idx + 1:
                end -= 1
            slots.append((pts[idx], pts[end]))
            idx = end
    slots = [s for s in slots if s[1] - s[0] > 0.5]
    fset = set(forced)

    def is_forced(t):
        return any(abs(t - f) < 1e-3 for f in fset)

    changed = True
    while changed:       # merge slivers into a neighbour, preferring not to erase an energy boundary
        changed = False
        for i, (a, b) in enumerate(slots):
            if b - a >= 3.0:
                continue
            opts = []
            if i > 0 and b - slots[i - 1][0] <= config.MAX_CLIP:
                opts.append((is_forced(a), "prev"))
            if i < len(slots) - 1 and slots[i + 1][1] - a <= config.MAX_CLIP:
                opts.append((is_forced(b), "next"))
            if not opts:
                continue
            opts.sort()
            crosses, side = opts[0]
            if crosses and not (is_forced(a) and is_forced(b)):
                continue                      # would erase a boundary and the sliver is not a whole span
            if side == "prev":
                slots[i - 1] = (slots[i - 1][0], b)
            else:
                slots[i + 1] = (a, slots[i + 1][1])
            del slots[i]
            changed = True
            break
    return slots, forced


def intro_end(audio, forced):
    """End of the calm intro, or 0.0 if the song opens loud."""
    if energy_at(audio, 0, 6) > 0.45:
        return 0.0
    for f in forced:
        if f >= 6.0:
            return f if f <= 16.0 else 12.0
    return 12.0 if audio["dur"] > 40 else 0.0


# ---------------------------------------------------------------- video side
def prep(video):
    v = dict(video)
    v["rows_np"] = np.array(video["rows"], dtype=float).reshape(-1, 4)
    v["faces_np"] = np.array(video["faces"], dtype=float).reshape(-1, 4)
    v["cuts_np"] = np.array(video["cuts"], dtype=float)
    return v


def window_stats(V, start, L):
    r = V["rows_np"]
    rr = r[(r[:, 0] >= start) & (r[:, 0] < start + L)]
    if len(rr) < 3:
        return None
    ok = rr[:, 2] <= 0.5
    f = V["faces_np"]
    ff = f[(f[:, 0] >= start) & (f[:, 0] < start + L)]
    if len(ff):
        risk = float(np.mean(ff[:, 2] * (ff[:, 1] > 0.07) * np.minimum(1, ff[:, 1] / 0.35)))
        facesz = float(ff[:, 1].mean())
    else:
        risk = facesz = 0.0
    return {"motion": float(np.median(rr[ok, 1])) if ok.any() else 0.0,
            "ncut": int((rr[:, 2] > 0.5).sum()),
            "bright": float(rr[:, 3].mean()),
            "risk": risk, "facesz": facesz}


def _overlaps(used, vi, a, b, pad=0.5):
    return any(u[0] == vi and a < u[2] + pad and u[1] - pad < b for u in used)


def _banned(banned, vi, a, b):
    return any(x[0] == vi and a < x[2] and x[1] < b for x in (banned or []))


def _pick_intro(slot, vids, audio, rng, used, banned):
    s, e = slot
    L = e - s
    best = None
    for vi, V in enumerate(vids):
        for sh_s, sh_e in V["shots"]:
            us, ue = sh_s + SHOT_MARGIN, sh_e - SHOT_MARGIN
            ulen = ue - us
            if ulen < 2.0 or us < 6.0 or ue > V["duration"] - 15.0:   # skip intros/outros: logos, end cards
                continue
            if ulen >= L:
                src_len, stretch = L, 1.0
                start = rng.uniform(us, ue - L)
            else:
                src_len, stretch = ulen, L / ulen
                start = us
                if stretch > MAX_STRETCH:
                    continue
            if _overlaps(used, vi, start, start + src_len):
                continue
            st = window_stats(V, start, src_len)
            if st is None or st["bright"] < 25:
                continue
            sc = -6.0 * st["risk"] - 3.0 * st["facesz"]     # wide, mouth-free
            sc -= 0.04 * abs(st["motion"] - 6.0)             # calm but alive
            sc -= 0.7 * (stretch - 1.0)                      # prefer little slow-motion
            sc += rng.uniform(0, 0.2)
            if _banned(banned, vi, start, start + src_len):
                sc -= 100
            if best is None or sc > best[0]:
                best = (sc, vi, start, src_len, stretch, st)
    return best


def build_plan(audio, videos, seed, banned=None):
    """banned: list of (video_idx, start, end) windows the reviewer rejected."""
    rng = random.Random(seed)
    vids = [prep(v) for v in videos]
    slots, forced = make_slots(audio, rng)
    i_end = intro_end(audio, forced)
    dur = audio["dur"]

    # candidate windows for the main slots
    cands = []
    for vi, V in enumerate(vids):
        lo, hi = 6.0, max(8.0, V["duration"] - 22.0)   # stay out of the last ~12 s (end cards)
        for _ in range(900):
            st = rng.uniform(lo, hi)
            c = V["cuts_np"][V["cuts_np"] >= st]
            if len(c) and c[0] - st < 2:
                st = float(c[0]) + 0.15          # start just after a cut, skipping blended frames
            cands.append((vi, st))

    probe = [window_stats(vids[v], s, 8) for v, s in cands]
    allm = np.array([p["motion"] for p in probe if p])
    if len(allm) == 0:
        raise RuntimeError("no usable footage in the downloaded videos")

    plan, used = [], []
    vcount = [0] * len(vids)
    lastv = -1
    for si, (s, e) in enumerate(slots):
        L = e - s
        en = energy_at(audio, s, e)
        vf = vocal_frac(audio, s, e)
        item = None
        if i_end > 0 and s < i_end - 0.05:
            pick = _pick_intro((s, e), vids, audio, rng, used, banned)
            if pick:
                _, vi, start, src_len, stretch, st = pick
                item = dict(kind="intro", video_idx=vi, src_start=start,
                            src_len=src_len if stretch > 1.05 else None,
                            fade_in=1.0 if si == 0 else None,
                            risk=st["risk"], motion_pct=None)
                used.append((vi, start, start + src_len))
                vcount[vi] += 1
                lastv = vi
        if item is None:
            best = None
            for (vi, start) in cands:
                V = vids[vi]
                if start + L > V["duration"] - 1:
                    continue
                if _overlaps(used, vi, start, start + L):
                    continue
                st = window_stats(V, start, L)
                if st is None:
                    continue
                mp = float((allm < st["motion"]).mean())
                score = -abs(mp - en) * 2.0
                if vf >= 0.4:
                    score -= 7.0 * st["risk"] * vf          # singing: hide the mouth
                    if st["risk"] > 0.2:
                        score -= 3.0
                elif en < 0.5:
                    score += 0.8 * st["risk"]               # instrumental bit: faces are welcome
                if st["bright"] < 25:
                    score -= 1.0
                if vi == lastv:
                    score -= 0.6
                score -= 0.12 * vcount[vi]
                if en < 0.5:
                    score -= 0.03 * st["ncut"]
                score += rng.uniform(0, 0.15)
                if _banned(banned, vi, start, start + L):
                    score -= 100
                if best is None or score > best[0]:
                    best = (score, vi, start, mp, st)
            if best is None:
                raise RuntimeError(f"no footage long enough for slot {si} ({L:.1f}s)")
            _, vi, start, mp, st = best
            item = dict(kind="main", video_idx=vi, src_start=start, src_len=None, fade_in=None,
                        risk=st["risk"], motion_pct=mp)
            used.append((vi, start, start + L))
            vcount[vi] += 1
            lastv = vi
        item.update(i=si, song_start=s, song_end=e, len=L, energy=en, vocal=vf,
                    video=vids[item["video_idx"]]["path"])
        plan.append(item)
    return plan
