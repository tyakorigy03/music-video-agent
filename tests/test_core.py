import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mvagent import config, planner, qc, selector, state as st
from mvagent.names import parse_song


# ---------------------------------------------------------------- names
def test_parse_basic():
    s = parse_song("songs/Dame Más - Karol G.mp3")
    assert s["title"] == "Dame Más" and s["artist"] == "Karol G"
    assert s["slug"] == "dame-mas-karol-g"


def test_parse_featuring_and_dash_variants():
    s = parse_song("Contigo – Karol G feat. Tiësto.mp3")
    assert s["title"] == "Contigo"
    assert s["artist_variants"] == ["Karol G feat. Tiësto", "Karol G"]


def test_parse_no_artist():
    s = parse_song("Just A Title.mp3")
    assert s["artist"] == "" and s["artist_variants"] == []
    assert selector.build_queries(s)[0].startswith("Just A Title")


# ---------------------------------------------------------------- selector
SONG = parse_song("Dame Más - Karol G.mp3")


def cand(i, title, channel="KarolGVEVO", dur=240, views=10_000_000):
    return {"id": i, "title": title, "channel": channel, "duration": dur, "views": views, "query": "q"}


def test_excludes_song_own_video_and_junk():
    assert selector.score(cand("a", "KAROL G - Dame Más (Official Video)"), SONG, {}) is None
    assert selector.score(cand("b", "Dame Mas Lyrics"), SONG, {}) is None
    assert selector.score(cand("c", "KAROL G - EL BARCO (Official Video)", dur=40), SONG, {}) is None
    assert selector.score(cand("d", "KAROL G - TQG (Lyric Video)"), SONG, {}) is None
    assert selector.score(cand("e", "Karol G reacts to everything"), SONG, {}) is None
    assert selector.score(cand("f", "KAROL G - EL BARCO (Official Video)"), SONG, {}) is not None


def test_ranking_prefers_official_artist_channel_and_variety():
    cands = [
        cand("1", "KAROL G - EL BARCO (Official Video)"),
        cand("2", "KAROL G - EL BARCO (Official Video) [4K]"),          # same stem -> dropped
        cand("3", "KAROL G - Si Antes Te Hubiera Conocido | Coke Studio", views=5_000_000),
        cand("4", "random fan edit of her best moments", channel="someone", views=1000),
        cand("5", "KAROL G, Tiësto - CONTIGO (Official Video)"),
    ]
    r = selector.heuristic_rank(cands, SONG, {})
    ids = [c["id"] for c in r]
    assert "2" not in ids and ids[0] in {"1", "5"} and "4" in ids and ids.index("4") == len(ids) - 1


def test_used_videos_are_demoted():
    cands = [cand("1", "KAROL G - EL BARCO (Official Video)"), cand("5", "KAROL G, Tiësto - CONTIGO (Official Video)")]
    first = selector.heuristic_rank(cands, SONG, {})[0]["id"]
    second = selector.heuristic_rank(cands, SONG, {first: 3})[0]["id"]
    assert first != second


def test_select_videos_with_fake_search_and_llm_fallback():
    def fake_search(q, n=10):
        return [cand("1", "KAROL G - EL BARCO (Official Video)"),
                cand("3", "KAROL G - Si Antes Te Hubiera Conocido | Coke Studio"),
                cand("5", "KAROL G, Tiësto - CONTIGO (Official Video)"),
                cand("9", "KAROL G - Dame Más (Official Video)")]
    out = selector.select_videos(SONG, {"video_use": {}}, 3, search=fake_search, llm=lambda r, s, k: None)
    assert {c["id"] for c in out[:3]} == {"1", "3", "5"} and all(c["chosen_by"] == "rules" for c in out)
    out = selector.select_videos(SONG, {"video_use": {}}, 3, search=fake_search, llm=lambda r, s, k: ["5", "3", "1"])
    assert [c["id"] for c in out[:3]] == ["5", "3", "1"] and out[0]["chosen_by"] == "llm"


def test_select_raises_when_search_is_blocked():
    def blocked(q, n=10):
        raise RuntimeError("Sign in to confirm you're not a bot")
    try:
        selector.select_videos(SONG, {"video_use": {}}, 3, search=blocked, llm=lambda *a: None)
        assert False
    except RuntimeError as e:
        assert "YT_COOKIES" in str(e)


def test_llm_garbage_is_ignored(monkeypatch=None):
    config.LLM_API_KEY = "x"
    config.LLM_BASE_URL = "http://127.0.0.1:9"      # nothing listens here
    ranked = [cand(str(i), f"t{i}") for i in range(5)]
    assert selector.llm_pick(ranked, SONG, 3) is None


# ---------------------------------------------------------------- state
def test_state_pending_and_attempts(tmp_path):
    (tmp_path / "A - X.mp3").write_bytes(b"1")
    (tmp_path / "B - Y.wav").write_bytes(b"1")
    (tmp_path / "notes.txt").write_text("x")
    s = {"songs": {}, "video_use": {}}
    assert [p.name for p in st.pending(s, tmp_path)] == ["A - X.mp3", "B - Y.wav"]
    st.mark(s, "A - X.mp3", "done")
    assert [p.name for p in st.pending(s, tmp_path)] == ["B - Y.wav"]
    for _ in range(config.MAX_ATTEMPTS):
        st.mark(s, "B - Y.wav", "failed")
    assert st.pending(s, tmp_path) == []


# ---------------------------------------------------------------- planner (synthetic data)
def fake_video(seed, dur=200.0, face_rate=0.3, shot_len=6.0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, dur, 0.125)
    cuts = list(np.arange(shot_len, dur, shot_len))
    rows = []
    for x in t:
        cut = 0.9 if any(abs(x - c) < 0.07 for c in cuts) else 0.05
        rows.append([round(float(x), 3), float(rng.uniform(2, 25)), cut, float(rng.uniform(60, 140))])
    faces = [[round(float(x), 3), float(rng.uniform(0.15, 0.5)) if rng.random() < face_rate else 0.0,
              1.0, 0.1] for x in np.arange(0, dur, 0.75)]
    bounds = [0.0] + cuts + [dur]
    return {"path": f"/tmp/fake{seed}.mp4", "duration": dur, "rows": rows, "faces": faces, "cuts": cuts,
            "shots": [[bounds[i], bounds[i + 1]] for i in range(len(bounds) - 1)], "width": 1920, "height": 1080}


def fake_audio(dur=120.0):
    beats = [float(b) for b in np.arange(0.3, dur, 0.6)]
    en = np.concatenate([np.full(6, 0.1), np.full(int(dur / 2) - 6, 0.7)]).tolist()
    en[20] = 0.3
    vocal = [0] * 5 + [1] * (int(dur) - 5)
    return {"dur": dur, "tempo": 100.0, "beats": beats, "energy2s": en, "vocal": vocal}


def test_plan_invariants():
    audio, vids = fake_audio(), [fake_video(i) for i in range(3)]
    plan = planner.build_plan(audio, vids, seed=1)
    rep = qc.check_plan(plan, audio)
    assert rep["longest_clip"] <= 10.0 + 1e-6
    assert not [i for i in rep["issues"] if "gap" in i or "start" in i or "end" in i or "longer" in i], rep
    assert plan[0]["fade_in"] == 1.0 and plan[0]["kind"] == "intro"
    assert len(set(p["video_idx"] for p in plan)) >= 2
    # same seed -> same plan; different seed -> different plan
    assert plan == planner.build_plan(audio, vids, seed=1)
    assert [p["src_start"] for p in plan] != [p["src_start"] for p in planner.build_plan(audio, vids, seed=2)]


def test_no_source_window_is_used_twice():
    audio, vids = fake_audio(), [fake_video(i) for i in range(3)]
    plan = planner.build_plan(audio, vids, seed=3)
    for i, a in enumerate(plan):
        for b in plan[i + 1:]:
            if a["video_idx"] == b["video_idx"]:
                la, lb = (a["src_len"] or a["len"]), (b["src_len"] or b["len"])
                assert a["src_start"] + la <= b["src_start"] or b["src_start"] + lb <= a["src_start"]


def test_vocal_slots_avoid_mouths_when_possible():
    audio, vids = fake_audio(), [fake_video(i, face_rate=0.25) for i in range(3)]
    plan = planner.build_plan(audio, vids, seed=4)
    vocal_risk = [p["risk"] for p in plan if p["vocal"] >= 0.4]
    assert np.mean(vocal_risk) < 0.2


def test_banned_window_is_avoided():
    audio, vids = fake_audio(), [fake_video(i) for i in range(3)]
    plan = planner.build_plan(audio, vids, seed=5)
    p = plan[6]
    span = p["src_len"] or p["len"]
    plan2 = planner.build_plan(audio, vids, seed=5, banned=[(p["video_idx"], p["src_start"], p["src_start"] + span)])
    q = plan2[6]
    assert not (q["video_idx"] == p["video_idx"] and abs(q["src_start"] - p["src_start"]) < span)


def test_energy_boundary_is_kept_as_a_cut():
    """The first big energy jump (the 'drop') must stay a slot boundary, and the intro ends on it."""
    import random
    audio = fake_audio()
    audio["energy2s"] = [0.0, 0.07, 0.14, 0.18, 0.24, 0.2] + [0.6] * (len(audio["energy2s"]) - 6)   # drop at 12 s
    slots, forced = planner.make_slots(audio, random.Random(1))
    drop = [f for f in forced if 10 < f < 14][0]
    assert any(abs(b - drop) < 1e-3 for _, b in slots), (slots, drop)
    assert all(b - a <= 10.0 + 1e-6 for a, b in slots)
    for seed in range(20):                      # robust across random splits
        slots, _ = planner.make_slots(audio, random.Random(seed))
        assert any(abs(b - drop) < 1e-3 for _, b in slots)
