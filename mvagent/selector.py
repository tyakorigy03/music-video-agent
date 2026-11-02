"""The agent's decision step: given 'Song - Artist', decide which YouTube videos to download.

Flow
1. Build several search queries from the artist (the hint in the file name).
2. Search with yt-dlp (no download), collect candidates.
3. Score candidates with transparent rules. The song's OWN video is excluded on purpose:
   we want other footage of the artist, not a re-upload of the original clip.
4. Optionally let an LLM (GitHub Models by default) choose among the best candidates.
   If the LLM is unavailable or answers badly, the rule-based ranking is used.
"""
import json
import math
import re
import urllib.request

from . import config
from .names import norm

BANNED = re.compile(
    r"\b(lyrics?|lyric video|audio|karaoke|reaction|reacts?|cover|tutorial|how to|shorts?|"
    r"compilation|full album|playlist|instrumental|slowed|sped up|nightcore|8d|remix|mashup|"
    r"review|interview|explained|behind the scenes|making of|teaser|trailer|podcast|"
    r"live ?stream|fan ?made|tiktok|whatsapp status|dance (?:tutorial|cover)|parody)\b",
    re.I,
)
OFFICIAL = re.compile(r"\bofficial (?:music )?video\b|\bvideo oficial\b", re.I)
PERFORMANCE = re.compile(r"\b(live|session|performance|concert|en vivo|sessions|unplugged)\b", re.I)
VISUALIZER = re.compile(r"\bvisuali[sz]er\b", re.I)


def build_queries(song: dict) -> list:
    """Search queries derived from the file name."""
    artists = song["artist_variants"] or []
    qs = []
    for a in artists[:2]:
        qs += [f"{a} official video", f"{a} live performance", f"{a} session", f"{a} music video"]
    if not qs:  # file name had no artist: lean on the title
        qs = [f"{song['title']} official video", f"{song['title']} live performance"]
    seen, out = set(), []
    for q in qs:
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:6]


def search_youtube(query: str, n: int = 10) -> list:
    """Flat yt-dlp search: metadata only, nothing is downloaded."""
    import yt_dlp  # imported lazily so tests run without it

    opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist",
            "skip_download": True, "noprogress": True}
    if config.COOKIES_FILE:
        opts["cookiefile"] = config.COOKIES_FILE
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(f"ytsearch{n}:{query}", download=False)
    out = []
    for e in (info or {}).get("entries", []) or []:
        if not e or not e.get("id"):
            continue
        out.append({
            "id": e["id"],
            "title": e.get("title") or "",
            "channel": e.get("channel") or e.get("uploader") or "",
            "duration": e.get("duration"),
            "views": e.get("view_count") or 0,
            "query": query,
        })
    return out


def score(c: dict, song: dict, used: dict):
    """Return a score, or None when the candidate must be excluded."""
    title = c["title"]
    nt = norm(song["title"])
    if len(nt) >= 4 and nt in norm(title):
        return None                       # the song's own video / lyric video / audio
    if BANNED.search(title):
        return None
    d = c.get("duration")
    if d is not None and not (90 <= d <= 600):
        return None
    s = 0.0
    if d is None:
        s -= 1.0
    elif 120 <= d <= 420:
        s += 1.0
    if OFFICIAL.search(title):
        s += 3.0
    elif re.search(r"\bofficial\b", title, re.I):
        s += 1.0
    if PERFORMANCE.search(title):
        s += 1.5
    if VISUALIZER.search(title):
        s -= 2.0                          # visualizers rarely show people
    ch = c.get("channel", "")
    for a in song["artist_variants"]:
        if norm(a) and norm(a) in norm(ch):
            s += 3.0
            break
    if "vevo" in ch.lower():
        s += 1.5
    s += 0.3 * min(math.log10((c.get("views") or 0) + 1), 9)
    s -= 0.8 * used.get(c["id"], 0)       # prefer footage not used for earlier songs
    return s


def _stem(title: str) -> str:
    return norm(re.sub(r"\(.*?\)|\[.*?\]", "", title))[:18]


def heuristic_rank(cands: list, song: dict, used: dict) -> list:
    scored = []
    for c in cands:
        s = score(c, song, used)
        if s is not None:
            scored.append((s, c))
    scored.sort(key=lambda x: -x[0])
    out, stems = [], set()
    for s, c in scored:
        st = _stem(c["title"])
        if st in stems:
            continue
        stems.add(st)
        out.append({**c, "score": round(s, 2)})
    return out


def llm_pick(ranked: list, song: dict, k: int):
    """Ask an LLM to choose k ids from the shortlist. Returns a list of ids or None."""
    if not (config.USE_LLM and config.LLM_API_KEY) or len(ranked) < k:
        return None
    lines = "\n".join(
        f"{c['id']} | {c['title']} | {c['channel']} | {c.get('duration')}s | {c.get('views')} views"
        for c in ranked
    )
    prompt = (
        f"Song: \"{song['title']}\" by {song['artist'] or 'unknown artist'}.\n"
        f"I am editing a music video for this song out of OTHER footage of the artist. "
        f"Choose exactly {k} videos from the list below that give the best varied footage: "
        f"the artist dancing, performing, wide shots and different locations or moods. "
        f"Never choose a video of this same song, lyric videos, audio-only uploads, covers, "
        f"reactions, compilations or interviews. Prefer official channels.\n\n{lines}\n\n"
        f"Reply with JSON only: {{\"ids\": [\"id1\", \"id2\", \"id3\"]}}"
    )
    body = json.dumps({
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a careful assistant for a video editor. Output JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(
        config.LLM_BASE_URL.rstrip("/") + "/chat/completions", data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {config.LLM_API_KEY}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            text = json.loads(r.read())["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", text, re.S)
        ids = json.loads(m.group(0))["ids"]
        valid = {c["id"] for c in ranked}
        ids = [i for i in dict.fromkeys(ids) if i in valid]
        return ids[:k] if len(ids) >= k else None
    except Exception as e:                 # network, quota, bad JSON ... never fatal
        print(f"[selector] LLM unavailable, using rules ({type(e).__name__}: {e})")
        return None


def select_videos(song: dict, state: dict, k: int = None, search=search_youtube, llm=llm_pick) -> list:
    """Return an ordered list: the first k are the picks, the rest are backups for failed downloads."""
    k = k or config.NUM_VIDEOS
    used = state.get("video_use", {})
    cands, seen = [], set()
    for q in build_queries(song):
        try:
            for c in search(q):
                if c["id"] not in seen:
                    seen.add(c["id"])
                    cands.append(c)
        except Exception as e:
            print(f"[selector] search failed for '{q}': {type(e).__name__}: {e}")
    ranked = heuristic_rank(cands, song, used)
    if len(ranked) < 2:
        raise RuntimeError(
            f"Only {len(ranked)} usable videos found for '{song['title']}' by '{song['artist']}'. "
            "If YouTube is blocking the runner, add a YT_COOKIES secret (see README)."
        )
    shortlist = ranked[:12]
    ids = llm(shortlist, song, k)
    if ids:
        by_id = {c["id"]: c for c in ranked}
        picks = [{**by_id[i], "chosen_by": "llm"} for i in ids]
        rest = [{**c, "chosen_by": "rules"} for c in ranked if c["id"] not in ids]
        return picks + rest
    return [{**c, "chosen_by": "rules"} for c in ranked]
