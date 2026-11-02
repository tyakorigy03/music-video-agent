"""state.json: which songs are done and which source videos were used (to keep variety)."""
import json
from datetime import datetime, timezone

from . import config


def load():
    s = json.loads(config.STATE_FILE.read_text()) if config.STATE_FILE.exists() else {}
    s.setdefault("songs", {})
    s.setdefault("video_use", {})
    return s


def save(state):
    config.STATE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


def list_songs(songs_dir=None):
    d = songs_dir or config.SONGS_DIR
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in config.AUDIO_EXT)


def pending(state, songs_dir=None):
    out = []
    for p in list_songs(songs_dir):
        rec = state["songs"].get(p.name, {})
        if rec.get("status") == "done":
            continue
        if rec.get("attempts", 0) >= config.MAX_ATTEMPTS:
            continue
        out.append(p)
    return out


def mark(state, song_file, status, **extra):
    rec = state["songs"].setdefault(song_file, {})
    if status == "failed":
        rec["attempts"] = rec.get("attempts", 0) + 1
    rec["status"] = status
    rec["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rec.update(extra)
    return rec
