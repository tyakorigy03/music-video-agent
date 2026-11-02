"""Download the chosen videos (video stream only: the original audio is never used)."""
import json
import subprocess
from pathlib import Path

from . import config


def probe(path) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    j = json.loads(r.stdout or "{}")
    st = (j.get("streams") or [{}])[0]
    return {"width": st.get("width"), "height": st.get("height"),
            "duration": float(j.get("format", {}).get("duration") or 0)}


def download_video(video_id: str, dest=None) -> Path:
    import yt_dlp  # lazy import

    dest = Path(dest or config.DOWNLOADS)
    dest.mkdir(parents=True, exist_ok=True)
    existing = [p for p in dest.glob(f"{video_id}.*") if p.suffix in {".mp4", ".webm", ".mkv"}]
    if existing:
        return existing[0]
    opts = {
        # video only, up to 1080p, prefer H.264 (fast to decode)
        "format": "bv*[height<=1080][vcodec^=avc1]/bv*[height<=1080]/b[height<=1080]",
        "outtmpl": str(dest / "%(id)s.%(ext)s"),
        "quiet": True, "no_warnings": True, "noprogress": True,
        "retries": 3, "fragment_retries": 3, "concurrent_fragment_downloads": 4,
    }
    if config.COOKIES_FILE:
        opts["cookiefile"] = config.COOKIES_FILE
    with yt_dlp.YoutubeDL(opts) as y:
        y.download([f"https://www.youtube.com/watch?v={video_id}"])
    files = [p for p in dest.glob(f"{video_id}.*") if p.suffix in {".mp4", ".webm", ".mkv"}]
    if not files:
        raise RuntimeError(f"yt-dlp produced no file for {video_id}")
    info = probe(files[0])
    if not info["width"] or info["duration"] < 30:
        files[0].unlink(missing_ok=True)
        raise RuntimeError(f"downloaded file for {video_id} is unusable: {info}")
    return files[0]


def prune_downloads(max_gb: float = None):
    """Keep the cache small: delete the oldest downloads beyond the size cap."""
    max_bytes = (max_gb or config.PRUNE_DOWNLOADS_GB) * 1024 ** 3
    files = sorted(config.DOWNLOADS.glob("*.*"), key=lambda p: p.stat().st_mtime) if config.DOWNLOADS.exists() else []
    total = sum(p.stat().st_size for p in files)
    for p in files:
        if total <= max_bytes:
            break
        total -= p.stat().st_size
        p.unlink(missing_ok=True)
