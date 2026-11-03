"""Download the chosen videos (video stream only: the original audio is never used).

Two downloaders:
* the converter API used by y2mate (default, no YouTube login needed, works on CI runners);
* yt-dlp, kept as a local fallback.
"""
import json
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

from . import config

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
_VIDEO_EXT = {".mp4", ".webm", ".mkv"}


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


def _headers(video_id: str, **extra) -> dict:
    h = {
        "User-Agent": _UA,
        "Accept": "*/*",
        "Referer": f"https://iframe.y2mate-uk.com/wwwindex.php?videoId={video_id}",
        "Origin": "https://iframe.y2mate-uk.com",
    }
    h.update(extra)
    return h


def _request_json(url: str, headers: dict, data: bytes = None, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _api_url(video_id: str):
    """Ask the y2mate converter for a temporary MP4 URL. Retries a few times."""
    base = config.API_BASE_URL.rstrip("/")
    last = None
    for attempt in range(3):
        try:
            key = _request_json(
                f"{base}/v2/sanity/key?id={urllib.parse.quote(video_id)}",
                _headers(video_id), timeout=40,
            )["key"]
            body = urllib.parse.urlencode({
                "link": f"https://youtu.be/{video_id}",
                "format": "mp4",
                "audioBitrate": config.API_AUDIO_BITRATE,
                "videoQuality": config.API_VIDEO_QUALITY,
                "filenameStyle": "pretty",
                "vCodec": "h264",
            }).encode()
            obj = _request_json(
                f"{base}/v2/converter",
                _headers(video_id, key=key,
                         **{"Content-Type": "application/x-www-form-urlencoded"}),
                data=body, timeout=180,
            )
            if obj.get("url"):
                return obj["url"], obj.get("filename") or ""
            last = RuntimeError(f"converter returned no url: {str(obj)[:200]}")
        except Exception as e:
            last = e
        if attempt < 2:
            time.sleep(2 + 2 * attempt)
    raise RuntimeError(f"converter API failed ({type(last).__name__}: {last})")


def _download_api(video_id: str, dest: Path) -> Path:
    url, _ = _api_url(video_id)
    out = dest / f"{video_id}.mp4"
    tmp = dest / f"{video_id}.part"
    req = urllib.request.Request(url, headers={
        "User-Agent": _UA,
        "Accept": "*/*",
        "Referer": f"https://iframe.y2mate-uk.com/wwwindex.php?videoId={video_id}",
    })
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
            ctype = (r.headers.get("Content-Type") or "").lower()
            if "text/html" in ctype:
                raise RuntimeError("converter API returned a web page instead of a video")
            shutil.copyfileobj(r, fh, length=1 << 20)
        if not tmp.exists() or tmp.stat().st_size < 64 * 1024:
            raise RuntimeError("converter API returned an empty/short file")
        tmp.replace(out)
        info = probe(out)
        if not info["width"] or info["duration"] < 30:
            raise RuntimeError(f"downloaded file is unusable: {info}")
        return out
    except Exception:
        tmp.unlink(missing_ok=True)
        out.unlink(missing_ok=True)
        raise


def _download_ytdlp(video_id: str, dest: Path) -> Path:
    import yt_dlp  # lazy import

    opts = {
        "format": "bv*[height<=1080][vcodec^=avc1]/bv*[height<=1080]/b[height<=1080]",
        "outtmpl": str(dest / "%(id)s.%(ext)s"),
        "quiet": True, "no_warnings": True, "noprogress": True,
        "retries": 3, "fragment_retries": 3, "concurrent_fragment_downloads": 4,
    }
    if config.COOKIES_FILE:
        opts["cookiefile"] = config.COOKIES_FILE
    with yt_dlp.YoutubeDL(opts) as y:
        y.download([f"https://www.youtube.com/watch?v={video_id}"])
    files = [p for p in dest.glob(f"{video_id}.*") if p.suffix in _VIDEO_EXT]
    if not files:
        raise RuntimeError("yt-dlp produced no file")
    info = probe(files[0])
    if not info["width"] or info["duration"] < 30:
        files[0].unlink(missing_ok=True)
        raise RuntimeError(f"downloaded file is unusable: {info}")
    return files[0]


def download_video(video_id: str, dest=None) -> Path:
    dest = Path(dest or config.DOWNLOADS)
    dest.mkdir(parents=True, exist_ok=True)
    existing = [p for p in dest.glob(f"{video_id}.*") if p.suffix in _VIDEO_EXT]
    if existing:
        return existing[0]

    order = {"api": ["api"], "ytdlp": ["ytdlp"]}.get(config.DOWNLOADER, ["api", "ytdlp"])
    errors = []
    for kind in order:
        try:
            return _download_api(video_id, dest) if kind == "api" else _download_ytdlp(video_id, dest)
        except Exception as e:
            errors.append(f"{kind}: {type(e).__name__}: {str(e)[:200]}")
            print(f"[download] {kind} failed for {video_id} ({type(e).__name__}: {str(e)[:150]})")
    raise RuntimeError(f"could not download {video_id} ({'; '.join(errors)})")


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
