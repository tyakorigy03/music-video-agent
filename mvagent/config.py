"""All tunables in one place. Most can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SONGS_DIR = ROOT / "songs"
CACHE = ROOT / ".cache"
DOWNLOADS = CACHE / "downloads"
ANALYSIS = CACHE / "analysis"
CLIPS = CACHE / "clips"
OUTPUT = ROOT / "output"
STATE_FILE = ROOT / "state.json"

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}

NUM_VIDEOS = int(os.getenv("NUM_VIDEOS", "3"))   # source videos used per song
MAX_CLIP = 10.0                                  # hard cap per clip, seconds
OUT_W = int(os.getenv("OUT_W", "1920"))
OUT_H = int(os.getenv("OUT_H", "1080"))
FPS = 24
MAX_ATTEMPTS = 3                                 # give up on a song after this many failures
PRUNE_DOWNLOADS_GB = float(os.getenv("PRUNE_DOWNLOADS_GB", "3"))

# Downloading. "auto" tries the y2mate-like converter API first, then yt-dlp.
# Use "api" to never touch yt-dlp, or "ytdlp" to disable the API.
DOWNLOADER = os.getenv("DOWNLOADER", "auto").lower()
API_BASE_URL = os.getenv("Y2MATE_API_BASE", "https://cnv.cx")
API_VIDEO_QUALITY = os.getenv("API_VIDEO_QUALITY", "1080")
API_AUDIO_BITRATE = os.getenv("API_AUDIO_BITRATE", "128")
INNERTUBE_KEY = os.getenv("YT_INNERTUBE_KEY", "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8")

# Agent brain (optional). Defaults to GitHub Models, authenticated by the workflow's GITHUB_TOKEN.
USE_LLM = os.getenv("USE_LLM", "1") == "1"
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://models.github.ai/inference")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4.1")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GITHUB_TOKEN") or ""

USE_VLM = os.getenv("USE_VLM", "0") == "1"
COOKIES_FILE = os.getenv("YT_COOKIES_FILE", "")
