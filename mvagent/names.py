"""Turn a file name like 'Dame Más - Karol G.mp3' into search hints."""
import re
import unicodedata
from pathlib import Path

_DASHES = str.maketrans({"–": "-", "—": "-", "‒": "-", "―": "-"})
_SPLIT_ARTIST = re.compile(
    r"\s+(?:feat\.?|ft\.?|featuring|with|x|&|y|con)\s+|\s*[,+/]\s*", re.I
)


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def norm(s: str) -> str:
    """Lowercase, accent-free, letters and digits only. For fuzzy comparisons."""
    return re.sub(r"[^a-z0-9]+", "", strip_accents(s).lower())


def slugify(s: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9]+", "-", strip_accents(s)).strip("-").lower()
    return out or "song"


def parse_song(path) -> dict:
    """'<Song Name> - <Artist Name>.ext' -> dict. Falls back gracefully."""
    stem = Path(path).stem.strip().translate(_DASHES)
    if " - " in stem:
        title, artist = stem.split(" - ", 1)
    else:
        title, artist = stem, ""
    title, artist = title.strip(), artist.strip()
    variants = []
    if artist:
        variants.append(artist)
        primary = _SPLIT_ARTIST.split(artist)[0].strip()
        if primary and primary not in variants:
            variants.append(primary)
    return {
        "file": Path(path).name,
        "title": title,
        "artist": artist,
        "artist_variants": variants,
        "slug": slugify(f"{title}-{artist}" if artist else title),
    }
