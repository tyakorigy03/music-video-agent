"""ffmpeg rendering. Each clip is cached by its parameters, so re-planning only re-renders what changed."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

from . import config


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd[:6])} ...\n{r.stderr[-1500:]}")


def _needs_blur_fit(sw, sh, W, H):
    return not sw or not sh or abs(sw / sh - W / H) > 0.02


def clip_key(p, src_size, W, H, fps):
    blob = json.dumps([p["video"], src_size, round(p["src_start"], 3), p.get("src_len"),
                       round(p["len"], 3), p.get("fade_in"), W, H, fps], sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()[:16]


def render_clip(p, vid_meta, W=None, H=None, fps=None) -> Path:
    W, H, fps = W or config.OUT_W, H or config.OUT_H, fps or config.FPS
    config.CLIPS.mkdir(parents=True, exist_ok=True)
    out = config.CLIPS / f"{clip_key(p, os.path.getsize(p['video']), W, H, fps)}.mp4"
    if out.exists():
        return out
    L = p["len"]
    slow = p.get("src_len") and L / p["src_len"] > 1.05
    read_len = p["src_len"] if slow else L
    pre = ""
    if slow:
        pre = (f"setpts=PTS*{L / p['src_len']:.5f},"
               f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1,")
    fade = f",fade=t=in:st=0:d={p['fade_in']}" if p.get("fade_in") else ""
    if _needs_blur_fit(vid_meta.get("width"), vid_meta.get("height"), W, H):
        fg = (f"[0:v]{pre}split[a][b];"
              f"[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=30:5[bg];"
              f"[b]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
              f"[bg][fg]overlay=(W-w)/2:(H-h)/2,fps={fps},format=yuv420p{fade}[v]")
    else:
        fg = f"[0:v]{pre}scale={W}:{H},fps={fps},format=yuv420p{fade}[v]"
    tmp = out.with_suffix(".part.mp4")
    _run(["ffmpeg", "-y", "-v", "error", "-ss", f"{p['src_start']:.3f}", "-t", f"{read_len:.3f}",
          "-i", p["video"], "-an", "-filter_complex", fg, "-map", "[v]", "-t", f"{L:.3f}",
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(tmp)])
    tmp.rename(out)
    return out


def assemble(clip_paths, audio_path, out_path, dur):
    lst = Path(out_path).with_suffix(".list.txt")
    lst.write_text("\n".join("file '" + str(Path(c).resolve()).replace("'", "'\\''") + "'" for c in clip_paths))
    fade_at = max(0.0, dur - 2.5)
    _run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(audio_path),
          "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
          "-af", f"afade=t=out:st={fade_at:.2f}:d=2.5", "-t", f"{dur:.3f}",
          "-movflags", "+faststart", str(out_path)])
    lst.unlink(missing_ok=True)


def render_all(plan, vid_metas, audio_path, out_path, dur, progress=print):
    clips = []
    for p in plan:
        meta = vid_metas[p["video_idx"]]
        clips.append(render_clip(p, meta))
        progress(f"  rendered clip {p['i'] + 1}/{len(plan)}")
    assemble(clips, audio_path, out_path, dur)
    return out_path
