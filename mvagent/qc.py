"""Quality control: a contact sheet for a human, and a rule-based report for the log."""
import subprocess


def contact_sheet(video, out_jpg, cols=5, rows=4, dur=None):
    interval = max(1.0, (dur or 120) / (cols * rows))
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vf",
         f"fps=1/{interval:.2f},scale=384:216,tile={cols}x{rows}", "-frames:v", "1", str(out_jpg)],
        check=False)


def check_plan(plan, audio) -> dict:
    dur = audio["dur"]
    issues = []
    if abs(plan[0]["song_start"]) > 0.01:
        issues.append("first slot does not start at 0")
    if abs(plan[-1]["song_end"] - dur) > 0.05:
        issues.append("last slot does not reach the end of the song")
    for a, b in zip(plan[:-1], plan[1:]):
        if abs(a["song_end"] - b["song_start"]) > 0.01:
            issues.append(f"gap or overlap between slots {a['i']} and {b['i']}")
    longest = max(p["len"] for p in plan)
    if longest > 10.0 + 1e-6:
        issues.append(f"clip longer than 10s ({longest:.2f}s)")
    singing_visible = [p["i"] for p in plan if p["vocal"] >= 0.4 and p["risk"] > 0.2]
    if singing_visible:
        issues.append(f"mouth may be visible during vocals in slots {singing_visible}")
    sources = {}
    for p in plan:
        sources[p["video_idx"]] = sources.get(p["video_idx"], 0) + 1
    return {"slots": len(plan), "longest_clip": round(longest, 2),
            "clips_per_source": sources, "issues": issues, "ok": not issues}
