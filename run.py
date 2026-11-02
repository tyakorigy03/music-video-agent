#!/usr/bin/env python3
"""Music-video agent.

    python run.py                       # take the next unprocessed song in songs/ and do everything
    python run.py --song "songs/Dame Más - Karol G.mp3"
    python run.py --check-pending       # print has_work=true|false (used by the workflow)
    python run.py --local-videos DIR    # skip YouTube, use the videos in DIR (testing)
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from mvagent import config, state as st
from mvagent.names import parse_song


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gh_output(key, value):
    f = os.getenv("GITHUB_OUTPUT")
    if f:
        with open(f, "a") as fh:
            fh.write(f"{key}={value}\n")


def gh_summary(text):
    f = os.getenv("GITHUB_STEP_SUMMARY")
    if f:
        with open(f, "a") as fh:
            fh.write(text + "\n")


def choose_videos(song, state, local_dir):
    """Agent step: decide which videos to use and get them onto disk."""
    from mvagent.download import download_video, probe

    if local_dir:
        files = sorted(p for p in Path(local_dir).iterdir() if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"})
        return [{"id": p.stem, "title": p.stem, "path": str(p), "chosen_by": "local"} for p in files[: config.NUM_VIDEOS]]

    from mvagent.selector import select_videos

    ranked = select_videos(song, state)
    log("agent shortlist: " + " | ".join(f"{c['title'][:50]} [{c['chosen_by']}]" for c in ranked[:config.NUM_VIDEOS + 2]))
    got = []
    for c in ranked:                                   # picks first, then backups
        if len(got) >= config.NUM_VIDEOS:
            break
        try:
            log(f"downloading {c['id']}: {c['title'][:60]}")
            c = {**c, "path": str(download_video(c["id"]))}
            c.update(probe(c["path"]))
            got.append(c)
        except Exception as e:
            log(f"  download failed ({type(e).__name__}: {str(e)[:200]}) -> trying next candidate")
    if len(got) < 2:
        raise RuntimeError("fewer than 2 videos could be downloaded (YouTube blocking? see README, YT_COOKIES)")
    return got


def process(song_path, state, args):
    from mvagent import planner, qc, render
    from mvagent.audio import analyze_audio
    from mvagent.video import analyze_video

    song = parse_song(song_path)
    log(f"song: '{song['title']}' by '{song['artist'] or '?'}'")
    out_dir = config.OUTPUT / song["slug"]
    out_dir.mkdir(parents=True, exist_ok=True)

    log("analysing song")
    audio = analyze_audio(song_path)
    log(f"  {audio['dur']:.0f}s, {audio['tempo']:.0f} BPM, {len(audio['beats'])} beats")

    vids = choose_videos(song, state, args.local_videos)
    analyses = []
    for v in vids:
        log(f"analysing frames: {Path(v['path']).name}")
        analyses.append(analyze_video(v["path"]))
        log(f"  {len(analyses[-1]['shots'])} shots, detector={analyses[-1]['detector']}")

    seed = planner.stable_seed(song["file"])
    banned = []
    plan = planner.build_plan(audio, analyses, seed)
    if config.USE_VLM:
        try:
            from mvagent import vlm
            for _ in range(2):
                flagged = vlm.review(plan)
                if not flagged:
                    break
                banned += flagged
                plan = planner.build_plan(audio, analyses, seed, banned=banned)
        except Exception as e:
            log(f"VLM review skipped ({type(e).__name__}: {str(e)[:200]})")

    report = qc.check_plan(plan, audio)
    log(f"plan: {report['slots']} clips, longest {report['longest_clip']}s, issues: {report['issues'] or 'none'}")

    video_out = out_dir / f"{song['slug']}.mp4"
    log("rendering")
    render.render_all(plan, analyses, song_path, video_out, audio["dur"], progress=log)
    qc.contact_sheet(video_out, out_dir / "contact_sheet.jpg", dur=audio["dur"])

    (out_dir / "plan.json").write_text(json.dumps(plan, indent=1, default=str))
    (out_dir / "report.json").write_text(json.dumps({"song": song, "qc": report, "videos": [
        {k: v.get(k) for k in ("id", "title", "channel", "chosen_by", "score")} for v in vids]}, indent=1, ensure_ascii=False))
    for v in vids:
        if v.get("chosen_by") != "local":
            state["video_use"][v["id"]] = state["video_use"].get(v["id"], 0) + 1
    size_mb = video_out.stat().st_size / 1e6
    log(f"done: {video_out} ({size_mb:.0f} MB)")
    gh_summary(f"### {song['title']} - {song['artist']}\n- {report['slots']} clips, {size_mb:.0f} MB\n"
               f"- sources: " + ", ".join(f"[{v['title'][:40]}](https://youtu.be/{v['id']})" for v in vids if v.get('chosen_by') != 'local')
               + (f"\n- issues: {report['issues']}" if report["issues"] else ""))
    return {"output": str(video_out.relative_to(config.ROOT)), "videos": [v["id"] for v in vids]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", help="process this specific song file")
    ap.add_argument("--check-pending", action="store_true")
    ap.add_argument("--local-videos", help="directory of videos to use instead of YouTube")
    ap.add_argument("--force", action="store_true", help="re-do a song even if marked done")
    ap.add_argument("--no-state", action="store_true", help="do not read/write state.json (testing)")
    args = ap.parse_args()

    state = st.load()
    if args.check_pending:
        todo = st.pending(state)
        print(f"{len(todo)} song(s) pending")
        gh_output("has_work", "true" if todo else "false")
        return 0

    if args.song:
        song_path = Path(args.song)
        if not song_path.exists():
            print(f"song not found: {song_path}")
            return 2
    else:
        todo = st.pending(state)
        if not todo:
            log("nothing to do: no unprocessed songs in songs/")
            return 0
        song_path = todo[0]
    if not args.force and state["songs"].get(song_path.name, {}).get("status") == "done" and args.song:
        log(f"{song_path.name} is already done (use --force to redo)")
        return 0

    try:
        res = process(song_path, state, args)
        if not args.no_state:
            st.mark(state, song_path.name, "done", **res)
        rc = 0
    except Exception as e:
        traceback.print_exc()
        if not args.no_state:
            st.mark(state, song_path.name, "failed", error=f"{type(e).__name__}: {str(e)[:300]}")
        gh_summary(f"### FAILED: {song_path.name}\n`{type(e).__name__}: {str(e)[:300]}`")
        rc = 1
    finally:
        if not args.no_state:
            st.save(state)
        try:
            from mvagent.download import prune_downloads
            prune_downloads()
        except Exception:
            pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
