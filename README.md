# Music-video agent

Drop a song into `songs/`. Every 5 hours GitHub Actions picks the next unprocessed song, **the agent
decides which YouTube videos to download**, cuts them to the beat of your song, and uploads the finished
video as a run artifact.

```
songs/Dame Más - Karol G.mp3     <- the only thing you do
```

## Your only job: upload songs, named `Song Name - Artist Name`

* `Dame Más - Karol G.mp3`  
* `Contigo - Karol G feat. Tiësto.mp3` (the first name, `Karol G`, is used for extra searches)  
* Formats: mp3, wav, m4a, flac, ogg, aac.

The artist in the file name is the hint the agent searches with. Without ` - ` the whole name is
treated as a title and the search is much weaker.

## Setup (once)

1. Create a **private** repository and push this folder to it.
2. Settings → Actions → General → **Workflow permissions: Read and write**.
3. Add songs to `songs/` and push. Run it now from Actions → *make-music-video* → *Run workflow*, or wait for the schedule.
4. Optional secret `YT_COOKIES` (see "If YouTube blocks the runner").

## Where to find the video

Actions → open a run → **Artifacts** → `music-video-<n>`. It contains the `.mp4`, `contact_sheet.jpg`
(a quick visual check), `plan.json` (every cut) and `report.json` (which videos the agent picked and why).
Artifacts are kept 30 days (`retention-days` in the workflow). The run summary links the YouTube sources.

## How the agent decides (mvagent/selector.py)

1. Builds queries from the artist: "official video", "live performance", "session", "music video".
2. Searches YouTube with yt-dlp (metadata only).
3. Scores candidates by rule: official channel / VEVO, "official video", live or performance footage,
   sensible length, popularity; and **excludes** lyric videos, audio uploads, covers, reactions, shorts,
   compilations, remixes, interviews, and **the song's own video** (it wants other footage of the artist).
   Videos used for earlier songs are demoted, so successive songs get variety.
4. Hands the best 12 to an LLM which chooses 3 (default: **GitHub Models**, free, uses the built-in
   `GITHUB_TOKEN`, no key needed). If that fails for any reason the rule ranking is used. Backups are
   downloaded if a pick fails.

Change the model with `LLM_MODEL`, point to another OpenAI-compatible service with `LLM_BASE_URL` +
`LLM_API_KEY`, or turn it off with `USE_LLM=0`.

## How the edit is made (mvagent/planner.py)

* Beat grid from the song (librosa); every cut lands on a beat; every clip is **at most 10 s**.
* Song energy picks the footage: quiet parts get calm shots, loud parts get high-motion shots.
* **Lip-sync can't match a different song**, so while the song has vocals, shots where her mouth is visible
  (MediaPipe Face Mesh) are heavily penalised: dancing, wide shots, hands and instruments win.
  Close-ups are used on instrumental moments.
* **Intro**: up to the first big energy jump the video uses calm, single-shot, mouth-free wide footage,
  slowed down (motion-interpolated) if the shot is short, with a fade in from black. The first cut lands on the drop.
* Sources alternate, no source moment is used twice, near-black footage and the first/last seconds of
  each video (logos, end cards) are avoided.

Same song name -> same seed -> same edit. Change the seed by renaming the file.

## Optional: second opinion from a small video-language model

`use_vlm` (workflow input) or `USE_VLM=1` runs **SmolVLM2-256M** on one frame per clip and bans clips that
show a close-up face under vocals or a logo/overlaid text, then re-plans. It installs CPU-only PyTorch, so it
adds a few minutes. **This part could not be tested by the author** (see below).

## Limits you should know about

* **YouTube may block GitHub's servers** ("Sign in to confirm you're not a bot"). Fix: export your YouTube
  cookies with a browser extension in Netscape format (use a throwaway Google account), and save the text as
  the repository secret `YT_COOKIES`. Cookies expire; refresh them if downloads start failing. Failed songs are
  retried up to 3 times, then skipped (see `state.json`).
* **Rights**: the output contains other people's videos and music. Downloading may violate YouTube's terms, and
  publishing the result needs permission from the owners. Keep the repo private and treat results as personal.
* **Free minutes**: public repos get free Actions minutes; private repos get a monthly allowance. One song takes
  roughly 20-40 minutes. If you run out, lower the cron frequency in `make-video.yml`.
* **Artifact storage** (~100 MB per video) is limited on free plans. Lower `retention-days` or download and delete.
* Scheduled workflows in a public repo are paused after 60 days without repository activity.
* Vocal detection is an audio heuristic, not a vocal separator. A few seconds may be misjudged.
* A burned-in logo or caption inside a source video can still slip through (only the optional VLM check looks for it).

## Local run (for testing)

```bash
sudo apt install ffmpeg libgl1
pip install -r requirements.txt
python run.py --check-pending
python run.py                                  # next pending song (needs internet + yt-dlp working)
python run.py --local-videos ./some_videos --no-state    # skip YouTube, use your own clips
python -m pytest tests
```

## What was tested

| Part | Status |
|---|---|
| Name parsing, scoring/exclusion rules, ranking, state, planner invariants (15 unit tests) | passes |
| Full pipeline on real files: song analysis, frame/mouth analysis, planning, rendering, mux, contact sheet | ran end to end |
| YouTube search/download (yt-dlp) and the GitHub Models call | **not run** (the authoring sandbox cannot reach YouTube or GitHub Models); written from the docs |
| GitHub Actions workflow | **not run**; first run on your repo is the real test |
| SmolVLM2 review | **not run** (no access to the model hub) |
