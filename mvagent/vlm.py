"""OPTIONAL second opinion on the footage with a small video-language model (SmolVLM2).

Off by default (USE_VLM=1 to enable, and `pip install -r requirements-vlm.txt`).
For every slot it looks at one frame from the middle of the clip and answers two questions:
"is a face shown as a close-up while the song has vocals?" and "is there a logo or overlaid text?".
Flagged windows are
passed back to the planner as `banned`, and the plan is rebuilt.

NOTE: this module could not be run in the authoring sandbox (no access to the model hub), so it
is the least-tested part of the repo. If it errors, the pipeline logs the problem and continues
without it.
"""
import subprocess
import tempfile
from pathlib import Path

MODEL = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"   # 500M / 2.2B variants also exist
QUESTION = ("Is a person's face shown large in the frame, as in a close-up where "
            "you can see their mouth? Answer with only yes or no.")

LOGO_QUESTION = ("Is there a large logo, brand name, caption or title text overlaid on this image? "
                 "Answer with only yes or no.")

_cache = {}


def _load():
    if "m" not in _cache:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        proc = AutoProcessor.from_pretrained(MODEL)
        model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.float32)
        model.eval()
        _cache["m"] = (proc, model)
    return _cache["m"]


def _frame(video, t, out):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(video),
                    "-frames:v", "1", "-vf", "scale=512:-2", str(out)], check=True)


def ask_yes_no(image_path, question=QUESTION) -> bool:
    import torch

    proc, model = _load()
    msgs = [{"role": "user", "content": [{"type": "image", "path": str(image_path)},
                                          {"type": "text", "text": question}]}]
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    text = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return text.strip().lower().startswith("yes")


def review(plan, vocal_threshold=0.4):
    """Return the list of (video_idx, start, end) windows to ban.

    Two checks: (1) close-up face under vocals, (2) logo / overlaid text in any clip."""
    flagged = []
    with tempfile.TemporaryDirectory() as d:
        for p in plan:
            span = p.get("src_len") or p["len"]
            t = p["src_start"] + span / 2
            img = Path(d) / f"s{p['i']}.jpg"
            _frame(p["video"], t, img)
            reason = None
            if p["vocal"] >= vocal_threshold and ask_yes_no(img):
                reason = "close-up face under vocals"
            elif ask_yes_no(img, LOGO_QUESTION):
                reason = "logo or overlaid text"
            if reason:
                flagged.append((p["video_idx"], p["src_start"], p["src_start"] + span))
                print(f"[vlm] slot {p['i']}: {reason} -> re-pick")
    return flagged
