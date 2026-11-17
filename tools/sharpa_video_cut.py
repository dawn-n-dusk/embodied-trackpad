#!/usr/bin/env python3
"""Cut SHARPA trackpad gesture episodes from the two source videos.

For each episode: hand.mp4 (hand+pad crop), result.mp4 (visible screen
result), paired.mp4 (side-by-side review), manifest.json (source URL/hash,
timestamps, crop boxes, labels, flags).

Usage: python sharpa_video_cut.py --root OUTPUT_DIR --spec EPISODES.json
Existing hand/result/paired files are kept (only manifests refreshed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

THREADS = 4


def sha256(path: Path, _bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(_bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{p.stderr[-2000:]}")


def ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,width,height,nb_frames,avg_frame_rate,r_frame_rate:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    fps = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1"
    num, den = (int(x) for x in fps.split("/"))
    return {"width": v.get("width"), "height": v.get("height"),
            "frames": int(v.get("nb_frames") or 0),
            "fps": round(num / den, 3) if den else 0.0,
            "duration_s": round(float(d["format"]["duration"]), 3)}


def cut(src: Path, start: float, end: float, crop: tuple[int, int, int, int] | None,
        out: Path, with_audio: bool = True) -> None:
    vf = f"crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]}" if crop else "null"
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
           "-threads", str(THREADS),
           "-vf", vf, "-c:v", "libx264", "-crf", "18", "-preset", "medium",
           "-pix_fmt", "yuv420p"]
    if with_audio:
        cmd += ["-c:a", "aac", "-b:a", "128k"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", str(out)]
    run(cmd)


def pair(hand: Path, result: Path, out: Path) -> None:
    # scale both to same height, stack horizontally, keep hand audio
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(hand), "-i", str(result),
           "-threads", str(THREADS),
           "-filter_complex",
           "[0:v]scale=-2:720[l];[1:v]scale=-2:720[r];[l][r]hstack=inputs=2[v]",
           "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-crf", "18",
           "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac",
           "-movflags", "+faststart", str(out)]
    run(cmd)


def validate(ep: dict, src_dims: tuple[int, int]) -> None:
    import math
    eid = ep["id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", eid) or ".." in eid:
        raise ValueError(f"unsafe episode id: {eid!r}")
    start, end = ep["start"], ep["end"]
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        raise ValueError(f"{eid}: bad interval {start}-{end}")
    sw, sh = src_dims
    for key in ("hand_crop", "result_crop"):
        c = ep.get(key)
        if c is None:
            continue
        if len(c) != 4 or any((not isinstance(v, int)) or v < 0 for v in c):
            raise ValueError(f"{eid}: bad {key} {c}")
        x, y, w, h = c
        if x + w > sw or y + h > sh:
            raise ValueError(f"{eid}: {key} {c} outside {sw}x{sh}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="output root containing DOWNLOADS/ and episodes/")
    ap.add_argument("--spec", required=True, help="episode spec JSON")
    ap.add_argument("--only", action="append", default=[],
                    help="episode id(s) to process; comma-separated or repeatable")
    ap.add_argument("--metadata-only", action="store_true",
                    help="refresh manifests without re-encoding; refuses on any "
                         "spec/media mismatch vs previous manifest")
    args = ap.parse_args()
    root = Path(args.root)
    spec = json.loads(Path(args.spec).read_text())
    only = {i for group in args.only for i in group.split(",") if i}
    outroot = root / "episodes"
    reports = []
    for ep in spec["episodes"]:
        eid = ep["id"]
        if only and eid not in only:
            continue
        src = root / "DOWNLOADS" / ep["source_file"]
        src_info = ffprobe(src)
        validate(ep, (src_info["width"], src_info["height"]))
        edir = outroot / eid
        edir.mkdir(parents=True, exist_ok=True)
        start, end = ep["start"], ep["end"]
        hand_crop = tuple(ep["hand_crop"]) if ep.get("hand_crop") else None
        result_crop = tuple(ep["result_crop"]) if ep.get("result_crop") else None

        hand = edir / "hand.mp4"
        result = edir / "result.mp4"
        paired = edir / "paired.mp4"

        if args.metadata_only:
            prev_f = edir / "manifest.json"
            if not (hand.exists() and result.exists() and paired.exists()):
                raise SystemExit(f"{eid}: metadata-only but media missing; recut instead")
            prev = json.loads(prev_f.read_text())
            for key, old, new in (
                ("source file", prev["source"]["file"], ep["source_file"]),
                ("source sha256", prev["source"]["sha256"], sha256(src)),
                ("start", prev["source_start_s"], start),
                ("end", prev["source_end_s"], end),
                ("hand crop", prev["hand"]["crop_xywh_requested"], list(hand_crop) if hand_crop else None),
                ("result crop", prev["result"]["crop_xywh_requested"], list(result_crop) if result_crop else None),
                ("hand effective crop", prev["hand"]["crop_xywh_effective"],
                 list(ep.get("hand_crop_effective", hand_crop)) if (ep.get("hand_crop_effective") or hand_crop) else None),
                ("result effective crop", prev["result"]["crop_xywh_effective"],
                 list(ep.get("result_crop_effective", result_crop)) if (ep.get("result_crop_effective") or result_crop) else None),
            ):
                if old != new:
                    raise SystemExit(
                        f"{eid}: metadata-only mismatch ({key}: {old!r} != {new!r}); "
                        f"recut this episode instead of relabelling old media")
            for f in (hand, result, paired):
                if sha256(f) != (prev.get("hand" if f == hand else "result" if f == result else "paired", {}).get("sha256")):
                    raise SystemExit(f"{eid}: media hash changed since previous manifest; recut instead")
        else:
            cut(src, start, end, hand_crop, hand)
            cut(src, start, end, result_crop, result)
            pair(hand, result, paired)

        # verify all files before publishing any manifest
        for f in (hand, result, paired):
            run(["ffmpeg", "-v", "error", "-threads", str(THREADS), "-i", str(f),
                 "-f", "null", "-"])
        hp, rp, pp = ffprobe(hand), ffprobe(result), ffprobe(paired)
        if hp["frames"] != rp["frames"]:
            raise SystemExit(f"{eid}: hand/result frame mismatch {hp['frames']} vs {rp['frames']}")
        for tag, probe, eff, full in (
            ("hand", hp, ep.get("hand_crop_effective", hand_crop), hand_crop is None),
            ("result", rp, ep.get("result_crop_effective", result_crop), result_crop is None),
        ):
            if full and eff is None:
                continue  # full-frame output, no crop declared
            if eff is None:
                raise SystemExit(f"{eid}: {tag} has output {probe['width']}x{probe['height']} but no effective crop declared")
            if (eff[2], eff[3]) != (probe["width"], probe["height"]):
                raise SystemExit(
                    f"{eid}: {tag} effective crop dims {eff[2]}x{eff[3]} != actual output "
                    f"{probe['width']}x{probe['height']}")

        manifest = {
            "episode_id": eid,
            "source": {
                "url": ep["url"],
                "file": ep["source_file"],
                "sha256": sha256(src),
                "fps": src_info["fps"],
                "width": src_info["width"],
                "height": src_info["height"],
                "upload_date": ep.get("upload_date"),
                "uploader": ep.get("uploader"),
            },
            "source_start_s": start,
            "source_end_s": end,
            "hand": {"file": "hand.mp4", "crop_xywh_requested": hand_crop,
                     "crop_xywh_effective": ep.get("hand_crop_effective", hand_crop),
                     **hp, "sha256": sha256(hand)},
            "result": {"file": "result.mp4", "crop_xywh_requested": result_crop,
                       "crop_xywh_effective": ep.get("result_crop_effective", result_crop),
                       **rp, "sha256": sha256(result)},
            "paired": {"file": "paired.mp4", **pp,
                       "sha256": sha256(paired)},
            "labels": ep["labels"],
            "source_kind": ep["source_kind"],
            "provenance": ep["provenance"],
            "visible_hand_motion": ep.get("visible_hand_motion", True),
            "reference_only": ep.get("reference_only", False),
            "observed_result": ep.get("observed_result", ""),
            "pairing_confidence": ep.get("pairing_confidence", "medium"),
            "limitations": ep.get("limitations", ""),
            "flags": {
                "robot_action_labels_available": False,
                "training_eligible_for_robot_action_bc": False,
                "notes": ep.get("flags_note", ""),
            },
        }
        tmp = edir / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        os.replace(tmp, edir / "manifest.json")
        reports.append({"id": eid, "hand": hp["frames"], "result": rp["frames"],
                        "paired_dur": pp["duration_s"]})
        print(json.dumps(reports[-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
