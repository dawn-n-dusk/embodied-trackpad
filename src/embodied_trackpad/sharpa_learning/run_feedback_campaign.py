"""Run one or more relative-trackpad feedback-expert episodes natively.

Logs per-step q/action/contact/body-pairs/GUI/teacher phase and writes the
standard episode manifest + trajectory.npz.  After a run, paired camera videos
are encoded from the saved NPZ at 15 Hz and a frozen dataset index is written
to the common data path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np

from embodied_trackpad.sharpa_learning.data import EpisodeWriter, write_dataset_index
from embodied_trackpad.sharpa_learning.env import SharpaEnv
from embodied_trackpad.sharpa_learning.feedback_expert import FeedbackExpert


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _encode_video_from_npz(npz_path: Path, out_path: Path, key: str, fps: int = 15) -> dict:
    """Write an MP4 from an observation array in the NPZ (T+1, H, W, C)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(npz_path) as z:
        arr = np.asarray(z[key])
    if arr.ndim != 4 or arr.shape[0] < 4:
        return {"returncode": None, "frames": int(arr.shape[0]) if arr.ndim else 0,
                "error": "too few frames"}
    frames_dir = out_path.parent / f"{out_path.stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # Ensure uint8 RGB.
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    for i, img in enumerate(arr):
        # PIL is available in the native env.
        from PIL import Image
        Image.fromarray(img).save(frames_dir / f"{out_path.stem}_{i:05d}.png")
    rc = subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(fps), "-i",
         str(frames_dir / f"{out_path.stem}_%05d.png"),
         "-pix_fmt", "yuv420p", str(out_path)],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    # Keep no derived PNG frame copies once NPZ + MP4 are verified: the video
    # can be regenerated from the lossless NPZ observations on demand.
    if rc == 0 and out_path.exists():
        shutil.rmtree(frames_dir, ignore_errors=True)
    return {"returncode": int(rc), "frames": int(arr.shape[0])}


class _TeacherLogger:
    """Records teacher action, phase, contacts and GUI state per decision."""

    def __init__(self, expert: FeedbackExpert, log_path: Path):
        self.expert = expert
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logf = self.log_path.open("w")
        self._last_info: dict | None = None

    def log(self, t: int, obs: dict, action: np.ndarray, info: dict):
        q = np.asarray(obs["proprio"], dtype=float)[:29]
        force = float(obs["touch"][0])
        valid = int(obs["touch_valid"][0])
        priv = self.expert.env._privileged()
        contacts = info.get("contacts", {})
        body_pairs = sorted({
            "|".join(sorted([ct["body0"], ct["body1"]]))
            for f, pts in contacts.items() for ct in pts
        })
        entry = {
            "t": int(t),
            "sim_time": float(info.get("sim_time", float("nan"))),
            "phase": self.expert._phase,
            "tap_subphase": self.expert._tap_subphase,
            "q_meas": [round(float(v), 6) for v in q],
            "action": [round(float(v), 4) for v in action],
            "index_force_n": round(force, 5),
            "index_valid": valid,
            "cursor": priv.get("oracle_cursor_px"),
            "target": priv.get("oracle_target_px"),
            "cursor_over_target": bool(priv.get("oracle_cursor_over_target", False)),
            "click_registered": int(priv.get("oracle_click_registered", 0)),
            "click_on_target": int(priv.get("oracle_click_on_target", 0)),
            "body_pairs": body_pairs,
            "max_force_per_finger": {
                f: round(max((ct["force_n"] for ct in pts), default=0.0), 5)
                for f, pts in contacts.items()
            },
            "contacts_valid": bool(info.get("device", {}).get("events_valid", True)),
        }
        self.logf.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.logf.flush()

    def close(self):
        self.logf.close()


def _finalize_manifest(manifest: dict, episode_dir: Path, npz_path: Path,
                       teacher_log: Path,
                       source_manifest_path: Path | None = None) -> dict:
    manifest["teacher_log"] = str(teacher_log)
    manifest["videos"] = {
        "screen_rgb": _encode_video_from_npz(npz_path, episode_dir / "screen_rgb.mp4", "screen_rgb"),
        "wrist_rgb": _encode_video_from_npz(npz_path, episode_dir / "wrist_rgb.mp4", "wrist_rgb"),
    }
    manifest["source_manifest_sha256"] = _sha256_file(source_manifest_path) \
        if source_manifest_path and source_manifest_path.exists() else None
    (episode_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def run_feedback_episode(env: SharpaEnv, expert: FeedbackExpert,
                         episode_dir: Path, episode_id: str,
                         seed: int, max_decisions: int,
                         source_manifest_path: Path | None = None,
                         split: str = "train", compress: bool = False) -> dict:
    episode_dir.mkdir(parents=True, exist_ok=True)
    logger = _TeacherLogger(expert, episode_dir / "teacher.jsonl")
    writer = EpisodeWriter(episode_dir, episode_id,
                           env.provenance if hasattr(env, "provenance") else {},
                           compress=compress)

    obs, info = env.reset(seed=seed, options={"task_id": "tap"})
    reset_sim_time = float(info.get("sim_time", 0.0))
    writer.add_observation(obs, sim_time=reset_sim_time)
    logger.log(0, obs, np.zeros(29, dtype=np.float32), info)

    outcome = "timeout"
    failure_reason = None
    target_px = info.get("oracle_target_px", (0.0, 0.0))
    last_obs = obs
    last_info = info
    for t in range(max_decisions):
        action = expert.action(obs, t)
        # Expert entered fail: still execute one honest env.step with the legal
        # hold action, retaining real physics observations / reward / time.
        if expert._phase == "fail":
            failure_reason = "expert_phase_fail"
            outcome = "truncated"
        obs, reward, terminated, truncated, info = env.step(action)
        info["teacher_action"] = np.asarray(action, dtype=np.float32)
        sim_time = float(info.get("sim_time", float("nan")))
        # Evaluator-imposed cap: the last non-terminal step must be truncated.
        if failure_reason == "expert_phase_fail" and not terminated:
            truncated = True
        if t == max_decisions - 1 and not terminated:
            truncated = True
            outcome = "truncated"
        logger.log(t, obs, action, info)
        writer.add_step(action, info, reward, terminated, truncated)
        writer.add_observation(obs, sim_time=sim_time)
        # Record the actual target from privileged info on the first step.
        if t == 0:
            target_px = info.get("oracle_target_px", target_px)
        last_obs = obs
        last_info = info
        if terminated:
            outcome = "success"
            break
        if truncated:
            break

    priv = info or {}
    success = bool(
        priv.get("oracle_click_on_target", 0) > 0
        and priv.get("oracle_click_registered", 0) > 0)
    if outcome == "success" and not success:
        outcome = "truncated"
    manifest_extra = {
        "source_kind": "feedback_expert_v1",
        "seed": int(seed),
        "task": "tap",
        "outcome": outcome,
        "failure_reason": failure_reason,
        "training_eligible_for_bc": outcome == "success" and failure_reason is None,
        "split": split,
        "config": {"task_id": "tap", "split": split, "seed": int(seed),
                   "target_px": list(target_px)},
    }
    manifest = writer.finalize(manifest_extra)
    manifest["outcome"] = outcome
    if failure_reason is not None:
        manifest["failure_reason"] = failure_reason

    # Save still frames before close (SimulationApp.close exits).
    env.rep.orchestrator.step(delta_time=0, pause_timeline=False)
    frames_dir = episode_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for cam_name, ann in env.products.items():
        data = ann.get_data()
        arr = data.get("data") if isinstance(data, dict) else data
        if arr is not None and getattr(arr, "size", 0):
            from PIL import Image
            Image.fromarray(arr).save(frames_dir / f"{cam_name}_last.png")

    npz_path = episode_dir / "trajectory.npz"
    manifest = _finalize_manifest(manifest, episode_dir, npz_path,
                                  logger.log_path, source_manifest_path)
    logger.close()
    return manifest


def build_dataset_index(data_root: Path, episodes: list[dict],
                        action_names: list[str] | None = None,
                        delta_scale: list[float] | None = None) -> Path:
    """Write the frozen common dataset index including all episodes."""
    data_root.mkdir(parents=True, exist_ok=True)
    index_path = data_root / "dataset.json"
    write_dataset_index(index_path, episodes,
                        action_names=action_names, delta_scale=delta_scale)
    return index_path


def _write_run_summary(out: Path, episodes: list[dict],
                       source_manifest: Path | None) -> Path:
    """Write a run summary with source manifest and checkpoint hashes."""
    summary = {
        "run_id": out.name,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_manifest_sha256": _sha256_file(source_manifest)
            if source_manifest and source_manifest.exists() else None,
        "episodes": [
            {
                "episode_id": e["episode_id"],
                "split": e.get("split"),
                "source_kind": e.get("source_kind"),
                "training_eligible_for_bc": e.get("training_eligible_for_bc"),
                "manifest": str(Path(e["path"]) / "manifest.json"),
                "manifest_sha256": _sha256_file(Path(e["path"]) / "manifest.json")
                    if (Path(e["path"]) / "manifest.json").exists() else None,
            }
            for e in episodes
        ],
    }
    path = out / "RUN-SUMMARY.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return path


def run_campaign(stage: str, out_dir: str, data_root: str,
                 gpu_index: int = 1, seed: int = 0,
                 n_episodes: int = 1, max_decisions: int = 400,
                 source_manifest_path: str | None = None,
                 split: str = "train", compress: bool = False,
                 seeds: list[int] | None = None) -> dict:
    """Run ``n_episodes`` feedback-expert episodes (or the exact ``seeds``
    list when given, so the launch receipt matches the actual reset seeds)."""
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_root_p = Path(data_root).resolve()
    data_root_p.mkdir(parents=True, exist_ok=True)
    source_manifest = Path(source_manifest_path) if source_manifest_path else None
    if seeds is not None:
        seed_list = [int(s) for s in seeds]
        if not seed_list:
            raise ValueError("explicit seeds list must not be empty")
    else:
        seed_list = [int(seed) + i for i in range(int(n_episodes))]

    env = SharpaEnv({"stage": stage, "out_dir": str(out),
                     "gpu_index": int(gpu_index)})
    episodes = []
    action_names = None
    delta_scale = None
    run_namespace = uuid.uuid4().hex
    try:
        for i, ep_seed in enumerate(seed_list):
            eid = f"fb-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{run_namespace}-{i:04d}"
            ep_dir = out / eid
            expert = FeedbackExpert(env)
            manifest = run_feedback_episode(
                env, expert, ep_dir, eid,
                seed=ep_seed, max_decisions=int(max_decisions),
                source_manifest_path=source_manifest,
                split=split,
                compress=compress,
            )
            episodes.append({
                "path": str(ep_dir), "episode_id": eid,
                "split": manifest.get("split", split),
                "source_kind": manifest.get("source_kind"),
                "training_eligible_for_bc": manifest.get("training_eligible_for_bc"),
                "seed": int(ep_seed),
                "action_names": manifest.get("action_names", []),
                "delta_scale": manifest.get("delta_scale"),
            })
            if action_names is None:
                action_names = manifest.get("action_names")
                delta_scale = manifest.get("delta_scale")
        # Write the dataset index and run summary BEFORE closing the env:
        # Isaac Sim fast_shutdown exits the process.
        index_path = build_dataset_index(data_root_p, episodes,
                                          action_names=action_names,
                                          delta_scale=delta_scale)
        _write_run_summary(out, episodes, source_manifest)
    finally:
        env.close()

    return {"episodes": episodes, "index": str(index_path)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--gpu-index", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-episodes", type=int, default=1)
    ap.add_argument("--seeds", default=None,
                    help="Comma-separated exact per-episode seeds; overrides "
                         "--seed/--n-episodes so the receipt matches actual resets.")
    ap.add_argument("--max-decisions", type=int, default=400)
    ap.add_argument("--source-manifest", default=None)
    ap.add_argument("--split", default="train",
                    choices=["train", "val", "test"])
    ap.add_argument("--compress-npz", action="store_true",
                    help="Lossless DEFLATE NPZ compression (campaign storage scope).")
    args = ap.parse_args(argv)

    result = run_campaign(
        stage=args.stage,
        out_dir=args.out_dir,
        data_root=args.data_root,
        gpu_index=args.gpu_index,
        seed=args.seed,
        n_episodes=args.n_episodes,
        max_decisions=args.max_decisions,
        source_manifest_path=args.source_manifest,
        split=args.split,
        compress=args.compress_npz,
        seeds=([int(s) for s in args.seeds.split(",") if s.strip()]
               if args.seeds else None),
    )
    print(json.dumps({
        "status": "complete",
        "index": result["index"],
        "episodes": [
            {"episode_id": e["episode_id"],
             "outcome": (Path(e["path"]) / "manifest.json").exists(),
             "training_eligible_for_bc": e["training_eligible_for_bc"]}
            for e in result["episodes"]
        ],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
