"""Episode collection: SharpaEnv + expert -> immutable episode dirs.

Stage-1 scripted expert collection only produces genuine executed episodes
(never relabels video/hover material). Failures are kept with accurate
outcome labels for dynamics learning and diagnosis.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .data import EpisodeWriter, write_dataset_index
from .env import SUBSTEPS, SharpaEnv, atomic_json


class ScriptedExpertPolicy:
    """15 Hz wrapper around the diagnostic-style scripted expert.

    Emits normalized delta actions toward the current waypoint in joint
    space; waypoints come from the real phase machine in expert.py. This is
    a bootstrapping scripted expert, NEVER labelled learned control.
    """

    def __init__(self, env: SharpaEnv, waypoint_generator_factory):
        self.env = env
        self._factory = waypoint_generator_factory
        self.gen = None
        self.reset()

    def reset(self):
        self.gen = self._factory()
        self._current = None

    def action(self, obs, t):
        from .env import ACTION_DIM
        q_meas = obs["proprio"][:ACTION_DIM].astype(float)
        target = next(self.gen, None)
        if target is None:
            return np.zeros(ACTION_DIM, dtype=np.float32)
        delta = (np.asarray(target) - q_meas) / self.env.delta_scale
        return np.clip(delta, -1.0, 1.0).astype(np.float32)


def collect_episode(env: SharpaEnv, expert, episode_dir: str,
                    episode_id: str, seed: int = 0,
                    options: dict | None = None,
                    max_decisions: int = 600) -> dict:
    """Run one episode through the contract reset/step interface and write
    the immutable episode directory. Returns the manifest."""
    if hasattr(expert, "reset"):
        expert.reset()
    obs, info = env.reset(seed=seed, options=options)
    writer = EpisodeWriter(episode_dir, episode_id,
                           info.get("provenance", {}))
    writer.add_observation(obs, sim_time=float(info.get("sim_time", 0.0)))
    outcome = "timeout"
    for t in range(max_decisions):
        action = expert.action(obs, t) if hasattr(expert, "action") \
            else expert.expert_action()
        obs, reward, terminated, truncated, info = env.step(action)
        info["teacher_action"] = np.asarray(action, dtype=np.float32)
        sim_time = float(info.get("sim_time", float("nan")))
        if not math.isfinite(sim_time):
            raise ValueError("env.step info must provide finite sim_time")
        # evaluator-imposed cap must be recorded on the last step
        if t == max_decisions - 1 and not terminated:
            truncated = True
        writer.add_step(action, info, reward, terminated, truncated)
        writer.add_observation(obs, sim_time=sim_time)
        if terminated or truncated:
            break
    # success requires explicit on-target click AND clean lift, not raw
    # termination (which may be a safety/limit failure in future variants).
    priv = info or {}
    success = bool(
        priv.get("oracle_click_on_target", 0) > 0
        and priv.get("oracle_click_registered", 0) > 0)
    if terminated and success:
        outcome = "success"
    elif truncated:
        outcome = "truncated"
    manifest_extra = {
        "source_kind": "sim_geometric_expert",
        "seed": int(seed),
        "task": (options or {}).get("task_id", "tap"),
        "outcome": outcome,
        "training_eligible_for_bc": outcome == "success",
        "split": (options or {}).get("split", "train"),
        "config": dict(options or {}),
    }
    manifest = writer.finalize(manifest_extra)
    manifest["outcome"] = outcome
    return manifest


def collect_dataset(env: SharpaEnv, expert, out_dir: str,
                    n_episodes: int = 4, seeds=None,
                    splits=None, max_decisions: int = 600) -> dict:
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    episodes = []
    seeds = seeds or list(range(n_episodes))
    splits = splits or (["train"] * n_episodes)
    action_names = list(getattr(env, "provenance", {}).get("action_names", []))
    delta_scale = getattr(env, "provenance", {}).get("delta_scale")
    if len(action_names) != ACTION_DIM:
        raise ValueError("env provenance missing canonical action_names")
    if delta_scale is None:
        raise ValueError("env provenance missing delta_scale")
    for i in range(n_episodes):
        eid = f"ep-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{i:04d}"
        # explicit per-episode variation: seed and split are the group
        opts = {
            "task_id": "tap",
            "split": splits[i % len(splits)],
            "seed": int(seeds[i]),
        }
        m = collect_episode(env, expert, out / eid, eid, seed=seeds[i],
                            options=opts, max_decisions=max_decisions)
        episodes.append({
            "path": str(out / eid), "episode_id": eid,
            "split": opts["split"], "source_kind": m["source_kind"],
            "training_eligible_for_bc": m["training_eligible_for_bc"],
            "action_names": m.get("action_names", []),
            "delta_scale": m.get("delta_scale"),
        })
        atomic_json(out / "COLLECT-STATE.json", {
            "done": i + 1, "of": n_episodes,
            "last_outcome": m["outcome"],
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    write_dataset_index(out / "dataset.json", episodes,
                        action_names=action_names, delta_scale=delta_scale)
    return {"episodes": episodes, "index": str(out / "dataset.json")}
