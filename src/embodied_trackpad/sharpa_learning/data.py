"""Episode data writer + dataset index (contract: schema sharpa-learning-v1).

Each immutable episode directory carries manifest.json + trajectory.npz
(no pickle). Actors/loaders whitelist observation keys; oracle_* arrays are
privileged. NPZ lengths follow the contract: observation arrays T+1,
action/executed_q_ref/reward/terminated/truncated/teacher_* length T.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

SCHEMA = "sharpa-learning-v1"
OBS_KEYS = ("screen_rgb", "wrist_rgb", "proprio", "touch", "touch_valid")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _npz_array_shas(npz_path: Path) -> dict:
    out = {}
    with np.load(npz_path) as z:
        for k in z.files:
            out[k] = hashlib.sha256(z[k].tobytes()).hexdigest()
    return out


class EpisodeWriter:
    """Accumulates one episode from SharpaEnv reset/step and writes the
    immutable episode directory. Observations precede actions; the final
    observation is appended after the last step."""

    def __init__(self, episode_dir, episode_id: str, provenance: dict):
        self.dir = Path(episode_dir)
        if (self.dir / "manifest.json").exists():
            raise FileExistsError(
                f"refusing to overwrite existing episode: {self.dir}")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.episode_id = episode_id
        self.provenance = dict(provenance)
        self.obs = {k: [] for k in OBS_KEYS}
        self.action: list = []
        self.executed_q_ref: list = []
        self.reward: list = []
        self.terminated: list = []
        self.truncated: list = []
        self.sim_time: list = []
        self.teacher_action: list = []
        self.teacher_valid: list = []
        self.oracle: dict[str, list] = {}
        self.info_first: dict | None = None

    def add_observation(self, obs: dict, sim_time: float = 0.0):
        for k in OBS_KEYS:
            self.obs[k].append(np.asarray(obs[k]))
        self.sim_time.append(float(sim_time))

    def add_step(self, action, info, reward, terminated, truncated):
        self.action.append(np.asarray(action, dtype=np.float32))
        self.executed_q_ref.append(
            np.asarray(info["executed_q_ref"], dtype=np.float32))
        self.reward.append(float(reward))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))
        ta = info.get("teacher_action")
        self.teacher_action.append(
            np.zeros(len(action), dtype=np.float32) if ta is None
            else np.asarray(ta, dtype=np.float32))
        self.teacher_valid.append(bool(ta is not None))
        if self.info_first is None:
            self.info_first = {"raw_action": info.get("raw_action")}
        for k, v in info.items():
            if k.startswith("oracle_"):
                self.oracle.setdefault(k, []).append(v)

    def finalize(self, manifest_extra: dict | None = None) -> dict:
        T = len(self.action)
        npz_path = self.dir / "trajectory.npz"
        arrays = {k: np.stack(v) for k, v in self.obs.items()}
        arrays.update({
            "action": np.stack(self.action) if T else np.zeros(
                (0, len(self.provenance.get("action_names", []))),
                dtype=np.float32),
            "executed_q_ref": np.stack(self.executed_q_ref),
            "reward": np.asarray(self.reward, dtype=np.float32),
            "terminated": np.asarray(self.terminated, dtype=np.bool_),
            "truncated": np.asarray(self.truncated, dtype=np.bool_),
            "sim_time": np.asarray(self.sim_time, dtype=np.float64),
            "teacher_action": np.stack(self.teacher_action),
            "teacher_valid": np.asarray(self.teacher_valid, dtype=np.bool_),
        })
        for k, v in self.oracle.items():
            arrays[k] = np.asarray(v)
        # verify contract lengths before writing
        for k, arr in arrays.items():
            expected = T + 1 if k in OBS_KEYS or k == "sim_time" else T
            if arr.shape[0] != expected:
                raise ValueError(
                    f"array {k} length {arr.shape[0]} != expected {expected}")
        np.savez(npz_path, **arrays)
        manifest = {
            "schema": SCHEMA,
            "episode_id": self.episode_id,
            "action_names": list(self.provenance.get("action_names", [])),
            "delta_scale": self.provenance.get("delta_scale"),
            "joint_limits": self.provenance.get("joint_limits"),
            "measured_contact_backend": self.provenance.get("contact_backend"),
            "stage": self.provenance.get("stage"),
            "timestamps": {
                "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                              time.gmtime())},
            "array_sha256": _npz_array_shas(npz_path),
            "trajectory_sha256": _sha256_file(npz_path),
            "length_T": int(T),
        }
        if manifest_extra:
            manifest.update(manifest_extra)
        (self.dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        return manifest


def write_dataset_index(index_path, episodes: list[dict],
                        action_names: list[str] | None = None,
                        delta_scale: list[float] | None = None) -> None:
    """dataset.json: top-level action_names/delta_scale + episodes.

    ``bc_ppo.load_dataset`` requires the top-level ``action_names`` and
    ``delta_scale`` fields; they must agree with every episode manifest.
    """
    if action_names is None:
        raise ValueError("write_dataset_index requires action_names")
    if delta_scale is None:
        raise ValueError("write_dataset_index requires delta_scale")
    index = {
        "schema": SCHEMA,
        "action_names": list(action_names),
        "delta_scale": [float(v) for v in delta_scale],
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "episodes": episodes,
    }
    Path(index_path).write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n")
