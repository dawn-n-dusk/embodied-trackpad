"""Lazy RGB supervised-example corpus for the LEAP v1 Full visual policy mainline.

This module prepares actual RGB supervised examples for the user-authorized
full pretrained pi0.5 mainline WITHOUT importing OpenPI. It consumes an
audited, coordinator-approved ``sample_index.jsonl`` (the same artifact the
small-policy baseline uses) plus the referenced PNG captures, and yields
``VisualExample`` records that separate the actor observation from the target
actions and provenance.

Native v1 semantics (see docs/decisions/0009-visual-policy-mainline.zh-CN.md):

* simulation dt is exactly 0.002 s and the action horizon is 25 substeps
  (0.05 s of executed control); the OpenPI default 32-dim/50-step layout and
  any padding/horizon conversion are downstream concerns and NOT applied here.
* actions are the actual executed absolute 16-channel joint references
  (``labels.executed_ctrl_rad``), never deltas, never nominal endpoints.
* the observation whitelist is exactly ``screen_rgb`` (uint8 HWC, native
  800x600, no resize/normalization), ``q_rad16``, ``qvel_rad_s16``,
  ``previous_ctrl_rad16`` and the absent-sensor placeholders (fsr, motor
  current, wrist are null with all-false masks). ``error_xy_px``,
  target_class, teacher state, force, future images and any oracle/GUI truth
  are privileged/metadata and never enter the actor observation. There is no
  language prompt and target_class is never fed as text.

Integrity model: the whole index SHA-256 and every structural/split/label
invariant is validated by :func:`small_policy.load_sample_index` before any
image work. ``example(i)`` then resolves ONLY the referenced PNG inside
``evidence_root`` (no absolute refs, no path traversal, no symlink escapes),
verifies the exact-byte SHA-256 recorded in the index, and decodes with
Pillow. Nothing is cached: repeated calls decode fresh bytes so callers can
mutate returned arrays without aliasing corpus state.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from embodied_trackpad.leap_v1 import small_policy

DT_S = 0.002  # native v1 simulation substep (visual_physics/model, pilot_dataset)
HORIZON = 25  # native v1 action substeps per example
HORIZON_S = HORIZON * DT_S  # 0.05 s of executed absolute control
NUM_CHANNELS = 16  # native v1 motor/joint channels; downstream 32-dim pad pending
PNG_WIDTH = 800
PNG_HEIGHT = 600
ACTION_SEMANTICS = "absolute_executed_joint_reference_rad"

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_CORPUS_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_SPLIT_NAMES = ("train", "validation", "test")

# Actor observation whitelist. Anything else (error_xy_px, target_class,
# teacher state, force, labels, future images, oracle/cursor truth) stays in
# the index row and out of the actor input.
OBSERVATION_KEYS = frozenset(
    {
        "screen_rgb",
        "q_rad16",
        "qvel_rad_s16",
        "previous_ctrl_rad16",
        "fsr_N4",
        "motor_current_A16",
        "wrist",
        "fsr_valid4",
        "motor_current_valid16",
        "wrist_valid",
    }
)


class VisualCorpusError(ValueError):
    """Raised when a corpus, sample, image, label or timing check fails."""


@dataclass(frozen=True)
class VisualExample:
    """One supervised example; observation/target/provenance are separated."""

    observation: dict[str, Any]
    target: dict[str, Any]
    provenance: dict[str, Any]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _check_sha256_hex(value: Any, name: str) -> str:
    sha = str(value)
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha.lower()):
        raise VisualCorpusError(f"{name} is not a 64-character lowercase/hex SHA-256")
    return sha.lower()


def _finite_f32(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    arr64 = np.asarray(value, dtype=np.float64)
    if arr64.shape != shape:
        raise VisualCorpusError(f"{name} shape mismatch: got {arr64.shape}, want {shape}")
    if not np.isfinite(arr64).all():
        raise VisualCorpusError(f"{name} contains non-finite values")
    with np.errstate(over="ignore"):
        arr32 = arr64.astype(np.float32)
    if not np.isfinite(arr32).all():
        raise VisualCorpusError(f"{name} overflows float32")
    return arr32


class VisualCorpus:
    """Validated, lazily decoded visual corpus over one approved sample index.

    ``index_path`` is the coordinator-approved JSONL sample index and
    ``expected_index_sha256`` must match its exact bytes. ``evidence_root``
    is the runtime root that relative ``actor_input.png_ref`` paths resolve
    against. ``corpus_id`` namespaces sample provenance so legacy episode /
    frame numbers from different indexes never collide.
    """

    def __init__(
        self,
        index_path: str | Path,
        evidence_root: str | Path,
        expected_index_sha256: str,
        corpus_id: str,
    ) -> None:
        if not isinstance(corpus_id, str) or not _CORPUS_ID_RE.fullmatch(corpus_id):
            raise VisualCorpusError("corpus_id must be nonempty matching [A-Za-z0-9_-]+")
        self.corpus_id = corpus_id
        self._expected_index_sha = _check_sha256_hex(expected_index_sha256, "expected_index_sha256")

        self._index_path = Path(index_path)
        if not self._index_path.is_file():
            raise VisualCorpusError(f"index not found: {self._index_path}")
        actual_index_sha = small_policy.sha256_file(self._index_path)
        if actual_index_sha.lower() != self._expected_index_sha:
            raise VisualCorpusError(
                f"index SHA-256 mismatch: expected {self._expected_index_sha[:16]}..., "
                f"got {actual_index_sha[:16]}..."
            )

        root = Path(evidence_root).resolve()
        if not root.is_dir():
            raise VisualCorpusError(f"evidence_root is not a directory: {evidence_root}")
        self._evidence_root = root

        # Whole-index structural/split/label validation BEFORE any image work.
        # This runs small_policy.load_sample_index on the entire index exactly
        # once per corpus; rows are never merged across indexes.
        self._records = small_policy.load_sample_index(self._index_path)
        if not self._records:
            raise VisualCorpusError("sample index is empty")
        for i, rec in enumerate(self._records):
            actor = rec["actor_input"]
            if not isinstance(actor.get("png_ref"), str) or not actor["png_ref"]:
                raise VisualCorpusError(f"row {i}: actor_input.png_ref missing or empty")
            _check_sha256_hex(actor.get("png_sha256"), f"row {i} png_sha256")

        self._index_sha256 = actual_index_sha.lower()
        self._splits: dict[str, list[int]] = {name: [] for name in _SPLIT_NAMES}
        for i, rec in enumerate(self._records):
            self._splits[str(rec["split"])].append(i)

    @property
    def index_path(self) -> Path:
        return self._index_path

    @property
    def index_sha256(self) -> str:
        return self._index_sha256

    @property
    def split_counts(self) -> dict[str, int]:
        return {name: len(ids) for name, ids in self._splits.items()}

    def split_indices(self, split: str) -> list[int]:
        """Global example indices belonging to ``split``, in index order."""
        if split not in self._splits:
            raise VisualCorpusError(f"unknown split {split!r}")
        return list(self._splits[split])

    def __len__(self) -> int:
        return len(self._records)

    def _resolve_png(self, rec: dict[str, Any], row: int) -> Path:
        ref = rec["actor_input"]["png_ref"]
        ref_path = Path(ref)
        if ref_path.is_absolute():
            raise VisualCorpusError(f"row {row}: absolute png_ref rejected: {ref!r}")
        resolved = (self._evidence_root / ref_path).resolve()
        try:
            resolved.relative_to(self._evidence_root)
        except ValueError as exc:
            raise VisualCorpusError(
                f"row {row}: png_ref escapes evidence_root: {ref!r}"
            ) from exc
        if not resolved.is_file():
            raise VisualCorpusError(f"row {row}: referenced PNG not found: {ref!r}")
        return resolved

    def _decode_png(self, png_path: Path, expected_sha: str, row: int) -> np.ndarray:
        data = png_path.read_bytes()
        if not data.startswith(PNG_MAGIC):
            raise VisualCorpusError(f"row {row}: {png_path.name} is not a PNG file")
        actual_sha = _sha256_bytes(data)
        if actual_sha.lower() != expected_sha.lower():
            raise VisualCorpusError(
                f"row {row}: PNG SHA-256 mismatch for {png_path.name}: "
                f"index {expected_sha[:16]}... vs actual {actual_sha[:16]}..."
            )
        try:
            with Image.open(BytesIO(data)) as img:
                if img.format != "PNG":
                    raise VisualCorpusError(f"row {row}: decoded format is {img.format!r}, not PNG")
                if img.size != (PNG_WIDTH, PNG_HEIGHT):
                    raise VisualCorpusError(
                        f"row {row}: PNG size {img.size} != ({PNG_WIDTH}, {PNG_HEIGHT})"
                    )
                if img.mode == "RGB":
                    rgb = np.array(img, dtype=np.uint8, copy=True)
                elif img.mode == "RGBA":
                    # The sandbox captures are RGBA with a fully opaque alpha
                    # channel; verify opacity and drop alpha losslessly rather
                    # than compositing onto an assumed background.
                    rgba = np.array(img, dtype=np.uint8, copy=True)
                    if (rgba[..., 3] != 255).any():
                        raise VisualCorpusError(
                            f"row {row}: RGBA PNG has non-opaque alpha; refusing to composite"
                        )
                    rgb = np.ascontiguousarray(rgba[..., :3])
                else:
                    raise VisualCorpusError(f"row {row}: PNG mode {img.mode!r} is not RGB/RGBA")
        except VisualCorpusError:
            raise
        except Exception as exc:
            raise VisualCorpusError(f"row {row}: PNG decode failed: {exc}") from exc
        if rgb.shape != (PNG_HEIGHT, PNG_WIDTH, 3) or rgb.dtype != np.uint8:
            raise VisualCorpusError(f"row {row}: decoded RGB shape/dtype unexpected: {rgb.shape} {rgb.dtype}")
        return rgb

    def _check_absent_sensors(self, actor: dict[str, Any], row: int) -> dict[str, Any]:
        valid = actor.get("valid")
        if not isinstance(valid, dict):
            raise VisualCorpusError(f"row {row}: actor_input.valid missing")
        for key, value in (("fsr_N4", None), ("motor_current_A16", None), ("wrist", None)):
            if actor.get(key) is not None:
                raise VisualCorpusError(f"row {row}: actor_input[{key!r}] must be null (sensor absent)")
        fsr_ok = valid.get("fsr_valid4") == [False] * 4
        cur_ok = valid.get("motor_current_valid16") == [False] * NUM_CHANNELS
        wrist_ok = valid.get("wrist_valid") is False
        if not (fsr_ok and cur_ok and wrist_ok):
            raise VisualCorpusError(f"row {row}: missing-sensor masks must be all-false")
        return {
            "fsr_N4": None,
            "motor_current_A16": None,
            "wrist": None,
            "fsr_valid4": [False] * 4,
            "motor_current_valid16": [False] * NUM_CHANNELS,
            "wrist_valid": False,
        }

    def _check_timing(self, rec: dict[str, Any], row: int) -> dict[str, Any]:
        window = small_policy.extract_action_window(rec)  # dt == DT_S, horizon == HORIZON
        first = float(rec["action_window"]["first_sim_time_s"])
        end = float(rec["action_window"]["end_sim_time_s"])
        prev = float(rec["action_window"]["consumed_prev_sim_time_s"])
        if not np.isfinite([first, end, prev]).all():
            raise VisualCorpusError(f"row {row}: action_window sim times non-finite")
        tol = 1e-6
        if abs((first - prev) - DT_S) > tol:
            raise VisualCorpusError(
                f"row {row}: first_sim_time_s - consumed_prev_sim_time_s != dt {DT_S}"
            )
        if abs((end - first) - (HORIZON - 1) * DT_S) > tol:
            raise VisualCorpusError(
                f"row {row}: end_sim_time_s - first_sim_time_s != {(HORIZON - 1) * DT_S}"
            )
        if window["dt_s"] != DT_S:
            raise VisualCorpusError(f"row {row}: dt_s {window['dt_s']} != native {DT_S}")
        return {
            "dt_s": DT_S,
            "horizon": HORIZON,
            "horizon_s": HORIZON_S,
            "first_step_index": window["first_step_index"],
            "end_exclusive_step_index": window["end_exclusive_step_index"],
            "consumed_prev_sim_time_s": prev,
            "first_sim_time_s": first,
            "end_sim_time_s": end,
            "action_semantics": ACTION_SEMANTICS,
        }

    def _build(self, i: int) -> VisualExample:
        rec = self._records[i]
        actor = rec["actor_input"]
        labels = rec["labels"]
        split = str(rec["split"])

        png_path = self._resolve_png(rec, i)
        rgb = self._decode_png(png_path, str(actor["png_sha256"]), i)

        q = _finite_f32(actor["q_rad16"], "q_rad16", (NUM_CHANNELS,))
        qvel = _finite_f32(actor["qvel_rad_s16"], "qvel_rad_s16", (NUM_CHANNELS,))
        prev_ctrl = _finite_f32(actor["previous_ctrl_rad16"], "previous_ctrl_rad16", (NUM_CHANNELS,))
        sensors = self._check_absent_sensors(actor, i)
        timing = self._check_timing(rec, i)

        # Native 25x16 absolute executed joint references (not deltas, not
        # nominal endpoints). extract_label_delta re-validates the optional
        # cumulative-offset consistency; its delta output is not used here.
        small_policy.extract_label_delta(rec)
        target_ctrl = _finite_f32(labels["executed_ctrl_rad"], "executed_ctrl_rad", (HORIZON, NUM_CHANNELS))

        observation = {
            "screen_rgb": rgb,
            "q_rad16": q,
            "qvel_rad_s16": qvel,
            "previous_ctrl_rad16": prev_ctrl,
            **sensors,
        }
        target = {"target_ctrl_rad": target_ctrl}

        base_id = rec.get("episode_key") or f"episode-{int(rec['episode_id']):03d}"
        provenance = {
            "sample_id": f"{self.corpus_id}:{base_id}:frame-{int(rec['frame_id'])}",
            "corpus_id": self.corpus_id,
            "split": split,
            "episode_id": int(rec["episode_id"]),
            "frame_id": int(rec["frame_id"]),
            "task_profile": rec.get("task_profile", "near_pointing_v1"),
            "collection_id": rec.get("collection_id"),
            "index_path": str(self._index_path),
            "index_sha256": self._index_sha256,
            "png_ref": str(actor["png_ref"]),
            "png_path": str(png_path),
            "png_sha256": str(actor["png_sha256"]),
            "frame_monotonic_time_s": rec.get("metadata", {}).get("frame_monotonic_time_s"),
            "action_window": timing,
        }
        return VisualExample(observation=observation, target=target, provenance=provenance)

    def example(self, i: int) -> VisualExample:
        """Decode and return example ``i`` (global index); nothing is cached."""
        if isinstance(i, bool) or not isinstance(i, int):
            raise VisualCorpusError(f"example index must be an int, got {i!r}")
        if not 0 <= i < len(self._records):
            raise IndexError(f"example index {i} out of range for corpus of {len(self._records)}")
        return self._build(i)
