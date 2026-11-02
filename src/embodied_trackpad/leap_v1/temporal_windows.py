"""Temporal window assembly for LEAP v1 pi0.5 20 ms control references.

Assembly of already-audited native 25-step example rows into
0.5 s windows of 250 contiguous 2 ms native control intervals and 25 coarse
20 ms endpoint references by reusing the pure math in
:mod:`embodied_trackpad.leap_v1.action_resampling`. The assembly functions do not decode images. The separate observation
loader decodes only the current start-frame image. There is no normalization
fitting or physics-equivalence claim.

Time semantics (unchanged from the native corpus; see visual_dataset):

* native sim dt is exactly 0.002 s; one index row covers 25 native steps
  (0.05 s) of absolute 16-channel actuator-position commands in radians.
* The collector physics_step writes sim_time after mj_step, so native command
  ``i`` recorded at ``record_time`` is effective on
  ``[record_time - dt, record_time)``. Recorded native times are never shifted.
* A coarse 20 ms endpoint reference for segment ``k`` (0-based) sits at
  ``consumed_prev_sim_time_s + (k + 1) * COARSE_DT_S``; its 10 native rows are
  effective on the 10 preceding 2 ms intervals.
* The episode sim clock is relative to that episode's reset; capture clocks
  (``frame_monotonic_time_s``) are a separate domain. Ordering and contiguity
  use integer step-index provenance in index order only; same-clock derived
  time comparisons use rtol=0, atol=1e-9 (pure float arithmetic). The two
  clocks are never compared or mapped onto each other.

Integrity vs. insufficiency: malformed rows (bad dt/shape/units/non-finite or
mutated arrays) raise :class:`TemporalWindowError`. Sound rows that cannot
form a window (gaps, sim-time jumps, previous-ctrl mismatch or insufficient
suffix) are counted in the report, never bridged. A backward/reset step axis
within one group run raises nonmonotone_step and rejects the assembly.
Any non-identical step-range overlap raises ``overlap_conflict`` and aborts
the whole assembly: no window may use a conflicting region. Exact duplicates
merge only when times and identity agree, preserving every row id. One corpus
/index per assembly. Phase/goal/reset fields are honored when present; the
audited corpora carry none, reported as a limitation.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from embodied_trackpad.leap_v1 import small_policy
from embodied_trackpad.leap_v1.action_resampling import (
    CHANNELS,
    COARSE_DT_S,
    NATIVE_DT_S,
    STEPS_PER_COARSE,
    coarsen_native_controls,
    interpolate_coarse_controls,
)
from embodied_trackpad.leap_v1.visual_dataset import HORIZON, VisualCorpusError

FORMAT_VERSION = "leap_v1_temporal_window/v1"
WINDOW_ROWS = 10  # contiguous 25-step example rows per window
WINDOW_STEPS = WINDOW_ROWS * HORIZON  # 250 native 2 ms intervals (0.5 s)
WINDOW_REFS = WINDOW_STEPS // STEPS_PER_COARSE  # 25 coarse 20 ms endpoints
WINDOW_HORIZON_S = WINDOW_STEPS * NATIVE_DT_S  # 0.5
ACTION_SEMANTICS = "absolute_executed_actuator_command_rad"
_TIME_ATOL_S = 1e-9  # same-clock derived float comparisons only
# Raw per-row clock/capture whitelist metadata retained for provenance.
_CLOCK_META_KEYS = (
    "native_capture_clock_id",
    "simulation_clock_id",
    "clock_offset_s",
    "clock_uncertainty_s",
    "frame_age_ms",
    "worst_frame_age_ms",
)
_GROUP_FIELDS = ("corpus_id", "split", "episode_key", "episode_id",
                 "task_profile", "collection_id", "phase", "goal", "reset")
_SPLIT_NAMES = frozenset({"train", "validation", "test"})

__all__ = [
    "ACTION_SEMANTICS",
    "AssemblyGroupReport",
    "AssemblyReport",
    "AssemblyResult",
    "FORMAT_VERSION",
    "TemporalWindow",
    "TemporalWindowError",
    "WindowMetrics",
    "WindowObservation",
    "WindowProvenance",
    "WindowRow",
    "WINDOW_HORIZON_S",
    "WINDOW_REFS",
    "WINDOW_ROWS",
    "WINDOW_STEPS",
    "assemble_windows",
    "load_window_observation",
    "rows_from_corpus",
]


class TemporalWindowError(ValueError):
    """Integrity error; ``code`` is a stable documented rejection code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def _finite_f64(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype.kind not in "iuf":
        raise TemporalWindowError(
            "invalid_units", f"{name} must be a real numeric array, got {arr.dtype}"
        )
    out = np.array(arr, dtype=np.float64, copy=True)
    if out.shape != shape:
        raise TemporalWindowError(
            "invalid_shape", f"{name} shape {out.shape} != {shape}"
        )
    if not np.isfinite(out).all():
        raise TemporalWindowError(
            "invalid_units", f"{name} must be finite absolute radians"
        )
    out.setflags(write=False)  # immutable snapshot; assembly re-copies
    return out


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TemporalWindowError("invalid_shape", f"{name} must be int, got {value!r}")
    return value


def _as_time(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TemporalWindowError("invalid_units", f"{name} must be a real scalar")
    v = float(value)
    if not np.isfinite(v):
        raise TemporalWindowError("invalid_units", f"{name} must be finite")
    return v


def _same_time(a: float, b: float) -> bool:
    return abs(a - b) <= _TIME_ATOL_S


@dataclass(frozen=True)
class WindowRow:
    """Validated, immutable metadata+label view of one index row.

    ``phase``/``goal``/``reset`` are ``None`` when the index carries no such
    fields (true for the audited corpora); grouping then does not use them.
    Arrays are stored read-only; assembly copies them again defensively.
    Integer indices are validated as ints before any use (no coercion).
    """

    corpus_id: str
    row_index: int  # position in the sample index (ordering provenance)
    split: str
    episode_id: int
    frame_id: int
    first_step_index: int
    end_exclusive_step_index: int
    consumed_prev_step_index: int
    dt_s: float
    consumed_prev_sim_time_s: float
    first_sim_time_s: float
    end_sim_time_s: float
    previous_ctrl_rad: np.ndarray  # (16,) command at consumed_prev step
    executed_ctrl_rad: np.ndarray  # (25,16) absolute commands for the rows' steps
    index_path: str = ""
    index_sha256: str = ""
    episode_key: str | None = None
    task_profile: str | None = None
    collection_id: str | None = None
    phase: str | None = None
    goal: str | None = None
    reset: str | None = None
    frame_monotonic_time_s: float | None = None  # capture clock (own domain)
    capture_timing: Mapping[str, Any] | None = None  # raw capture/receive ages
    clock_meta: Mapping[str, Any] | None = None  # raw clock whitelist snapshot

    def __post_init__(self) -> None:
        object.__setattr__(self, "row_index", _as_int(self.row_index, "row_index"))
        if self.row_index < 0:
            raise TemporalWindowError("invalid_shape", "row_index must be >= 0")
        if not isinstance(self.corpus_id, str) or not self.corpus_id:
            raise TemporalWindowError("invalid_shape", "corpus_id must be a nonempty str")
        if not isinstance(self.split, str) or self.split not in _SPLIT_NAMES:
            raise TemporalWindowError(
                "invalid_split",
                f"split must be one of train/validation/test, got {self.split!r}",
            )
        for name in ("episode_id", "frame_id", "first_step_index",
                     "end_exclusive_step_index", "consumed_prev_step_index"):
            object.__setattr__(self, name, _as_int(getattr(self, name), name))
        if self.end_exclusive_step_index - self.first_step_index != HORIZON:
            raise TemporalWindowError(
                "invalid_shape",
                f"row must cover exactly {HORIZON} steps, got "
                f"{self.end_exclusive_step_index - self.first_step_index}",
            )
        if self.consumed_prev_step_index + 1 != self.first_step_index:
            raise TemporalWindowError(
                "invalid_shape",
                "consumed_prev_step_index + 1 must equal first_step_index",
            )
        dt = self.dt_s
        if isinstance(dt, bool) or not isinstance(dt, (int, float)):
            raise TemporalWindowError("invalid_dt", f"dt_s must be a real scalar, got {dt!r}")
        if float(dt) != NATIVE_DT_S:
            raise TemporalWindowError(
                "invalid_dt", f"dt_s must be exactly native {NATIVE_DT_S}, got {dt!r}"
            )
        object.__setattr__(self, "dt_s", float(dt))
        for name in ("consumed_prev_sim_time_s", "first_sim_time_s", "end_sim_time_s"):
            object.__setattr__(self, name, _as_time(getattr(self, name), name))
        if not _same_time(self.first_sim_time_s - self.consumed_prev_sim_time_s, NATIVE_DT_S):
            raise TemporalWindowError(
                "invalid_units",
                "first_sim_time_s - consumed_prev_sim_time_s must equal dt",
            )
        if abs((self.end_sim_time_s - self.first_sim_time_s) - (HORIZON - 1) * NATIVE_DT_S) > _TIME_ATOL_S:
            raise TemporalWindowError(
                "invalid_units", "end_sim_time_s - first_sim_time_s != (HORIZON-1)*dt"
            )
        if self.frame_monotonic_time_s is not None:
            object.__setattr__(self, "frame_monotonic_time_s",
                               _as_time(self.frame_monotonic_time_s, "frame_monotonic_time_s"))
        object.__setattr__(self, "previous_ctrl_rad",
                           _finite_f64(self.previous_ctrl_rad, "previous_ctrl_rad", (CHANNELS,)))
        object.__setattr__(self, "executed_ctrl_rad",
                           _finite_f64(self.executed_ctrl_rad, "executed_ctrl_rad", (HORIZON, CHANNELS)))

    @property
    def group_key(self) -> tuple[Any, ...]:
        return tuple(getattr(self, n) for n in _GROUP_FIELDS)

    def same_step_range(self, other: "WindowRow") -> bool:
        return (self.first_step_index == other.first_step_index
                and self.end_exclusive_step_index == other.end_exclusive_step_index)


@dataclass(frozen=True)
class WindowMetrics:
    """Per-channel pure numerical metrics for one window (no physics claim)."""

    recon_rms_rad: np.ndarray  # (16,) RMS of (native - linear reconstruction)
    recon_max_abs_rad: np.ndarray  # (16,) max abs of the same mismatch
    native_jump_rad: np.ndarray  # (16,) max |diff| along prev + 250 native rows
    coarse_jump_rad: np.ndarray  # (16,) max |diff| along prev + 25 coarse refs
    constant_channels: np.ndarray  # (16,) bool, native_jump == 0 exactly


@dataclass(frozen=True)
class WindowProvenance:
    format_version: str
    action_semantics: str
    corpus_id: str
    index_path: str
    index_sha256: str
    row_indices: tuple[int, ...]  # all merged exact-duplicate rows preserved
    episode_id: int
    frame_id: int  # observation frame (window start row)
    split: str
    episode_key: str | None
    task_profile: str | None
    collection_id: str | None
    phase: str | None
    goal: str | None
    reset: str | None
    first_step_index: int
    end_exclusive_step_index: int
    consumed_prev_step_index: int
    consumed_prev_sim_time_s: float
    first_sim_time_s: float
    end_sim_time_s: float
    row_window_times_s: tuple[tuple[float, float], ...]  # per source row (first, end)
    frame_monotonic_time_s: float | None = None  # start-row capture clock
    capture_timing: Mapping[str, Any] | None = None  # start-row capture ages
    clock_meta: Mapping[str, Any] | None = None  # start-row clock whitelist


@dataclass(frozen=True)
class TemporalWindow:
    """One assembled 0.5 s label window; observation is fetched lazily."""

    native_ctrl_rad: np.ndarray  # (250,16) float64 fresh writable copy
    previous_ctrl_rad: np.ndarray  # (16,) prior command (segment base)
    coarse_ctrl_rad: np.ndarray  # (25,16) 20 ms endpoint references
    coarse_endpoint_sim_time_s: np.ndarray  # (25,) consumed_prev + (k+1)*20ms
    metrics: WindowMetrics
    provenance: WindowProvenance


@dataclass(frozen=True)
class WindowObservation:
    """Current observation + coarse-window target; distinct from VisualExample."""

    observation: Mapping[str, Any]  # native whitelist incl. absent-sensor masks
    target: Mapping[str, Any]  # coarse 25x16 refs with explicit 20 ms timing
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class AssemblyGroupReport:
    group: tuple[Any, ...]
    rows: int
    segments: int
    windows: int
    unique_covered_steps: int
    rejections: Mapping[str, int]


@dataclass(frozen=True)
class AssemblyReport:
    format_version: str
    rows_total: int
    groups_total: int
    windows_total: int
    rejections: Mapping[str, int]  # code -> count across all groups
    groups: tuple[AssemblyGroupReport, ...] = ()
    phase_goal_reset_present: bool = False
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class AssemblyResult:
    windows: tuple[TemporalWindow, ...]
    report: AssemblyReport


def _parse_rows(data: bytes, corpus_id: str, index_path: str, index_sha256: str) -> list[WindowRow]:
    """Parse raw index bytes into rows; ``row_index`` follows decoded records."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TemporalWindowError("invalid_shape", f"index is not UTF-8: {exc}") from exc
    rows: list[WindowRow] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TemporalWindowError("invalid_shape", f"index row {len(rows)}: {exc}") from exc
        rows.append(_row_from_record(rec, len(rows), corpus_id, index_path, index_sha256))
    return rows


def rows_from_corpus(corpus: Any) -> list[WindowRow]:
    """Build validated rows from a VisualCorpus via its public API only.

    The index bytes are read exactly once, hashed, and bound into every row
    (``index_path``/``index_sha256``) against ``corpus.index_sha256`` (no
    TOCTOU re-read); the decoded count must equal ``len(corpus)``. Labels come
    from the raw rows so no PNG is decoded and the private ``_records`` is
    never touched. Original delta label fields are validated in place (never
    fabricated). Per-row capture-clock metadata (monotonic time, capture
    timing/ages) is retained for provenance; no teacher configuration is read.
    """
    index_path = Path(corpus.index_path)
    try:
        data = index_path.read_bytes()
    except OSError as exc:
        raise TemporalWindowError("invalid_shape", f"cannot read index: {exc}") from exc
    actual_sha = hashlib.sha256(data).hexdigest()
    if actual_sha.lower() != str(corpus.index_sha256).lower():
        raise TemporalWindowError(
            "index_sha256_mismatch",
            f"index hash {actual_sha[:16]}... != corpus {str(corpus.index_sha256)[:16]}...",
        )
    rows = _parse_rows(data, str(corpus.corpus_id), str(index_path), actual_sha.lower())
    if len(rows) != len(corpus):
        raise TemporalWindowError(
            "index_count_mismatch", f"parsed {len(rows)} rows != corpus length {len(corpus)}"
        )
    return rows


def _row_from_record(
    rec: Mapping[str, Any], row_index: int, corpus_id: str, index_path: str, index_sha256: str
) -> WindowRow:
    if not isinstance(rec, Mapping):
        raise TemporalWindowError("invalid_shape", f"row {row_index}: not an object")
    # Validate ORIGINAL label/delta fields and native window contract in place.
    try:
        small_policy.extract_action_window(rec)
        small_policy.extract_label_delta(rec)
    except small_policy.SmallPolicyError as exc:
        raise TemporalWindowError("invalid_units", f"row {row_index}: {exc}") from exc
    try:
        aw = rec["action_window"]
        actor = rec["actor_input"]
        labels = rec["labels"]
        metadata = rec.get("metadata") or {}
        return WindowRow(
            corpus_id=corpus_id,
            row_index=row_index,
            split=rec["split"],
            episode_id=rec["episode_id"],
            frame_id=rec["frame_id"],
            first_step_index=aw["first_step_index"],
            end_exclusive_step_index=aw["end_exclusive_step_index"],
            consumed_prev_step_index=aw["consumed_prev_step_index"],
            dt_s=aw["dt_s"],
            consumed_prev_sim_time_s=aw["consumed_prev_sim_time_s"],
            first_sim_time_s=aw["first_sim_time_s"],
            end_sim_time_s=aw["end_sim_time_s"],
            previous_ctrl_rad=actor["previous_ctrl_rad16"],
            executed_ctrl_rad=labels["executed_ctrl_rad"],
            index_path=index_path,
            index_sha256=index_sha256,
            episode_key=(rec["episode_key"] if rec.get("episode_key") is not None else None),
            task_profile=(rec["task_profile"] if rec.get("task_profile") is not None else None),
            collection_id=(rec["collection_id"] if rec.get("collection_id") is not None else None),
            phase=_opt_str(metadata, "phase") or _opt_str(rec, "phase"),
            goal=_opt_str(metadata, "goal") or _opt_str(rec, "goal"),
            reset=_opt_str(metadata, "reset") or _opt_str(rec, "reset"),
            frame_monotonic_time_s=metadata.get("frame_monotonic_time_s"),
            capture_timing=deepcopy(metadata.get("capture_timing")),
            clock_meta={k: deepcopy(metadata[k]) for k in _CLOCK_META_KEYS if k in metadata},
        )
    except TemporalWindowError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise TemporalWindowError("invalid_shape", f"row {row_index}: {exc}") from exc


def _opt_str(container: Any, key: str) -> str | None:
    if isinstance(container, Mapping) and container.get(key) is not None:
        return str(container[key])
    return None


def _try_merge_identical(a: WindowRow, b: WindowRow) -> WindowRow | None:
    """Merge an exact duplicate, preserving all provenance (3+ duplicates OK).

    Requires identical step range, identical sim times, identical identity
    fields, identical previous/executed controls (previous-ctrl continuity).
    Returns ``None`` when anything differs (then the overlap is a conflict).
    """
    if not a.same_step_range(b):
        return None
    if not (_same_time(a.consumed_prev_sim_time_s, b.consumed_prev_sim_time_s)
            and _same_time(a.first_sim_time_s, b.first_sim_time_s)
            and _same_time(a.end_sim_time_s, b.end_sim_time_s)):
        return None
    if not (np.array_equal(a.previous_ctrl_rad, b.previous_ctrl_rad)
            and np.array_equal(a.executed_ctrl_rad, b.executed_ctrl_rad)):
        return None
    if a.split != b.split or a.episode_id != b.episode_id or a.frame_id != b.frame_id:
        return None
    merged = WindowRow(
        corpus_id=a.corpus_id, row_index=a.row_index, split=a.split,
        episode_id=a.episode_id, frame_id=a.frame_id,
        first_step_index=a.first_step_index,
        end_exclusive_step_index=a.end_exclusive_step_index,
        consumed_prev_step_index=a.consumed_prev_step_index, dt_s=a.dt_s,
        consumed_prev_sim_time_s=a.consumed_prev_sim_time_s,
        first_sim_time_s=a.first_sim_time_s, end_sim_time_s=a.end_sim_time_s,
        previous_ctrl_rad=a.previous_ctrl_rad, executed_ctrl_rad=a.executed_ctrl_rad,
        index_path=a.index_path, index_sha256=a.index_sha256,
        episode_key=a.episode_key, task_profile=a.task_profile,
        collection_id=a.collection_id, phase=a.phase, goal=a.goal, reset=a.reset,
        frame_monotonic_time_s=a.frame_monotonic_time_s, capture_timing=a.capture_timing,
        clock_meta=a.clock_meta,
    )
    prior = getattr(a, "_merged_row_indices", (a.row_index,))
    object.__setattr__(merged, "_merged_row_indices", tuple(prior) + (b.row_index,))
    return merged


def _merged_indices(row: WindowRow) -> tuple[int, ...]:
    return tuple(getattr(row, "_merged_row_indices", (row.row_index,)))


def _build_window(rows: Sequence[WindowRow]) -> TemporalWindow:
    first, last = rows[0], rows[-1]
    # Defensive fresh copies: input arrays are read-only snapshots and must
    # never alias the returned writable window arrays (mutation isolation).
    native = np.concatenate(
        [np.array(np.asarray(r.executed_ctrl_rad), dtype=np.float64, copy=True) for r in rows],
        axis=0,
    )
    prev = np.array(np.asarray(first.previous_ctrl_rad), dtype=np.float64, copy=True)
    coarse = coarsen_native_controls(native, prev)
    reconstructed = interpolate_coarse_controls(coarse.coarse, prev)
    mismatch = native - reconstructed
    native_chain = np.concatenate([prev[None, :], native], axis=0)
    coarse_chain = np.concatenate([prev[None, :], coarse.coarse], axis=0)
    native_jump = np.max(np.abs(np.diff(native_chain, axis=0)), axis=0)
    metrics = WindowMetrics(
        recon_rms_rad=np.sqrt(np.mean(mismatch * mismatch, axis=0)),
        recon_max_abs_rad=np.max(np.abs(mismatch), axis=0),
        native_jump_rad=native_jump,
        coarse_jump_rad=np.max(np.abs(np.diff(coarse_chain, axis=0)), axis=0),
        constant_channels=native_jump == 0.0,
    )
    endpoints = first.consumed_prev_sim_time_s + (
        np.arange(1, WINDOW_REFS + 1, dtype=np.float64) * COARSE_DT_S
    )
    prov = WindowProvenance(
        format_version=FORMAT_VERSION,
        action_semantics=ACTION_SEMANTICS,
        corpus_id=first.corpus_id,
        index_path=first.index_path,
        index_sha256=first.index_sha256,
        row_indices=tuple(i for r in rows for i in _merged_indices(r)),
        episode_id=first.episode_id,
        frame_id=first.frame_id,
        split=first.split,
        episode_key=first.episode_key,
        task_profile=first.task_profile,
        collection_id=first.collection_id,
        phase=first.phase,
        goal=first.goal,
        reset=first.reset,
        first_step_index=first.first_step_index,
        end_exclusive_step_index=last.end_exclusive_step_index,
        consumed_prev_step_index=first.consumed_prev_step_index,
        consumed_prev_sim_time_s=first.consumed_prev_sim_time_s,
        first_sim_time_s=first.first_sim_time_s,
        end_sim_time_s=last.end_sim_time_s,
        row_window_times_s=tuple((r.first_sim_time_s, r.end_sim_time_s) for r in rows),
        frame_monotonic_time_s=first.frame_monotonic_time_s,
        capture_timing=first.capture_timing,
        clock_meta=first.clock_meta,
    )
    return TemporalWindow(
        native_ctrl_rad=native,
        previous_ctrl_rad=prev,
        coarse_ctrl_rad=np.array(coarse.coarse, dtype=np.float64, copy=True),
        coarse_endpoint_sim_time_s=endpoints,
        metrics=metrics,
        provenance=prov,
    )


def assemble_windows(rows: Sequence[WindowRow]) -> AssemblyResult:
    """Assemble 0.5 s windows from validated rows in index order.

    Row order is provenance and is never re-sorted. Rows are processed in
    contiguous index-order runs of equal group key (corpus/episode/split/
    profile, plus phase/goal/reset when present); recurring groups across an
    intervening boundary never reconnect. Within a run, a nonmonotone step
    axis (reset), gap, sim-time jump, step-link break or previous-ctrl
    mismatch cuts a segment; breaks are counted, never bridged. Adjacent rows
    must agree on the shared boundary within ``_TIME_ATOL_S`` and on the
    previous control exactly. Any non-identical overlap raises
    ``overlap_conflict`` and aborts the whole assembly (fail closed). One
    corpus/index per assembly.
    """
    rows = list(rows)
    for i, r in enumerate(rows):
        if not isinstance(r, WindowRow):
            raise TemporalWindowError("invalid_shape", f"rows[{i}] is not a WindowRow")
        # Re-validate snapshot integrity (guards against external mutation).
        _finite_f64(r.previous_ctrl_rad, "previous_ctrl_rad", (CHANNELS,))
        _finite_f64(r.executed_ctrl_rad, "executed_ctrl_rad", (HORIZON, CHANNELS))
    for a, b in zip(rows, rows[1:]):
        if b.row_index <= a.row_index:
            raise TemporalWindowError(
                "nonmonotone_row_order",
                f"row_index {b.row_index} follows {a.row_index}; keep index order",
            )
    corpus_ids = {r.corpus_id for r in rows}
    if len(corpus_ids) > 1:
        raise TemporalWindowError(
            "mixed_corpus", f"one corpus per assembly, got {sorted(corpus_ids)}"
        )
    bindings = {(r.index_path, r.index_sha256) for r in rows}
    if len(bindings) > 1:
        raise TemporalWindowError(
            "mixed_corpus", "rows carry mixed index_path/index_sha256 binding"
        )

    # Contiguous runs of equal group key in index order (no dict regrouping).
    runs: list[tuple[tuple[Any, ...], list[WindowRow]]] = []
    for r in rows:
        if runs and runs[-1][0] == r.group_key:
            runs[-1][1].append(r)
        else:
            runs.append((r.group_key, [r]))

    counts: dict[str, int] = {}
    windows: list[TemporalWindow] = []
    group_reports: list[AssemblyGroupReport] = []
    present = any(r.phase is not None or r.goal is not None or r.reset is not None for r in rows)

    def bump(bucket: dict[str, int], code: str, n: int = 1) -> None:
        bucket[code] = bucket.get(code, 0) + n

    for key, group_rows in runs:
        g_counts: dict[str, int] = {}
        segments: list[list[WindowRow]] = []
        cur: list[WindowRow] = []
        for row in group_rows:
            if not cur:
                cur = [row]
                continue
            anchor = cur[-1]
            if row.same_step_range(anchor):
                merged = _try_merge_identical(anchor, row)
                if merged is not None:
                    cur[-1] = merged
                    bump(g_counts, "identical_overlap_merged")
                    continue
                raise TemporalWindowError(
                    "overlap_conflict",
                    f"rows {anchor.row_index} and {row.row_index} claim step range "
                    f"[{anchor.first_step_index}, {anchor.end_exclusive_step_index}) "
                    "with non-identical content; refusing the whole assembly",
                )
            if row.first_step_index <= anchor.first_step_index:
                raise TemporalWindowError(
                    "nonmonotone_step",
                    f"row {row.row_index} step axis goes backward/resets after row "
                    f"{anchor.row_index} ({row.first_step_index} <= "
                    f"{anchor.first_step_index}); refusing the whole assembly",
                )
            if row.first_step_index < anchor.end_exclusive_step_index:
                raise TemporalWindowError(
                    "overlap_conflict",
                    f"rows {anchor.row_index} and {row.row_index} overlap on "
                    f"[{row.first_step_index}, {anchor.end_exclusive_step_index}); "
                    "refusing the whole assembly",
                )
            if row.first_step_index > anchor.end_exclusive_step_index:
                bump(g_counts, "gap")
                if not _same_time(row.consumed_prev_sim_time_s, anchor.end_sim_time_s):
                    bump(g_counts, "time_jump")
                segments.append(cur)
                cur = [row]
                continue
            # Exact boundary continuity (row.first == anchor.end).
            if row.consumed_prev_step_index != anchor.end_exclusive_step_index - 1:
                bump(g_counts, "boundary_step")
                segments.append(cur)
                cur = [row]
                continue
            if not _same_time(row.consumed_prev_sim_time_s, anchor.end_sim_time_s):
                bump(g_counts, "time_jump")
                segments.append(cur)
                cur = [row]
                continue
            if not np.array_equal(row.previous_ctrl_rad, anchor.executed_ctrl_rad[-1]):
                bump(g_counts, "prev_ctrl_mismatch")
                segments.append(cur)
                cur = [row]
                continue
            cur.append(row)
        if cur:
            segments.append(cur)

        g_windows = 0
        covered: set[int] = set()
        for seg in segments:
            n = len(seg)
            for start in range(n):
                if start + WINDOW_ROWS > n:
                    bump(g_counts, "insufficient_suffix")
                    continue
                chunk = seg[start:start + WINDOW_ROWS]
                windows.append(_build_window(chunk))
                g_windows += 1
                covered.update(range(chunk[0].first_step_index, chunk[-1].end_exclusive_step_index))
        for code, n in g_counts.items():
            bump(counts, code, n)
        group_reports.append(AssemblyGroupReport(
            group=key, rows=len(group_rows), segments=len(segments),
            windows=g_windows, unique_covered_steps=len(covered),
            rejections=dict(g_counts),
        ))

    limitations: list[str] = []
    if rows and not present:
        limitations.append(
            "phase/goal/reset fields are absent from the index; windows may "
            "span unannotated phase or goal changes (none were invented)."
        )
    report = AssemblyReport(
        format_version=FORMAT_VERSION,
        rows_total=len(rows),
        groups_total=len(runs),
        windows_total=len(windows),
        rejections=counts,
        groups=tuple(group_reports),
        phase_goal_reset_present=present,
        limitations=tuple(limitations),
    )
    return AssemblyResult(windows=tuple(windows), report=report)


def load_window_observation(corpus: Any, window: TemporalWindow) -> WindowObservation:
    """Fetch exactly one current observation for a window (no future images).

    Uses only the public ``VisualCorpus.example`` API on the window's start
    row. Corpus identity/hash and the decoded example's corpus_id,
    episode_id, frame_id, split, action-window start step and start times are
    re-verified against the window; any mismatch raises. The previous control
    is compared at VisualCorpus's documented float32 observation precision
    (the f64 native labels stay unchanged). The returned
    :class:`WindowObservation` preserves the native observation whitelist
    (including absent-sensor masks and the frame's own capture-clock
    provenance, with no cross-clock map) but replaces the target with the 25
    coarse 20 ms references and explicit endpoint timing.
    """
    prov = window.provenance
    if str(getattr(corpus, "corpus_id", "")) != prov.corpus_id:
        raise TemporalWindowError(
            "mixed_corpus", f"corpus_id {getattr(corpus, 'corpus_id', None)!r} != {prov.corpus_id!r}"
        )
    if str(getattr(corpus, "index_sha256", prov.index_sha256)).lower() != prov.index_sha256.lower():
        raise TemporalWindowError(
            "index_sha256_mismatch", "corpus index hash does not bind to window provenance"
        )
    start_row = prov.row_indices[0]
    try:
        example = corpus.example(start_row)
    except VisualCorpusError as exc:
        raise TemporalWindowError("future_observation", str(exc)) from exc
    ex_prov = dict(getattr(example, "provenance", {}) or {})
    ex_obs = dict(getattr(example, "observation", {}) or {})
    if str(ex_prov.get("corpus_id", "")) != prov.corpus_id:
        raise TemporalWindowError("future_observation", "example corpus_id mismatch")
    if str(ex_prov.get("index_sha256", "")).lower() != prov.index_sha256.lower():
        raise TemporalWindowError("future_observation", "example index hash mismatch")
    if int(ex_prov.get("episode_id", -1)) != prov.episode_id:
        raise TemporalWindowError("future_observation", "example episode mismatch")
    if int(ex_prov.get("frame_id", -1)) != prov.frame_id:
        raise TemporalWindowError(
            "future_observation", "example frame_id is not the window start frame"
        )
    if str(ex_prov.get("split", "")) != prov.split:
        raise TemporalWindowError("future_observation", "example split mismatch")
    aw = ex_prov.get("action_window") or {}
    if not (isinstance(aw.get("first_step_index"), int)
            and aw["first_step_index"] == prov.first_step_index):
        raise TemporalWindowError("future_observation", "example is not the window start row")
    for name, want in (("first_sim_time_s", prov.first_sim_time_s),
                       ("consumed_prev_sim_time_s", prov.consumed_prev_sim_time_s)):
        got = aw.get(name)
        if got is None or not _same_time(float(got), want):
            raise TemporalWindowError(
                "future_observation", f"example {name} mismatch; refusing observation"
            )
    prev = np.asarray(ex_obs.get("previous_ctrl_rad16"), dtype=np.float64)
    prev_f32 = np.asarray(window.previous_ctrl_rad, dtype=np.float32).astype(np.float64)
    if prev.shape != (CHANNELS,) or not np.array_equal(prev, prev_f32):
        raise TemporalWindowError(
            "future_observation", "example previous_ctrl does not match window base"
        )
    observation = {k: ex_obs[k] for k in ex_obs if k in (
        "screen_rgb", "q_rad16", "qvel_rad_s16", "previous_ctrl_rad16",
        "fsr_N4", "motor_current_A16", "wrist",
        "fsr_valid4", "motor_current_valid16", "wrist_valid",
    )}
    target = {
        "target_ctrl_rad": np.array(window.coarse_ctrl_rad, copy=True),
        "dt_s": COARSE_DT_S,
        "horizon": WINDOW_REFS,
        "horizon_s": WINDOW_REFS * COARSE_DT_S,
        "coarse_endpoint_sim_time_s": np.array(window.coarse_endpoint_sim_time_s, copy=True),
        "action_semantics": ACTION_SEMANTICS,
    }
    provenance = {
        "format_version": FORMAT_VERSION,
        "sample_id": ex_prov.get("sample_id"),
        "corpus_id": prov.corpus_id,
        "index_sha256": prov.index_sha256,
        "row_indices": prov.row_indices,
        "episode_id": prov.episode_id,
        "frame_id": prov.frame_id,
        "split": prov.split,
        "task_profile": prov.task_profile,
        "collection_id": prov.collection_id,
        "phase": prov.phase,
        "goal": prov.goal,
        "reset": prov.reset,
        "frame_monotonic_time_s": (ex_prov.get("frame_monotonic_time_s")
                                   if ex_prov.get("frame_monotonic_time_s") is not None
                                   else prov.frame_monotonic_time_s),
        "capture_timing": prov.capture_timing,
        "clock_meta": deepcopy(prov.clock_meta),
        "action_window": aw,
        "prev_action_semantics": "absolute_executed_joint_reference_rad",
    }
    return WindowObservation(observation=observation, target=target, provenance=provenance)
