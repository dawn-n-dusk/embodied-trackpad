"""20 ms window normalization stats for the LEAP v1 pi0.5 window pipeline.

Pure NumPy fitting core plus a public corpus wrapper. ``_fit_window_arrays``
reduces already-selected train-window arrays to official-style
``mean``/``std``/``q01``/``q99`` vectors per group, matching the native
fitter's arithmetic (float64, population std ``ddof=0``, linear
``np.quantile`` at 0.01/0.99). Constant channels are reported as-is; no
variance is invented and nothing is clipped or normalized here.

``fit_pi05_window_norm_stats`` fits only eligible full TRAIN windows from one
or more validated ``VisualCorpus`` objects, reusing the public
``rows_from_corpus``/``assemble_windows`` path (index bytes are re-read and
re-hashed by that path and must bind to the same SHA). Each train window
contributes one state vector — the original JSON ``actor_input.q_rad16`` of
its start row in float64 — and its 25x16 ``coarse_ctrl_rad`` absolute joint
references; windows are weighted uniformly, so overlapping references are
intentionally repeated. Validation/test windows never contribute. No PNG is
decoded, ``example()``/``_records`` are never touched, and no disk path enters
the returned stats. Normalization application is a separate downstream module.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from embodied_trackpad.leap_v1.action_resampling import COARSE_DT_S, NATIVE_DT_S
from embodied_trackpad.leap_v1.temporal_windows import (
    WINDOW_REFS,
    TemporalWindowError,
    assemble_windows,
    rows_from_corpus,
)
from embodied_trackpad.leap_v1.visual_dataset import (
    HORIZON,
    NUM_CHANNELS,
    VisualCorpus,
    VisualCorpusError,
)

_FORMAT_VERSION = "leap-v1-pi05-window-norm-v1"
_ACTION_SEMANTICS = "absolute_executed_joint_reference_rad"
_HORIZON_S = WINDOW_REFS * COARSE_DT_S  # 25 * 0.02 = 0.5
_QUANTILE_LEVELS = (0.01, 0.99)
_STAT_KEYS = ("mean", "std", "q01", "q99")
_GROUPS = ("state", "actions")
_PROVENANCE_SERIALIZATION = (
    "sha256 of the UTF-8 bytes of json.dumps(records, sort_keys=True, "
    "separators=(',', ':')); each record is an object with keys corpus_id, "
    "index_sha256, episode_id, frame_id, first_step_index, row_indices, where "
    "row_indices is a JSON array of integers in assembler order; records are "
    "ordered by supplied corpus order, then assembler window order"
)
_STATE_NOTE = (
    "state fitted from the original JSON actor_input.q_rad16 as float64 "
    "(native-fitter arithmetic); live observations round q_rad16 to float32 "
    "later, so only representation rounding differs at apply time"
)
_METHOD = (
    "float64 mean, population std (ddof=0), linear np.quantile q01/q99 per "
    "group over uniformly weighted eligible train windows"
)


def _fit_window_arrays(
    state_rows: Any, action_windows: Any
) -> dict[str, dict[str, list[float]]]:
    """Fit per-channel stats from train-window state/actions arrays.

    ``state_rows`` is (N, 16) and ``action_windows`` is (N, 25, 16); actions
    are flattened across windows and time into (N*25, 16). Both are converted
    to float64 and must be finite. Returns JSON-serializable ``state`` and
    ``actions`` groups with exact keys ``mean``/``std``/``q01``/``q99``.
    """
    states = np.asarray(state_rows)
    if states.dtype.kind not in "iuf":
        raise VisualCorpusError(f"state_rows must be numeric, got {states.dtype}")
    states = np.array(states, dtype=np.float64, copy=True)
    if states.ndim != 2 or states.shape[1] != NUM_CHANNELS:
        raise VisualCorpusError(
            f"state_rows must be (N, {NUM_CHANNELS}), got {states.shape}"
        )
    actions = np.asarray(action_windows)
    if actions.dtype.kind not in "iuf":
        raise VisualCorpusError(f"action_windows must be numeric, got {actions.dtype}")
    actions = np.array(actions, dtype=np.float64, copy=True)
    if actions.ndim != 3 or actions.shape[1:] != (HORIZON, NUM_CHANNELS):
        raise VisualCorpusError(
            f"action_windows must be (N, {HORIZON}, {NUM_CHANNELS}), got {actions.shape}"
        )
    n = states.shape[0]
    if n == 0:
        raise VisualCorpusError("no train windows; refusing to fit empty stats")
    if actions.shape[0] != n:
        raise VisualCorpusError(
            f"state/action window count mismatch: {n} vs {actions.shape[0]}"
        )
    if not np.isfinite(states).all():
        raise VisualCorpusError("state_rows contains non-finite values")
    if not np.isfinite(actions).all():
        raise VisualCorpusError("action_windows contains non-finite values")

    flat_actions = actions.reshape(-1, NUM_CHANNELS)
    result: dict[str, dict[str, list[float]]] = {}
    for name, arr in (("state", states), ("actions", flat_actions)):
        q01, q99 = np.quantile(arr, _QUANTILE_LEVELS, axis=0)  # linear interpolation
        group = {
            "mean": np.mean(arr, axis=0),
            "std": np.std(arr, axis=0, ddof=0),  # population std
            "q01": q01,
            "q99": q99,
        }
        for key, vec in group.items():
            if not np.isfinite(vec).all():
                raise VisualCorpusError(f"{name} {key} is not finite")
            group[key] = [float(v) for v in vec]
        result[name] = group
    return result


def _parse_raw_records(data: bytes, corpus_id: str) -> list[Any]:
    """Parse nonblank JSONL index bytes into records, zero-based record order."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TemporalWindowError("invalid_shape", f"index is not UTF-8: {exc}") from exc
    records: list[Any] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise TemporalWindowError(
                "invalid_shape", f"corpus {corpus_id!r} index record {len(records)}: {exc}"
            ) from exc
    return records


def _checked_start_state(
    rec: Any, row: Any, prov: Any, corpus_id: str, sha: str
) -> np.ndarray:
    """Cross-check a train window start row, then return its raw JSON q (f64).

    The raw index record, the validated ``WindowRow``, and the window
    provenance must agree on corpus ID, index SHA, split, episode, frame and
    ``first_step_index`` before the state vector may be used.
    """
    idx = prov.row_indices[0]
    if row.row_index != idx or row.corpus_id != corpus_id or row.index_sha256 != sha:
        raise TemporalWindowError(
            "provenance_mismatch",
            f"train window start row {idx} does not bind to corpus {corpus_id!r}",
        )
    if prov.corpus_id != corpus_id or prov.index_sha256 != sha:
        raise TemporalWindowError(
            "provenance_mismatch",
            f"train window at row {idx} carries a different corpus/index binding",
        )
    action_window = rec.get("action_window") if isinstance(rec, dict) else None
    if (
        rec.get("split") != "train"
        or row.split != "train"
        or prov.split != "train"
        or rec.get("episode_id") != prov.episode_id
        or rec.get("episode_id") != row.episode_id
        or rec.get("frame_id") != prov.frame_id
        or rec.get("frame_id") != row.frame_id
        or not isinstance(action_window, dict)
        or action_window.get("first_step_index") != prov.first_step_index
        or action_window.get("first_step_index") != row.first_step_index
    ):
        raise TemporalWindowError(
            "provenance_mismatch",
            f"train window start row {idx}: raw record, row and window provenance disagree",
        )
    q = np.asarray((rec.get("actor_input") or {}).get("q_rad16"))
    if q.dtype.kind not in "iuf":
        raise TemporalWindowError(
            "invalid_units", f"row {idx}: q_rad16 must be a real numeric array, got {q.dtype}"
        )
    q64 = np.array(q, dtype=np.float64, copy=True)
    if q64.shape != (NUM_CHANNELS,) or not np.isfinite(q64).all():
        raise TemporalWindowError(
            "invalid_units", f"row {idx}: q_rad16 must be finite float64[{NUM_CHANNELS}]"
        )
    return q64


def fit_pi05_window_norm_stats(corpora: Sequence[VisualCorpus]) -> dict[str, Any]:
    """Fit 20 ms window NormStats from the TRAIN split of each corpus (f64).

    Each eligible full train window contributes its start-row raw JSON
    ``q_rad16`` once as the state and its 25x16 ``coarse_ctrl_rad`` as the
    actions; windows are uniformly weighted (overlapping references repeat).
    Index bytes are verified against ``corpus.index_sha256`` here and again by
    the public ``rows_from_corpus``/``assemble_windows`` path, which never
    bind a different SHA. Returns JSON-serializable ``state``/``actions``
    stats plus an explicit ``metadata`` block (format version, time contract,
    action semantics, weighting, corpus bindings, per-corpus train-window
    counts, totals, constant channels and the ordered train-window provenance
    digest). No disk paths are recorded and nothing is normalized or clipped.
    """
    if isinstance(corpora, (str, bytes)):
        raise VisualCorpusError("corpora must be a sequence of VisualCorpus, not a string")
    try:
        items = list(corpora)
    except TypeError as exc:
        raise VisualCorpusError("corpora must be a sequence of VisualCorpus") from exc
    if not items:
        raise VisualCorpusError("corpora is empty")

    seen_ids: set[str] = set()
    seen_shas: set[str] = set()
    corpus_ids: list[str] = []
    index_shas: list[str] = []
    train_counts: list[int] = []
    state_rows: list[np.ndarray] = []
    action_windows: list[np.ndarray] = []
    provenance_records: list[dict[str, Any]] = []

    for corpus in items:
        if not isinstance(corpus, VisualCorpus):
            raise VisualCorpusError(
                "corpora entries must be validated VisualCorpus instances, "
                f"got {type(corpus).__name__}"
            )
        corpus_id = corpus.corpus_id
        if corpus_id in seen_ids:
            raise VisualCorpusError(f"duplicate corpus_id {corpus_id!r} would double-weight windows")
        seen_ids.add(corpus_id)
        index_path = Path(corpus.index_path)
        try:
            data = index_path.read_bytes()
        except OSError as exc:
            raise TemporalWindowError("invalid_shape", f"cannot read index: {exc}") from exc
        sha = hashlib.sha256(data).hexdigest().lower()
        if sha != str(corpus.index_sha256).lower():
            raise TemporalWindowError(
                "index_sha256_mismatch",
                f"index hash {sha[:16]}... != corpus {corpus_id!r} recorded "
                f"{str(corpus.index_sha256)[:16]}...",
            )
        if sha in seen_shas:
            raise VisualCorpusError(
                f"duplicate index SHA-256 {sha[:16]}... across corpora "
                "(same windows would be weighted twice)"
            )
        seen_shas.add(sha)
        corpus_ids.append(corpus_id)
        index_shas.append(sha)

        records = _parse_raw_records(data, corpus_id)
        rows = rows_from_corpus(corpus)  # independently re-reads/binds the same SHA
        if len(records) != len(rows):
            raise TemporalWindowError(
                "index_count_mismatch",
                f"parsed {len(records)} raw records != {len(rows)} validated rows",
            )
        assembled = assemble_windows(rows)

        train_n = 0
        for window in assembled.windows:
            prov = window.provenance
            if prov.split != "train":
                continue
            start = prov.row_indices[0]
            q = _checked_start_state(records[start], rows[start], prov, corpus_id, sha)
            coarse = np.asarray(window.coarse_ctrl_rad)
            if coarse.dtype.kind not in "iuf" or coarse.shape != (HORIZON, NUM_CHANNELS):
                raise TemporalWindowError(
                    "invalid_shape",
                    f"train window at row {start}: coarse_ctrl_rad malformed",
                )
            state_rows.append(q)
            action_windows.append(np.array(coarse, dtype=np.float64, copy=True))
            train_n += 1
            provenance_records.append({
                "corpus_id": corpus_id,
                "index_sha256": sha,
                "episode_id": int(prov.episode_id),
                "frame_id": int(prov.frame_id),
                "first_step_index": int(prov.first_step_index),
                "row_indices": [int(i) for i in prov.row_indices],
            })
        train_counts.append(train_n)

    if not state_rows:
        raise VisualCorpusError(
            "no complete train windows across corpora; refusing to fit empty stats"
        )

    states = np.stack(state_rows)
    actions = np.stack(action_windows)
    stats = _fit_window_arrays(states, actions)

    constant_channels = {
        group: [i for i, (lo, hi) in enumerate(zip(stats[group]["q01"], stats[group]["q99"])) if lo == hi]
        for group in _GROUPS
    }
    payload = json.dumps(provenance_records, sort_keys=True, separators=(",", ":"))
    provenance_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    stats["metadata"] = {
        "format_version": _FORMAT_VERSION,
        "method": _METHOD,
        "fit_split": "train",
        "native_dt_s": NATIVE_DT_S,
        "target_dt_s": COARSE_DT_S,
        "horizon": int(WINDOW_REFS),
        "horizon_s": _HORIZON_S,
        "action_semantics": _ACTION_SEMANTICS,
        "window_weighting": "uniform_per_eligible_train_window (overlap intentionally repeated)",
        "state_note": _STATE_NOTE,
        "corpus_ids": corpus_ids,
        "index_sha256": index_shas,
        "train_windows_per_corpus": train_counts,
        "train_windows_total": int(states.shape[0]),
        "state_rows_total": int(states.shape[0]),
        "action_refs_total": int(actions.shape[0] * actions.shape[1]),
        "constant_channel_indices": constant_channels,
        "train_window_provenance_sha256": provenance_sha,
        "train_window_provenance_serialization": _PROVENANCE_SERIALIZATION,
    }
    return stats
