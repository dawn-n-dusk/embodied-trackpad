"""Train-split normalization stats for the native LEAP v1 pi0.5 mainline.

Pure NumPy, no OpenPI/Torch/JAX imports. ``fit_pi05_norm_stats`` fits
official-style NormStats (exact keys ``mean``/``std``/``q01``/``q99``, one
16-dim float vector each) from the TRAIN split of one or more already
validated ``VisualCorpus`` objects only, and ``normalize_pi05_native_sample``
applies the exact official OpenPI (rev 215abfb) ``transforms.Normalize``
linear quantile map with ``use_quantiles=True``:

    y = (x - q01) / (q99 - q01 + 1e-6) * 2 - 1

to the raw native ``state`` [16] and ``actions`` [25, 16] radian arrays.

Fitting contract:
- Each corpus contributes only rows whose ``split == "train"``: ``q_rad16``
  once per train row, ``executed_ctrl_rad`` over all 25 fine steps per train
  row. No PNG is decoded; validation/test rows never contribute.
- Every index is re-hashed whole before and after ``small_policy.load_sample_index``
  and is loaded through that existing validator; index files are never
  concatenated and no validator is relaxed. Provenance namespace is
  preserved: duplicate ``corpus_id`` or duplicate index SHA-256 across the
  sequence is rejected so one corpus can never be weighted twice.
- Stats are fitted in float64 with population std (``ddof=0``) and linear
  ``np.quantile`` at 0.01/0.99; all returned values must be finite.

Normalization contract:
- No clipping, deltas, padding, resize, tokenization or loss application.
  Zero-variance channels stay finite through the official 1e-6 epsilon; the
  upstream formula is intentionally left unchanged (no invented variance).
- The exact native input contract is re-checked first: loss_mask32 must be
  TRUE on the first 16 entries and FALSE on the rest; image_mask must be
  base TRUE with both wrist views FALSE and all-zero; and
  ``pending_transforms['normalization']`` must be exactly ``"PENDING"`` so a
  sample is never normalized twice.
- RGB views, image/loss masks, prompt, unused-sensor provenance and the
  loss mask are preserved in independent copies;
  ``provenance["pending_transforms"]["normalization"]`` is set to
  ``"APPLIED"`` with a stats identity while every other pending flag is
  retained. Out-of-range counts and constant / sub-epsilon-range channels are
  recorded per group in provenance for reporting only; values are never
  clipped or refit.
- Fit metadata (``corpus_ids`` / ``index_sha256`` / per-corpus train-row
  counts / totals / method) is REQUIRED for normalization: absent, empty, or
  malformed metadata is rejected. The sample's corpus_id and index SHA-256
  must match the fitted lists, and the index SHA-256s ride in the
  normalization identity so stats are never applied unbound.

Saving is the caller's job: these functions only return plain dicts.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from embodied_trackpad.leap_v1 import small_policy
from embodied_trackpad.leap_v1.pi05_inputs import ACTION_DIM, Pi05NativeSample
from embodied_trackpad.leap_v1.visual_dataset import (
    HORIZON,
    NUM_CHANNELS,
    PNG_HEIGHT,
    PNG_WIDTH,
    VisualCorpus,
    VisualCorpusError,
)

_EPSILON = 1e-6  # official transforms.Normalize(use_quantiles=True) epsilon
_QUANTILE_LEVELS = (0.01, 0.99)
_METHOD = "openpi_215abfb_transforms.Normalize(use_quantiles=True)"
_STAT_KEYS = ("mean", "std", "q01", "q99")
_GROUPS = ("state", "actions")
_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_MODEL_DATA_KEYS = {"state", "image", "image_mask", "actions", "prompt"}


def fit_pi05_norm_stats(corpora: Sequence[VisualCorpus]) -> dict[str, Any]:
    """Fit 16-dim NormStats from the train split of each corpus (float64).

    Returns a JSON-serializable dict with exact official keys ``mean``,
    ``std``, ``q01``, ``q99`` under ``"state"`` and ``"actions"`` plus a
    ``"metadata"`` block (method, corpus_ids, per-corpus index SHA-256,
    train/state/action-step totals). The caller saves it if needed.
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
    action_rows: list[np.ndarray] = []

    for corpus in items:
        if not isinstance(corpus, VisualCorpus):
            raise VisualCorpusError(
                "corpora entries must be validated VisualCorpus instances, "
                f"got {type(corpus).__name__}"
            )
        corpus_id = corpus.corpus_id
        if corpus_id in seen_ids:
            raise VisualCorpusError(f"duplicate corpus_id {corpus_id!r} would double-weight samples")
        seen_ids.add(corpus_id)

        index_path = Path(corpus.index_path)
        sha_before = small_policy.sha256_file(index_path).lower()
        if sha_before != corpus.index_sha256.lower():
            raise VisualCorpusError(
                f"index re-hash mismatch for corpus {corpus_id!r}: file no longer matches "
                "the SHA recorded at corpus validation"
            )
        # Whole-index structural/split/feature validation via the existing
        # validator; indexes are loaded one at a time, never concatenated.
        records = small_policy.load_sample_index(index_path)
        sha_after = small_policy.sha256_file(index_path).lower()
        if sha_after != sha_before:
            raise VisualCorpusError(
                f"index {index_path} changed while fitting stats for corpus {corpus_id!r}"
            )
        if sha_after in seen_shas:
            raise VisualCorpusError(
                f"duplicate index SHA-256 {sha_after[:16]}... across corpora "
                "(same samples would be weighted twice)"
            )
        seen_shas.add(sha_after)
        corpus_ids.append(corpus_id)
        index_shas.append(sha_after)

        train_n = 0
        for rec in records:
            if rec.get("split") != "train":
                continue
            q = np.asarray(rec["actor_input"]["q_rad16"], dtype=np.float64)
            if q.shape != (NUM_CHANNELS,) or not np.isfinite(q).all():
                raise VisualCorpusError(f"corpus {corpus_id!r}: train q_rad16 malformed")
            executed = np.asarray(rec["labels"]["executed_ctrl_rad"], dtype=np.float64)
            if executed.shape != (HORIZON, NUM_CHANNELS) or not np.isfinite(executed).all():
                raise VisualCorpusError(f"corpus {corpus_id!r}: train executed_ctrl_rad malformed")
            state_rows.append(q)
            action_rows.append(executed.reshape(-1, NUM_CHANNELS))
            train_n += 1
        train_counts.append(train_n)

    if not state_rows:
        raise VisualCorpusError("no train rows across corpora; refusing to fit empty stats")

    state_fit = np.stack(state_rows)  # (train_rows, 16) float64
    actions_fit = np.concatenate(action_rows)  # (train_rows * 25, 16) float64

    result: dict[str, Any] = {}
    for name, arr in (("state", state_fit), ("actions", actions_fit)):
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

    result["metadata"] = {
        "method": _METHOD,
        "corpus_ids": corpus_ids,
        "index_sha256": index_shas,
        "train_rows_per_corpus": train_counts,
        "train_rows_total": len(state_rows),
        "state_rows_total": int(state_fit.shape[0]),
        "action_steps_total": int(actions_fit.shape[0]),
    }
    return result


def _validated_stats(stats: Mapping[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    """Return float64[16] mean/std/q01/q99 per group after contract checks."""
    if not isinstance(stats, Mapping):
        raise VisualCorpusError("stats must be a mapping with 'state' and 'actions' groups")
    out: dict[str, dict[str, np.ndarray]] = {}
    for group in _GROUPS:
        raw = stats.get(group)
        if not isinstance(raw, Mapping):
            raise VisualCorpusError(f"stats missing group {group!r}")
        vecs: dict[str, np.ndarray] = {}
        for key in _STAT_KEYS:
            if key not in raw:
                raise VisualCorpusError(f"stats[{group!r}] missing key {key!r}")
            try:
                arr = np.asarray(raw[key], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise VisualCorpusError(f"stats[{group!r}][{key!r}] is not numeric: {exc}") from exc
            if arr.shape != (NUM_CHANNELS,):
                raise VisualCorpusError(
                    f"stats[{group!r}][{key!r}] must be float64[16], got shape {arr.shape}"
                )
            if not np.isfinite(arr).all():
                raise VisualCorpusError(f"stats[{group!r}][{key!r}] contains non-finite values")
            vecs[key] = arr
        if np.any(vecs["q99"] < vecs["q01"]):
            raise VisualCorpusError(f"stats[{group!r}] has q99 < q01")
        if np.any(vecs["std"] < 0.0):
            raise VisualCorpusError(f"stats[{group!r}] has negative std")
        out[group] = vecs
    return out


def _validate_native_sample(sample: Any) -> None:
    """Re-check the native input contract for the fields this transform uses."""
    if not isinstance(sample, Pi05NativeSample):
        raise VisualCorpusError(f"sample must be a Pi05NativeSample, got {type(sample).__name__}")
    raw = sample.raw_model_data
    if not isinstance(raw, dict) or set(raw) != _MODEL_DATA_KEYS:
        raise VisualCorpusError("raw_model_data keys do not match the native pi0.5 contract")
    for name, shape in (("state", (NUM_CHANNELS,)), ("actions", (HORIZON, NUM_CHANNELS))):
        arr = np.asarray(raw[name])
        if arr.shape != shape or arr.dtype != np.float32:
            raise VisualCorpusError(
                f"raw_model_data[{name!r}] must be float32{shape}, got {arr.dtype}{tuple(arr.shape)}"
            )
        if not np.isfinite(arr).all():
            raise VisualCorpusError(f"raw_model_data[{name!r}] contains non-finite values")
    image = raw["image"]
    image_mask = raw["image_mask"]
    if not isinstance(image, dict) or set(image) != set(_IMAGE_KEYS):
        raise VisualCorpusError("raw_model_data['image'] keys do not match the native contract")
    if not isinstance(image_mask, dict) or set(image_mask) != set(_IMAGE_KEYS):
        raise VisualCorpusError("raw_model_data['image_mask'] keys do not match the native contract")
    for key in _IMAGE_KEYS:
        arr = np.asarray(image[key])
        if arr.shape != (PNG_HEIGHT, PNG_WIDTH, 3) or arr.dtype != np.uint8:
            raise VisualCorpusError(
                f"image[{key!r}] must be uint8{(PNG_HEIGHT, PNG_WIDTH, 3)}, "
                f"got {arr.dtype}{tuple(arr.shape)}"
            )
    if type(image_mask["base_0_rgb"]) is not bool or image_mask["base_0_rgb"] is not True:
        raise VisualCorpusError("image_mask['base_0_rgb'] must be boolean True")
    for key in ("left_wrist_0_rgb", "right_wrist_0_rgb"):
        if type(image_mask[key]) is not bool or image_mask[key] is not False:
            raise VisualCorpusError(f"image_mask[{key!r}] must be boolean False (missing wrist view)")
        if np.any(image[key]):
            raise VisualCorpusError(f"image[{key!r}] must be all-zero (missing wrist view)")
    prompt = raw["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise VisualCorpusError("raw_model_data['prompt'] must be a nonempty string")
    loss_mask = np.asarray(sample.loss_mask32)
    if loss_mask.shape != (ACTION_DIM,) or loss_mask.dtype != np.bool_:
        raise VisualCorpusError(
            f"loss_mask32 must be bool[{ACTION_DIM}], got {loss_mask.dtype}{tuple(loss_mask.shape)}"
        )
    if not loss_mask[:NUM_CHANNELS].all() or loss_mask[NUM_CHANNELS:].any():
        raise VisualCorpusError(
            f"loss_mask32 must be TRUE on the first {NUM_CHANNELS} entries and FALSE on the rest"
        )
    provenance = sample.provenance
    if not isinstance(provenance, dict):
        raise VisualCorpusError("provenance must be a dict")
    pending = provenance.get("pending_transforms")
    if not isinstance(pending, dict) or "normalization" not in pending:
        raise VisualCorpusError("provenance['pending_transforms'] missing normalization flag")
    if pending["normalization"] != "PENDING":
        raise VisualCorpusError(
            f"pending_transforms['normalization'] is {pending['normalization']!r}; "
            "refusing to normalize twice"
        )


def _fitted_metadata(stats: Mapping[str, Any]) -> dict[str, list[str]]:
    """Aligned fitted corpus ids/index SHA lists. Metadata is REQUIRED.

    Raises ``VisualCorpusError`` for absent/non-mapping/empty metadata,
    absent/empty/misaligned ``corpus_ids``/``index_sha256`` lists, duplicate
    corpus ids, or any index SHA that is not 64-character hex. Normalization
    never runs with unbound stats: there is no implicit fixture exception.
    """
    metadata = stats.get("metadata")
    if not isinstance(metadata, Mapping):
        raise VisualCorpusError("stats metadata is required for normalization")
    if not metadata:
        raise VisualCorpusError("stats metadata is empty")
    ids = metadata.get("corpus_ids")
    shas = metadata.get("index_sha256")
    if not isinstance(ids, (list, tuple)) or not isinstance(shas, (list, tuple)):
        raise VisualCorpusError("stats metadata corpus_ids/index_sha256 must be aligned string lists")
    if not ids or not shas:
        raise VisualCorpusError("stats metadata corpus_ids/index_sha256 must be nonempty")
    if len(ids) != len(shas):
        raise VisualCorpusError("stats metadata corpus_ids/index_sha256 are misaligned")
    if not all(isinstance(c, str) and c for c in ids):
        raise VisualCorpusError("stats metadata corpus_ids must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise VisualCorpusError("stats metadata has duplicate corpus_id entries")
    normalized_shas: list[str] = []
    for i, s in enumerate(shas):
        if (
            not isinstance(s, str)
            or len(s) != 64
            or not all(c in "0123456789abcdef" for c in s.lower())
        ):
            raise VisualCorpusError(
                f"stats metadata index_sha256[{i}] is not a 64-character hex SHA-256"
            )
        normalized_shas.append(s.lower())
    return {"corpus_ids": list(ids), "index_sha256": normalized_shas}


def _validate_stats_binding(bound: dict[str, list[str]], provenance: Mapping[str, Any]) -> None:
    """Bind fitted stats to the sample's corpus provenance (no silent unbound use)."""
    prov_corpus = provenance.get("corpus_id")
    prov_sha = provenance.get("index_sha256")
    if not isinstance(prov_corpus, str) or not isinstance(prov_sha, str):
        raise VisualCorpusError(
            "sample provenance lacks corpus_id/index_sha256; cannot bind fitted stats"
        )
    if prov_corpus not in bound["corpus_ids"]:
        raise VisualCorpusError(
            f"sample corpus {prov_corpus!r} is not among the fitted corpora "
            f"{bound['corpus_ids']}"
        )
    slot = bound["corpus_ids"].index(prov_corpus)
    if bound["index_sha256"][slot] != prov_sha.lower():
        raise VisualCorpusError(
            f"sample index SHA-256 for corpus {prov_corpus!r} does not match the "
            "SHA the stats were fitted on"
        )


def _normalize_array(vecs: Mapping[str, np.ndarray], value: Any, name: str) -> np.float32:
    """Apply the official quantile map in float64 and cast to finite float32."""
    x64 = np.asarray(value, dtype=np.float64)
    y = (x64 - vecs["q01"]) / (vecs["q99"] - vecs["q01"] + _EPSILON) * 2.0 - 1.0
    y32 = y.astype(np.float32)
    if not np.isfinite(y32).all():
        raise VisualCorpusError(f"normalized {name} contains non-finite float32 values")
    return y32


def _channel_report(vecs: Mapping[str, np.ndarray], normalized: np.ndarray) -> dict[str, Any]:
    """Report constant / sub-epsilon-range channels and out-of-[-1,1] counts."""
    rng = vecs["q99"] - vecs["q01"]
    constant = [int(i) for i in np.flatnonzero(rng == 0.0)]
    below_epsilon = [int(i) for i in np.flatnonzero((rng > 0.0) & (rng < _EPSILON))]
    out_mask = (normalized < -1.0) | (normalized > 1.0)
    if out_mask.ndim == 1:  # one observation: the mask itself is the count
        counts = out_mask.astype(np.int64)
    else:
        counts = np.count_nonzero(out_mask, axis=0)
    return {
        "constant_channels": constant,
        "below_epsilon_channels": below_epsilon,
        "out_of_range_counts": [int(c) for c in np.atleast_1d(counts)],
    }


def _stats_identity(stats: Mapping[str, Any]) -> dict[str, Any]:
    """Stable identity: numeric stats digest plus the fit provenance.

    The ``stats_sha256`` digest covers ONLY the numeric mean/std/q01/q99
    vectors of both groups; fit metadata (method, corpus ids, index SHA-256s)
    is carried alongside, not digested, so re-labelling a stats dict onto
    different corpora changes the identity without changing the digest.
    Metadata must already be validated (``_fitted_metadata``) before calling.
    """
    payload = {
        group: {key: [float(v) for v in np.asarray(stats[group][key], dtype=np.float64)] for key in _STAT_KEYS}
        for group in _GROUPS
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    identity: dict[str, Any] = {"stats_sha256": digest, "method": _METHOD}
    bound = _fitted_metadata(stats)
    if bound is not None:
        identity["corpus_ids"] = bound["corpus_ids"]
        identity["index_sha256"] = bound["index_sha256"]
    return identity


def normalize_pi05_native_sample(
    sample: Pi05NativeSample, stats: Mapping[str, Any]
) -> Pi05NativeSample:
    """Return a fresh sample with official quantile normalization applied.

    Only ``state`` and ``actions`` change, via the exact official
    ``(x - q01) / (q99 - q01 + 1e-6) * 2 - 1`` map cast to finite float32.
    Images (including all-zero missing wrist views), image/loss masks, prompt
    and other provenance are preserved in independent copies;
    ``pending_transforms['normalization']`` must be exactly ``"PENDING"`` and
    becomes ``"APPLIED"`` so a second call is rejected instead of double
    normalizing. A report of constant/sub-epsilon channels and out-of-range
    counts is recorded under ``provenance['normalization']`` without clipping
    or refitting anything.

    When ``stats`` carries fit metadata (corpus_ids/index_sha256 lists from
    ``fit_pi05_norm_stats``), the sample provenance must record the same
    corpus_id and index SHA-256. Metadata is required — absent, empty, or
    malformed metadata and provenance mismatches are rejected so production
    data is never silently normalized with unbound statistics.
    """
    groups = _validated_stats(stats)
    _validate_native_sample(sample)
    bound = _fitted_metadata(stats)
    _validate_stats_binding(bound, sample.provenance)
    raw = sample.raw_model_data

    new_state = _normalize_array(groups["state"], raw["state"], "state")
    new_actions = _normalize_array(groups["actions"], raw["actions"], "actions")

    raw_new = {
        "state": new_state,
        "image": {key: np.array(raw["image"][key], dtype=np.uint8, copy=True) for key in _IMAGE_KEYS},
        "image_mask": {key: raw["image_mask"][key] for key in _IMAGE_KEYS},
        "actions": new_actions,
        "prompt": raw["prompt"],
    }

    provenance = copy.deepcopy(sample.provenance)
    provenance["pending_transforms"]["normalization"] = "APPLIED"
    provenance["normalization"] = {
        "method": _METHOD,
        "epsilon": _EPSILON,
        "stats": _stats_identity(stats),
        "bound_to_sample_provenance": True,
        "state": _channel_report(groups["state"], new_state),
        "actions": _channel_report(groups["actions"], new_actions),
        "clipping": "none (out-of-range counts are reporting only)",
    }
    return Pi05NativeSample(
        raw_model_data=raw_new,
        loss_mask32=np.array(sample.loss_mask32, dtype=np.bool_, copy=True),
        provenance=provenance,
    )
