"""Boundary tests for the lazy RGB visual corpus (visual_dataset).

Synthetic fixtures only: tiny JSONL indexes plus real 800x600 PNGs written
under a temporary evidence_root. Covers integrity (index/PNG SHA), path
confinement, native 16/25/dt=0.002 semantics, observation whitelist,
mutation/alias isolation and per-corpus provenance namespacing.
"""
from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from embodied_trackpad.leap_v1 import small_policy
from embodied_trackpad.leap_v1.visual_dataset import (
    ACTION_SEMANTICS,
    DT_S,
    HORIZON,
    HORIZON_S,
    NUM_CHANNELS,
    OBSERVATION_KEYS,
    PNG_HEIGHT,
    PNG_WIDTH,
    VisualCorpus,
    VisualCorpusError,
)

EVIDENCE_SUBDIR = "evidence"


def _png_bytes(array: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(array, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _actor(png_ref: str, png_sha: str) -> dict:
    return {
        "png_ref": png_ref,
        "png_sha256": png_sha,
        "error_xy_px": [8.0, 4.0],
        "q_rad16": [0.01] * NUM_CHANNELS,
        "qvel_rad_s16": [0.0] * NUM_CHANNELS,
        "previous_ctrl_rad16": [0.0] * NUM_CHANNELS,
        "valid": {
            "png_valid": True,
            "visual_valid": True,
            "q_valid": True,
            "qvel_valid": True,
            "previous_ctrl_valid": True,
            "fsr_valid4": [False] * 4,
            "motor_current_valid16": [False] * NUM_CHANNELS,
            "wrist_valid": False,
        },
        "fsr_N4": None,
        "motor_current_A16": None,
        "wrist": None,
    }


def _labels(scale: float = 1e-5) -> dict:
    prev = np.zeros(NUM_CHANNELS, dtype=np.float64)
    executed = np.zeros((HORIZON, NUM_CHANNELS), dtype=np.float64)
    executed[:, 0] = scale * np.sin(np.linspace(0, np.pi, HORIZON))
    return {
        "executed_ctrl_rad": executed.tolist(),
        "delta_from_frame_previous_ctrl_rad": (executed - prev).tolist(),
    }


def _window() -> dict:
    return {
        "consumed_prev_step_index": 1450,
        "first_step_index": 1451,
        "end_exclusive_step_index": 1476,
        "consumed_prev_sim_time_s": 2.9,
        "first_sim_time_s": 2.902,
        "end_sim_time_s": 2.95,
        "dt_s": DT_S,
    }


def _row(png_ref: str, png_sha: str, *, episode_id: int = 0, frame_id: int = 1,
         split: str = "train", labels: dict | None = None,
         window: dict | None = None) -> dict:
    return {
        "episode_id": episode_id,
        "frame_id": frame_id,
        "split": split,
        "target_class": episode_id % 12,
        "actor_input": _actor(png_ref, png_sha),
        "action_window": window if window is not None else _window(),
        "labels": labels if labels is not None else _labels(),
    }


def _write_index(path: Path, records: list[dict]) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


def _write_png(root: Path, rel: str, array: np.ndarray) -> tuple[Path, bytes]:
    png_path = root / rel
    png_path.parent.mkdir(parents=True, exist_ok=True)
    data = _png_bytes(array)
    png_path.write_bytes(data)
    return png_path, data


def _base_image(value: int) -> np.ndarray:
    rng = np.random.default_rng(value)
    return rng.integers(0, 256, size=(PNG_HEIGHT, PNG_WIDTH, 3), dtype=np.uint8)


def _corpus(tmp_path: Path, records: list[dict], *, corpus_id: str = "corp-a",
            sha: str | None = None) -> VisualCorpus:
    index_path = tmp_path / "sample_index.jsonl"
    _write_index(index_path, records)
    actual = small_policy.sha256_file(index_path)
    return VisualCorpus(
        index_path,
        tmp_path / EVIDENCE_SUBDIR,
        sha if sha is not None else actual,
        corpus_id,
    )


# --------------------------------------------------------------------------- #
# integrity: index SHA before any image work
# --------------------------------------------------------------------------- #

def test_wrong_expected_index_sha_rejected_before_image_work(tmp_path):
    root = tmp_path / EVIDENCE_SUBDIR
    rel = "ep0/cycle-001.png"
    _, data = _write_png(root, rel, _base_image(0))
    rec = _row(rel, _sha(data))
    index_path = tmp_path / "sample_index.jsonl"
    _write_index(index_path, [rec])
    with pytest.raises(VisualCorpusError, match="index SHA-256 mismatch"):
        VisualCorpus(index_path, root, "0" * 64, "corp-a")
    # Tampered index bytes (valid JSON, changed field) fail against the
    # previously recorded digest even though the JSON itself still parses.
    original_sha = small_policy.sha256_file(index_path)
    rec["frame_id"] = 99
    _write_index(index_path, [rec])
    with pytest.raises(VisualCorpusError, match="index SHA-256 mismatch"):
        VisualCorpus(index_path, root, original_sha, "corp-a")


def test_empty_index_rejected(tmp_path):
    root = tmp_path / EVIDENCE_SUBDIR
    root.mkdir(parents=True)
    index_path = tmp_path / "sample_index.jsonl"
    _write_index(index_path, [])
    with pytest.raises(VisualCorpusError, match="empty"):
        VisualCorpus(index_path, root, small_policy.sha256_file(index_path), "corp-a")


def test_missing_png_ref_rejected_at_init(tmp_path):
    root = tmp_path / EVIDENCE_SUBDIR
    root.mkdir(parents=True)
    rec = _row("", "0" * 64)
    with pytest.raises(VisualCorpusError, match="png_ref"):
        _corpus(tmp_path, [rec])


# --------------------------------------------------------------------------- #
# lazy decode: shape, dtype, split accounting, pixel fidelity
# --------------------------------------------------------------------------- #

def test_len_split_counts_and_lazy_decode(tmp_path):
    root = tmp_path / EVIDENCE_SUBDIR
    records = []
    for seed, split, tag in ((0, "train", 1), (1, "validation", 2), (4, "test", 3)):
        rel = f"ep{seed}/cycle-001.png"
        _, data = _write_png(root, rel, _base_image(tag))
        records.append(_row(rel, _sha(data), episode_id=seed, frame_id=10 + seed, split=split))
    corpus = _corpus(tmp_path, records)
    assert len(corpus) == 3
    assert corpus.split_counts == {"train": 1, "validation": 1, "test": 1}
    assert corpus.split_indices("validation") == [1]

    image = np.array(Image.open(BytesIO(_png_bytes(_base_image(1)))), dtype=np.uint8)
    ex = corpus.example(0)
    assert ex.observation["screen_rgb"].shape == (PNG_HEIGHT, PNG_WIDTH, 3)
    assert ex.observation["screen_rgb"].dtype == np.uint8
    assert np.array_equal(ex.observation["screen_rgb"], image)
    assert int(ex.observation["screen_rgb"].min()) < 256  # actual pixel range
    assert set(ex.observation.keys()) == set(OBSERVATION_KEYS)
    assert "error_xy_px" not in ex.observation
    assert "target_class" not in ex.observation


def test_pixels_vary_when_image_changes_with_matched_hash(tmp_path):
    root = tmp_path / EVIDENCE_SUBDIR
    records = []
    for i, (seed, frame_id) in enumerate(((0, 1), (0, 2))):
        rel = f"ep0/cycle-{frame_id:03d}.png"
        _, data = _write_png(root, rel, _base_image(100 + i))
        records.append(_row(rel, _sha(data), episode_id=seed, frame_id=frame_id))
    corpus = _corpus(tmp_path, records)
    a = corpus.example(0).observation["screen_rgb"]
    b = corpus.example(1).observation["screen_rgb"]
    assert a.shape == b.shape == (PNG_HEIGHT, PNG_WIDTH, 3)
    assert not np.array_equal(a, b)


def test_native_target_actions_absolute_25x16_dt(tmp_path):
    root = tmp_path / EVIDENCE_SUBDIR
    rel = "ep0/cycle-001.png"
    _, data = _write_png(root, rel, _base_image(0))
    labels = _labels(scale=3e-4)
    corpus = _corpus(tmp_path, [_row(rel, _sha(data), labels=labels)])
    ex = corpus.example(0)
    target = ex.target["target_ctrl_rad"]
    assert target.shape == (HORIZON, NUM_CHANNELS)
    assert target.dtype == np.float32
    np.testing.assert_allclose(target, np.asarray(labels["executed_ctrl_rad"], dtype=np.float32))
    # Absolute executed references, not deltas: mid-horizon refs moved away
    # from the all-zero previous control.
    assert target[12, 0] != 0.0
    assert not np.allclose(target, np.zeros_like(target))
    aw = ex.provenance["action_window"]
    assert aw["dt_s"] == DT_S and aw["horizon"] == HORIZON and aw["horizon_s"] == HORIZON_S
    assert aw["action_semantics"] == ACTION_SEMANTICS


# --------------------------------------------------------------------------- #
# tampered / malformed images
# --------------------------------------------------------------------------- #

def test_tampered_png_rejected(tmp_path):
    root = tmp_path / EVIDENCE_SUBDIR
    rel = "ep0/cycle-001.png"
    png_path, data = _write_png(root, rel, _base_image(0))
    corpus = _corpus(tmp_path, [_row(rel, _sha(data))])
    png_path.write_bytes(_png_bytes(_base_image(7)))  # same size class, new content
    with pytest.raises(VisualCorpusError, match="PNG SHA-256 mismatch"):
        corpus.example(0)


def test_non_png_rejected_even_with_matched_hash(tmp_path):
    root = tmp_path / EVIDENCE_SUBDIR
    rel = "ep0/cycle-001.png"
    fake = b"definitely not a png payload"
    (root / "ep0").mkdir(parents=True)
    (root / rel).write_bytes(fake)
    corpus = _corpus(tmp_path, [_row(rel, _sha(fake))])
    with pytest.raises(VisualCorpusError, match="not a PNG"):
        corpus.example(0)


def test_wrong_size_png_rejected(tmp_path):
    root = tmp_path / EVIDENCE_SUBDIR
    rel = "ep0/cycle-001.png"
    small = np.zeros((100, 100, 3), dtype=np.uint8)
    _, data = _write_png(root, rel, small)
    corpus = _corpus(tmp_path, [_row(rel, _sha(data))])
    with pytest.raises(VisualCorpusError, match="PNG size"):
        corpus.example(0)


def test_rgba_png_opaque_alpha_decoded_to_rgb(tmp_path):
    root = tmp_path / EVIDENCE_SUBDIR
    rel = "ep0/cycle-001.png"
    base = _base_image(11)
    rgba = np.concatenate([base, np.full((PNG_HEIGHT, PNG_WIDTH, 1), 255, dtype=np.uint8)], axis=2)
    png_path = root / rel
    png_path.parent.mkdir(parents=True, exist_ok=True)
    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    data = buf.getvalue()
    png_path.write_bytes(data)
    corpus = _corpus(tmp_path, [_row(rel, _sha(data))])
    ex = corpus.example(0)
    assert ex.observation["screen_rgb"].shape == (PNG_HEIGHT, PNG_WIDTH, 3)
    assert np.array_equal(ex.observation["screen_rgb"], base)


def test_rgba_png_non_opaque_alpha_rejected(tmp_path):
    root = tmp_path / EVIDENCE_SUBDIR
    rel = "ep0/cycle-001.png"
    rgba = np.concatenate(
        [_base_image(12), np.full((PNG_HEIGHT, PNG_WIDTH, 1), 128, dtype=np.uint8)], axis=2
    )
    png_path = root / rel
    png_path.parent.mkdir(parents=True, exist_ok=True)
    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    data = buf.getvalue()
    png_path.write_bytes(data)
    corpus = _corpus(tmp_path, [_row(rel, _sha(data))])
    with pytest.raises(VisualCorpusError, match="non-opaque alpha"):
        corpus.example(0)


# --------------------------------------------------------------------------- #
# path confinement
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ref", ["../outside.png", "/etc/hostname", "a/../../b.png"])
def test_escaping_or_absolute_refs_rejected(tmp_path, ref):
    root = tmp_path / EVIDENCE_SUBDIR
    root.mkdir(parents=True)
    corpus = _corpus(tmp_path, [_row(ref, "0" * 64)])
    with pytest.raises(VisualCorpusError, match="png_ref|not found"):
        corpus.example(0)


def test_symlink_escape_rejected(tmp_path):
    root = tmp_path / EVIDENCE_SUBDIR
    outside, outside_data = _write_png(tmp_path, "outside.png", _base_image(0))
    link = root / "link.png"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside)
    corpus = _corpus(tmp_path, [_row("link.png", _sha(outside_data))])
    with pytest.raises(VisualCorpusError, match="escapes evidence_root"):
        corpus.example(0)


def test_missing_png_rejected(tmp_path):
    root = tmp_path / EVIDENCE_SUBDIR
    root.mkdir(parents=True)
    corpus = _corpus(tmp_path, [_row("ep0/none.png", "0" * 64)])
    with pytest.raises(VisualCorpusError, match="not found"):
        corpus.example(0)


# --------------------------------------------------------------------------- #
# labels and timing
# --------------------------------------------------------------------------- #

def test_nonfinite_labels_rejected_at_index_validation(tmp_path):
    root = tmp_path / EVIDENCE_SUBDIR
    rel = "ep0/cycle-001.png"
    _, data = _write_png(root, rel, _base_image(0))
    labels = _labels()
    labels["executed_ctrl_rad"][3][0] = float("nan")
    # Whole-index validation (small_policy.load_sample_index) rejects first.
    with pytest.raises(ValueError, match="non-finite"):
        _corpus(tmp_path, [_row(rel, _sha(data), labels=labels)])


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_float64_finite_but_float32_overflow_labels_rejected(tmp_path):
    root = tmp_path / EVIDENCE_SUBDIR
    rel = "ep0/cycle-001.png"
    _, data = _write_png(root, rel, _base_image(0))
    labels = _labels()
    labels["executed_ctrl_rad"][0][0] = 1e39  # finite float64, inf as float32
    # Keep the optional cumulative-offset field consistent so index validation
    # passes and the example-time cast check is what fires.
    labels["delta_from_frame_previous_ctrl_rad"] = np.asarray(
        labels["executed_ctrl_rad"], dtype=np.float64
    ).tolist()
    corpus = _corpus(tmp_path, [_row(rel, _sha(data), labels=labels)])
    with pytest.raises(VisualCorpusError, match="overflows float32"):
        corpus.example(0)


def test_invalid_action_window_timing_rejected(tmp_path):
    root = tmp_path / EVIDENCE_SUBDIR
    rel = "ep0/cycle-001.png"
    _, data = _write_png(root, rel, _base_image(0))
    window = _window()
    window["end_sim_time_s"] = 2.96  # inconsistent with (horizon-1)*dt
    corpus = _corpus(tmp_path, [_row(rel, _sha(data), window=window)])
    with pytest.raises(VisualCorpusError, match="end_sim_time_s"):
        corpus.example(0)


def test_wrong_dt_rejected_at_index_validation(tmp_path):
    root = tmp_path / EVIDENCE_SUBDIR
    rel = "ep0/cycle-001.png"
    _, data = _write_png(root, rel, _base_image(0))
    window = _window()
    window["dt_s"] = 0.0005  # v2/leap_sim semantics are not v1
    with pytest.raises(ValueError, match="dt_s"):
        _corpus(tmp_path, [_row(rel, _sha(data), window=window)])


# --------------------------------------------------------------------------- #
# isolation, whitelist, corpora separation
# --------------------------------------------------------------------------- #

def test_mutation_of_returned_arrays_does_not_alias_corpus(tmp_path):
    root = tmp_path / EVIDENCE_SUBDIR
    rel = "ep0/cycle-001.png"
    _, data = _write_png(root, rel, _base_image(0))
    corpus = _corpus(tmp_path, [_row(rel, _sha(data))])
    ex1 = corpus.example(0)
    pristine_rgb = ex1.observation["screen_rgb"].copy()
    pristine_q = ex1.observation["q_rad16"].copy()
    ex1.observation["screen_rgb"][:] = 0
    ex1.observation["q_rad16"][:] = -1.0
    ex1.target["target_ctrl_rad"][:] = 0.0
    ex2 = corpus.example(0)
    assert np.array_equal(ex2.observation["screen_rgb"], pristine_rgb)
    assert np.array_equal(ex2.observation["q_rad16"], pristine_q)
    assert not np.all(ex2.target["target_ctrl_rad"] == 0.0)


def test_two_corpora_same_episode_frame_numbers_stay_separate(tmp_path):
    root = tmp_path / EVIDENCE_SUBDIR
    indexes = []
    for corpus_id, tag in (("corp-a", 0), ("corp-b", 1)):
        rel = f"{corpus_id}/ep0/cycle-001.png"
        _, data = _write_png(root, rel, _base_image(50 + tag))
        index_path = tmp_path / f"index_{corpus_id}.jsonl"
        _write_index(index_path, [_row(rel, _sha(data), episode_id=0, frame_id=1)])
        indexes.append((corpus_id, index_path))
    corpora = [
        VisualCorpus(path, root, small_policy.sha256_file(path), cid)
        for cid, path in indexes
    ]
    assert corpora[0].corpus_id != corpora[1].corpus_id
    for corpus in corpora:
        assert len(corpus) == 1
        ex = corpus.example(0)
        assert ex.provenance["episode_id"] == 0
        assert ex.provenance["frame_id"] == 1
        assert ex.provenance["corpus_id"] == corpus.corpus_id
        assert ex.provenance["sample_id"].startswith(f"{corpus.corpus_id}:")
    assert corpora[0].example(0).provenance["sample_id"] != corpora[1].example(0).provenance["sample_id"]
    assert not np.array_equal(
        corpora[0].example(0).observation["screen_rgb"],
        corpora[1].example(0).observation["screen_rgb"],
    )
    # Index identity: each corpus hashes its own index, never the other's.
    assert corpora[0].index_sha256 != corpora[1].index_sha256


def test_example_index_bounds_and_type(tmp_path):
    root = tmp_path / EVIDENCE_SUBDIR
    rel = "ep0/cycle-001.png"
    _, data = _write_png(root, rel, _base_image(0))
    corpus = _corpus(tmp_path, [_row(rel, _sha(data))])
    with pytest.raises(IndexError):
        corpus.example(1)
    with pytest.raises(IndexError):
        corpus.example(-1)
    with pytest.raises(VisualCorpusError, match="int"):
        corpus.example(True)
