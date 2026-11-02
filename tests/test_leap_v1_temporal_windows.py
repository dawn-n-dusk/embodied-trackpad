"""Focused boundary tests for leap_v1.temporal_windows (no real corpus)."""
from __future__ import annotations

import numpy as np
import pytest

from embodied_trackpad.leap_v1.action_resampling import COARSE_DT_S, NATIVE_DT_S
from embodied_trackpad.leap_v1.temporal_windows import (
    FORMAT_VERSION,
    WINDOW_REFS,
    WINDOW_ROWS,
    WINDOW_STEPS,
    TemporalWindowError,
    WindowObservation,
    WindowRow,
    assemble_windows,
    load_window_observation,
)

DT = NATIVE_DT_S
T = lambda step: step * DT  # one shared formula => exact float agreement


def _ctrl(step_start: np.ndarray, ch=16) -> np.ndarray:
    steps = step_start[:, None] + np.zeros((1, ch))
    slopes = np.arange(ch, dtype=np.float64)[None, :] * 1e-4
    return steps * 1e-3 + slopes


def make_row(k, *, corpus="corp", episode=0, split="train", phase=None,
             first=None, prev_ctrl=None, time_shift=0.0, frame_id=None,
             episode_key=None, task_profile=None, collection_id=None,
             mutate_last=None) -> WindowRow:
    first = 100 + 25 * k if first is None else first
    steps = np.arange(first, first + 25)
    executed = _ctrl(steps)
    if mutate_last is not None:
        executed = executed.copy()
        executed[-1] = mutate_last
    if prev_ctrl is None:
        prev_ctrl = _ctrl(np.array([first - 1]))[0]
    return WindowRow(
        corpus_id=corpus, row_index=k, split=split, episode_id=episode,
        frame_id=frame_id if frame_id is not None else 10 + k,
        first_step_index=first, end_exclusive_step_index=first + 25,
        consumed_prev_step_index=first - 1, dt_s=DT,
        consumed_prev_sim_time_s=T(first - 1) + time_shift,
        first_sim_time_s=T(first) + time_shift,
        end_sim_time_s=T(first + 24) + time_shift,
        previous_ctrl_rad=prev_ctrl, executed_ctrl_rad=executed,
        index_path="idx.jsonl", index_sha256="a" * 64,
        episode_key=episode_key, task_profile=task_profile,
        collection_id=collection_id, phase=phase,
    )


def chain(n, **kw):
    return [make_row(k, **kw) for k in range(n)]


def reindex(rows):
    for i, r in enumerate(rows):
        object.__setattr__(r, "row_index", i)
    return rows


# --- continuous handcomputed ramp -------------------------------------------

def test_continuous_ramp_handcomputed():
    rows = chain(WINDOW_ROWS)
    res = assemble_windows(rows)
    assert len(res.windows) == 1
    assert res.report.rejections.get("insufficient_suffix", 0) == 9
    w = res.windows[0]
    assert w.native_ctrl_rad.shape == (WINDOW_STEPS, 16)
    assert w.coarse_ctrl_rad.shape == (WINDOW_REFS, 16)
    assert np.array_equal(w.native_ctrl_rad, np.concatenate([r.executed_ctrl_rad for r in rows]))
    assert np.array_equal(w.coarse_ctrl_rad, w.native_ctrl_rad[9::10])
    expected_ep = rows[0].consumed_prev_sim_time_s + np.arange(1, 26) * COARSE_DT_S
    assert np.allclose(w.coarse_endpoint_sim_time_s, expected_ep)
    assert w.provenance.first_step_index == rows[0].first_step_index
    assert w.provenance.end_exclusive_step_index == rows[-1].end_exclusive_step_index
    assert w.provenance.consumed_prev_sim_time_s == rows[0].consumed_prev_sim_time_s
    assert w.provenance.index_sha256 == "a" * 64
    # linear ramp per 10-step segment -> reconstruction at float rounding only
    assert np.all(w.metrics.recon_rms_rad <= 1e-15)
    assert np.all(w.metrics.recon_max_abs_rad <= 1e-15)
    assert not w.metrics.constant_channels.any()
    assert np.allclose(w.metrics.native_jump_rad, 1e-3)
    assert res.report.windows_total == 1
    assert res.report.limitations and "phase/goal/reset" in res.report.limitations[0]


def test_constant_channel_detected_from_native_not_coarse():
    rows = chain(WINDOW_ROWS)
    exe = np.stack([r.executed_ctrl_rad for r in rows]).copy()
    exe[:, :, 3] = 0.5  # pin channel 3 flat across all 250 native rows
    exe.setflags(write=False)
    for i, r in enumerate(rows):
        prev = np.array(r.previous_ctrl_rad, copy=True)
        prev[3] = 0.5  # keep the prev->native chain flat too
        prev.setflags(write=False)
        object.__setattr__(r, "previous_ctrl_rad", prev)
        object.__setattr__(r, "executed_ctrl_rad", exe[i])
    w = assemble_windows(rows).windows[0]
    assert w.metrics.constant_channels[3]
    assert w.metrics.native_jump_rad[3] == 0.0
    assert not w.metrics.constant_channels[0]


# --- gap / time jump / prev-ctrl mismatch ------------------------------------

def test_gap_rejected_counted():
    # shift this and ALL later rows so the step axis has a true gap, no overlap
    rows = chain(5) + [make_row(k, first=100 + 25 * (k + 1)) for k in range(5, 12)]
    res = assemble_windows(rows)
    assert res.report.rejections.get("gap", 0) == 1
    assert res.report.windows_total == 0
    assert res.report.rejections.get("insufficient_suffix", 0) == 12


def test_time_jump_rejected():
    rows = chain(WINDOW_ROWS)
    rows[5] = make_row(5, time_shift=0.004)
    res = assemble_windows(rows)
    assert res.report.rejections.get("time_jump", 0) == 2  # both broken adjacencies
    assert res.report.windows_total == 0


def test_prev_ctrl_mismatch_rejected():
    rows = chain(WINDOW_ROWS)
    rows[5] = make_row(5, prev_ctrl=np.full(16, 9.9))
    res = assemble_windows(rows)
    assert res.report.rejections.get("prev_ctrl_mismatch", 0) == 1
    assert res.report.windows_total == 0


# --- overlaps: exact duplicates merge, any other overlap aborts --------------

def test_exact_duplicates_merge_with_full_provenance():
    dup1, dup2 = make_row(0), make_row(0)
    rows = reindex([dup1, dup2] + chain(WINDOW_ROWS))  # 3 consecutive duplicates
    res = assemble_windows(rows)
    assert res.report.rejections.get("identical_overlap_merged", 0) == 2
    assert len(res.windows) == 1
    ids = res.windows[0].provenance.row_indices
    assert ids[:3] == (0, 1, 2) and len(ids) == 12  # 10 rows + 2 duplicate ids
    assert res.report.rejections.get("insufficient_suffix", 0) == 9


def test_duplicate_time_disagreement_is_conflict():
    rows = chain(WINDOW_ROWS)
    rows = reindex(rows[:4] + [make_row(3, time_shift=0.002)] + rows[4:])
    with pytest.raises(TemporalWindowError) as e:
        assemble_windows(rows)
    assert e.value.code == "overlap_conflict"


def test_partial_overlap_conflict_raises():
    rows = reindex(chain(1) + [make_row(0, first=110)] + chain(9, first=125))
    with pytest.raises(TemporalWindowError) as e:
        assemble_windows(rows)
    assert e.value.code == "overlap_conflict"


def test_conflict_followed_by_valid_rows_still_raises():
    rows = reindex(chain(4) + [make_row(0, first=100 + 10)] + chain(10, episode=0))
    with pytest.raises(TemporalWindowError) as e:
        assemble_windows(rows)
    assert e.value.code in ("overlap_conflict", "nonmonotone_step")


# --- boundaries: split / reset / phase / corpus / runs ------------------------

def test_reset_nonmonotone_step_rejected():
    rows = chain(WINDOW_ROWS)
    r = rows[6]
    object.__setattr__(r, "first_step_index", 100)
    object.__setattr__(r, "end_exclusive_step_index", 125)
    object.__setattr__(r, "consumed_prev_step_index", 99)
    with pytest.raises(TemporalWindowError) as e:
        assemble_windows(rows)
    assert e.value.code == "nonmonotone_step"  # reset/restart never emits windows


def test_split_groups_never_cross():
    rows = chain(WINDOW_ROWS + 4)
    for r in rows[:6]:
        object.__setattr__(r, "split", "validation")
    res = assemble_windows(rows)
    assert {g.group[1] for g in res.report.groups} == {"train", "validation"}
    assert res.report.windows_total == 0


def test_phase_groups_never_cross():
    rows = chain(WINDOW_ROWS + 4)
    for r in rows[:6]:
        object.__setattr__(r, "phase", "reach")
    for r in rows[6:]:
        object.__setattr__(r, "phase", "press")
    res = assemble_windows(rows)
    assert res.report.phase_goal_reset_present
    assert len(res.report.groups) == 2
    assert res.report.windows_total == 0


def test_recurring_group_never_reconnects_across_boundary():
    a1 = [make_row(k, episode=1) for k in range(6)]
    b = [make_row(k, episode=2) for k in range(10)]
    a2 = [make_row(k, episode=1) for k in range(4)]  # episode 1 recurs later
    res = assemble_windows(reindex(a1 + b + a2))
    assert res.report.groups_total == 3  # contiguous runs, no dict regrouping
    assert res.report.windows_total == 1  # only the uninterrupted B run
    assert all(w.provenance.episode_id == 2 for w in res.windows)
    assert all(not any(r in {0, 1, 2, 3, 4, 5, 16, 17, 18, 19}
                       for r in w.provenance.row_indices) for w in res.windows)


def test_mixed_corpus_rejected():
    rows = chain(WINDOW_ROWS)
    rows[3] = make_row(3, corpus="other")
    with pytest.raises(TemporalWindowError) as e:
        assemble_windows(rows)
    assert e.value.code == "mixed_corpus"


def test_nonmonotone_row_order_rejected():
    rows = chain(WINDOW_ROWS)
    object.__setattr__(rows[4], "row_index", 2)
    with pytest.raises(TemporalWindowError) as e:
        assemble_windows(rows)
    assert e.value.code == "nonmonotone_row_order"


def test_insufficient_suffix_counted():
    res = assemble_windows(chain(WINDOW_ROWS + 3))
    assert len(res.windows) == 4
    assert res.report.rejections.get("insufficient_suffix", 0) == 9


# --- row integrity: dt / units / shapes --------------------------------------

@pytest.mark.parametrize("kw,code", [
    (dict(dt_s=0.02), "invalid_dt"),
    (dict(previous_ctrl_rad=np.full(16, np.nan)), "invalid_units"),
    (dict(executed_ctrl_rad=np.zeros((24, 16))), "invalid_shape"),
    (dict(executed_ctrl_rad=np.zeros((25, 15))), "invalid_shape"),
    (dict(previous_ctrl_rad=[False] * 16), "invalid_units"),
    (dict(previous_ctrl_rad=["x"] * 16), "invalid_units"),
    (dict(previous_ctrl_rad=np.full(16, 1 + 2j)), "invalid_units"),
    (dict(row_index=True), "invalid_shape"),
    (dict(split="bogus"), "invalid_split"),
])
def test_row_integrity_errors(kw, code):
    base = dict(corpus_id="c", row_index=0, split="train", episode_id=0, frame_id=0,
                first_step_index=100, end_exclusive_step_index=125,
                consumed_prev_step_index=99, dt_s=DT,
                consumed_prev_sim_time_s=T(99), first_sim_time_s=T(100),
                end_sim_time_s=T(124),
                previous_ctrl_rad=np.zeros(16), executed_ctrl_rad=np.zeros((25, 16)))
    base.update(kw)
    with pytest.raises(TemporalWindowError) as e:
        WindowRow(**base)
    assert e.value.code == code


# --- mutation isolation -------------------------------------------------------

def test_mutation_isolation():
    src_exec = _ctrl(np.arange(100, 125))
    src_prev = _ctrl(np.array([99]))[0]
    rows = chain(WINDOW_ROWS)
    object.__setattr__(rows[0], "executed_ctrl_rad", src_exec)
    object.__setattr__(rows[0], "previous_ctrl_rad", src_prev)
    w = assemble_windows(rows).windows[0]
    before = w.native_ctrl_rad[:25].copy()
    src_exec[:] = 999.0
    src_prev[:] = 999.0
    assert np.array_equal(w.native_ctrl_rad[:25], before)
    w.native_ctrl_rad[0, 0] = 12345.0
    w.coarse_ctrl_rad[0, 0] = 54321.0
    assert rows[0].executed_ctrl_rad[0, 0] != 12345.0
    assert rows[0].executed_ctrl_rad[0, 0] != 54321.0


def test_readonly_snapshots():
    r = make_row(0)
    assert not r.executed_ctrl_rad.flags.writeable
    assert not r.previous_ctrl_rad.flags.writeable


# --- observation loader: current frame only, coarse target -------------------

class _StubExample:
    def __init__(self, prov, obs):
        self.provenance = prov
        self.observation = obs
        self.target = {"target_ctrl_rad": np.zeros((25, 16))}  # native, must not leak


class _StubCorpus:
    def __init__(self, example, *, corpus_id="corp", index_sha256="a" * 64):
        self._ex = example
        self.corpus_id = corpus_id
        self.index_sha256 = index_sha256
        self.requested: list[int] = []

    def example(self, i):
        self.requested.append(i)
        return self._ex


def _obs_fixture():
    rows = chain(WINDOW_ROWS)
    w = assemble_windows(rows).windows[0]
    prov = {
        "sample_id": "corp:ep:frame-10", "corpus_id": "corp",
        "index_sha256": "a" * 64, "episode_id": 0, "frame_id": 10,
        "split": "train", "frame_monotonic_time_s": 1234.5,
        "action_window": {
            "first_step_index": rows[0].first_step_index,
            "first_sim_time_s": rows[0].first_sim_time_s,
            "consumed_prev_sim_time_s": rows[0].consumed_prev_sim_time_s,
        },
    }
    obs = {
        "screen_rgb": np.zeros((4, 4, 3), np.uint8), "q_rad16": np.zeros(16),
        "qvel_rad_s16": np.zeros(16),
        "previous_ctrl_rad16": rows[0].previous_ctrl_rad.astype(np.float32).astype(np.float64),
        "fsr_N4": None, "motor_current_A16": None, "wrist": None,
        "fsr_valid4": [False] * 4, "motor_current_valid16": [False] * 16,
        "wrist_valid": False, "oracle_xy": np.zeros(2),  # must be filtered out
    }
    return w, prov, obs


def test_observation_current_frame_only_coarse_target():
    w, prov, obs = _obs_fixture()
    corpus = _StubCorpus(_StubExample(prov, obs))
    out = load_window_observation(corpus, w)
    assert isinstance(out, WindowObservation)
    assert corpus.requested == [w.provenance.row_indices[0]]
    assert out.target["target_ctrl_rad"].shape == (25, 16)
    assert out.target["dt_s"] == COARSE_DT_S
    assert "native_ctrl_rad" not in out.target
    assert set(out.observation) == {
        "screen_rgb", "q_rad16", "qvel_rad_s16", "previous_ctrl_rad16",
        "fsr_N4", "motor_current_A16", "wrist",
        "fsr_valid4", "motor_current_valid16", "wrist_valid",
    }
    assert out.provenance["frame_monotonic_time_s"] == 1234.5
    assert out.provenance["format_version"] == FORMAT_VERSION
    assert np.allclose(out.target["coarse_endpoint_sim_time_s"],
                       w.coarse_endpoint_sim_time_s)


@pytest.mark.parametrize("mutate", [
    dict(frame_id=11), dict(episode_id=9), dict(split="validation"),
    dict(index_sha256="b" * 64),
    dict(action_window={"first_step_index": -1}),
    dict(action_window={"first_step_index": 100, "first_sim_time_s": 9.9,
                        "consumed_prev_sim_time_s": 9.898}),
])
def test_observation_mismatch_rejected(mutate):
    w, prov, obs = _obs_fixture()
    prov = dict(prov)
    base_aw = dict(prov["action_window"])
    prov.update(mutate)
    if "action_window" in mutate and "first_step_index" in mutate["action_window"]:
        merged = dict(base_aw)
        merged.update(mutate["action_window"])
        prov["action_window"] = merged
    corpus = _StubCorpus(_StubExample(prov, obs),
                         index_sha256=mutate.get("index_sha256", "a" * 64))
    if "corpus_id" in mutate:
        corpus.corpus_id = "corp"
    with pytest.raises(TemporalWindowError) as e:
        load_window_observation(corpus, w)
    assert e.value.code in ("future_observation", "mixed_corpus", "index_sha256_mismatch")


def test_observation_wrong_corpus_rejected():
    w, prov, obs = _obs_fixture()
    corpus = _StubCorpus(_StubExample(prov, obs), corpus_id="evil")
    with pytest.raises(TemporalWindowError) as e:
        load_window_observation(corpus, w)
    assert e.value.code == "mixed_corpus"


def test_observation_prev_ctrl_mismatch_rejected():
    w, prov, obs = _obs_fixture()
    obs = dict(obs)
    obs["previous_ctrl_rad16"] = np.full(16, 7.7)
    corpus = _StubCorpus(_StubExample(prov, obs))
    with pytest.raises(TemporalWindowError) as e:
        load_window_observation(corpus, w)
    assert e.value.code == "future_observation"


def test_observation_accepts_documented_f32_prev_rounding():
    w, prov, obs = _obs_fixture()
    obs = dict(obs)
    # f64 value that only differs beyond float32 precision must be accepted
    tweaked = w.previous_ctrl_rad.copy()
    tweaked[0] += 1e-9
    obs["previous_ctrl_rad16"] = tweaked.astype(np.float32).astype(np.float64)
    corpus = _StubCorpus(_StubExample(prov, obs))
    out = load_window_observation(corpus, w)
    assert isinstance(out, WindowObservation)
