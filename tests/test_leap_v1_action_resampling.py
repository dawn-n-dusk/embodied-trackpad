"""Pure NumPy tests for the LEAP v1 control-resampling helper.

These cover only the numerical contract of
:mod:`embodied_trackpad.leap_v1.action_resampling`. They are not a corpus
sampler and require no pilot metadata.
"""
from __future__ import annotations

import numpy as np
import pytest

from embodied_trackpad.leap_v1.action_resampling import (
    CHANNELS,
    COARSE_DT_S,
    NATIVE_DT_S,
    coarsen_native_controls,
    interpolate_coarse_controls,
)


def _fine(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.5, 0.5, size=(n, CHANNELS))


def test_endpoint_selection_off_by_one():
    fine = _fine(20, seed=1)
    prev = np.full(CHANNELS, 0.01)
    result = coarsen_native_controls(fine, prev)
    # 20 fine steps -> endpoints 9 and 19, not 10 and 20
    np.testing.assert_array_equal(result.coarse[0], fine[9])
    np.testing.assert_array_equal(result.coarse[1], fine[19])
    assert result.coarse.shape == (2, CHANNELS)
    assert result.fine_steps == 20
    assert result.coarse_steps == 2
    assert result.native_dt_s == NATIVE_DT_S == 0.002
    assert result.coarse_dt_s == COARSE_DT_S == 0.020
    assert result.horizon_s == pytest.approx(20 * 0.002)


def test_linear_ramp_exact_reconstruction():
    prev = np.full(CHANNELS, 0.25)
    total = np.linspace(-0.4, 0.6, CHANNELS)
    fine = prev + (np.arange(1, 21) / 20.0)[:, None] * total
    result = coarsen_native_controls(fine, prev)
    reconstructed = interpolate_coarse_controls(result.coarse, prev)
    np.testing.assert_allclose(reconstructed, fine, rtol=0.0, atol=1e-14)
    assert result.rms_rad < 1e-15
    assert result.max_abs_rad < 1e-15


def test_one_channel_step_reconstruction_error():
    fine = np.zeros((20, CHANNELS))
    fine[10:, 3] = 1.0  # step on one channel from fine index 10 onward
    prev = np.zeros(CHANNELS)
    result = coarsen_native_controls(fine, prev)
    assert result.max_abs_rad == pytest.approx(0.9, abs=1e-15)
    assert result.rms_rad == pytest.approx(np.sqrt(2.85 / (20 * CHANNELS)))


@pytest.mark.parametrize(
    "call_fine,call_prev,exc",
    [
        (np.full((20, CHANNELS), np.nan), np.zeros(CHANNELS), ValueError),
        (np.full((20, CHANNELS), np.inf), np.zeros(CHANNELS), ValueError),
        (_fine(20), np.array([np.inf] * CHANNELS), ValueError),
        (np.zeros((20, CHANNELS), dtype=bool), np.zeros(CHANNELS), TypeError),
        (_fine(20).astype(np.complex128), np.zeros(CHANNELS), TypeError),
        (np.array([["a"] * CHANNELS] * 20), np.zeros(CHANNELS), TypeError),
        (_fine(20), np.zeros(CHANNELS, dtype=bool), TypeError),
        (_fine(20)[:, :15], np.zeros(CHANNELS), ValueError),
        (_fine(20)[:, None, :], np.zeros(CHANNELS), ValueError),
        (_fine(25), np.zeros(CHANNELS), ValueError),  # N not divisible by 10
        (_fine(0), np.zeros(CHANNELS), ValueError),  # N = 0
        (_fine(20), np.zeros(15), ValueError),
        (_fine(20), np.zeros((1, CHANNELS)), ValueError),
    ],
)
def test_malformed_inputs_rejected(call_fine, call_prev, exc):
    with pytest.raises(exc):
        coarsen_native_controls(call_fine, call_prev)


@pytest.mark.parametrize(
    "coarse_in,prev_in,exc",
    [
        (np.zeros((2, 8)), np.zeros(CHANNELS), ValueError),
        (np.zeros((0, CHANNELS)), np.zeros(CHANNELS), ValueError),
        (np.full((2, CHANNELS), np.nan), np.zeros(CHANNELS), ValueError),
        (np.zeros((2, CHANNELS), dtype=np.complex64), np.zeros(CHANNELS), TypeError),
        (np.zeros((2, CHANNELS)), np.zeros(CHANNELS, dtype=bool), TypeError),
    ],
)
def test_interpolate_malformed_inputs_rejected(coarse_in, prev_in, exc):
    with pytest.raises(exc):
        interpolate_coarse_controls(coarse_in, prev_in)


def test_mutation_isolation_and_output_freshness():
    fine = _fine(30, seed=7)
    prev = np.linspace(-0.2, 0.2, CHANNELS)
    fine_copy = fine.copy()
    prev_copy = prev.copy()

    result = coarsen_native_controls(fine, prev)
    reconstructed = interpolate_coarse_controls(result.coarse, prev)

    np.testing.assert_array_equal(fine, fine_copy)
    np.testing.assert_array_equal(prev, prev_copy)
    for out in (result.coarse, reconstructed):
        assert out.dtype == np.float64
        assert out.flags["C_CONTIGUOUS"]
    # Outputs do not alias inputs: scribbling on them changes nothing there.
    coarse_copy = result.coarse.copy()
    result.coarse[:] = 123.0
    reconstructed[:] = -456.0
    np.testing.assert_array_equal(fine, fine_copy)
    np.testing.assert_array_equal(prev, prev_copy)
    # And the coarse view returned by coarsen is a fresh copy of fine[9::10].
    assert not np.shares_memory(result.coarse, fine)
    assert not np.shares_memory(reconstructed, result.coarse)
    result.coarse[:] = coarse_copy  # restore before metric assertions
    assert result.coarse_steps == 3
    assert result.coarse.shape == (3, CHANNELS)
