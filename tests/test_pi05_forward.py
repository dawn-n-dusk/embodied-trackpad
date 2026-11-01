"""Focused pure-NumPy tests for the forward-only diagnostic helper.

JAX execution is intentionally not exercised here (marked unverified); these
tests cover ``summarize_forward_loss`` and the input-validation paths of
``run_forward_diagnostic`` that run before any deferred JAX/Flax import.
"""

import numpy as np
import pytest

from embodied_trackpad.leap_v1.pi05_forward import (
    run_forward_diagnostic,
    summarize_forward_loss,
)


def _good_loss():
    rng = np.random.default_rng(7)
    return rng.normal(size=(1, 25)).astype(np.float32)


def _stub_bundle(mask=None):
    import types

    if mask is None:
        mask = np.zeros(32, dtype=bool)
        mask[:16] = True
    return types.SimpleNamespace(loss_mask32=mask, observation=None, actions=None)


class TestSummarizeForwardLoss:
    def test_aggregates_mean_min_max(self):
        loss = np.arange(25, dtype=np.float32).reshape(1, 25)
        stats = summarize_forward_loss(loss)
        assert stats["mean"] == pytest.approx(12.0)
        assert stats["min"] == pytest.approx(0.0)
        assert stats["max"] == pytest.approx(24.0)

    def test_rejects_wrong_shapes(self):
        for bad in (np.zeros(25), np.zeros((2, 25)), np.zeros((1, 24)), np.zeros((1, 25, 1))):
            with pytest.raises(ValueError, match="shape"):
                summarize_forward_loss(bad)

    def test_rejects_nonfinite_and_preserves_input(self):
        loss = _good_loss()
        original = loss.copy()
        for bad_value in (np.nan, np.inf, -np.inf):
            broken = loss.copy()
            broken[0, 3] = bad_value
            with pytest.raises(ValueError, match="finite"):
                summarize_forward_loss(broken)
        np.testing.assert_array_equal(loss, original)


class TestRunForwardDiagnosticValidation:
    def test_rejects_bad_loss_masks_before_jax(self):
        trailing_true = np.zeros(32, dtype=bool)
        trailing_true[16:] = True
        missing_true = np.zeros(32, dtype=bool)
        missing_true[:15] = True
        wrong_dtype = np.ones(32, dtype=np.float32)
        wrong_shape = np.zeros(16, dtype=bool)
        for bad_mask in (trailing_true, missing_true, wrong_dtype, wrong_shape):
            with pytest.raises(ValueError, match="loss_mask32"):
                run_forward_diagnostic(model=None, bundles=[_stub_bundle(bad_mask)], seeds=(17,))

    def test_accepts_contract_mask_but_rejects_length_mismatch(self):
        bundles = [_stub_bundle(), _stub_bundle()]
        with pytest.raises(ValueError, match="seeds length"):
            run_forward_diagnostic(model=None, bundles=bundles, seeds=(17,))
        with pytest.raises(ValueError, match="seeds length"):
            run_forward_diagnostic(model=None, bundles=bundles, seeds=(17, 18, 19))

    def test_rejects_empty_bundles(self):
        with pytest.raises(ValueError, match="nonempty"):
            run_forward_diagnostic(model=None, bundles=[], seeds=())
