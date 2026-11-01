"""Focused pure-NumPy tests for the valid-16 loss helper."""

import numpy as np
import pytest

from embodied_trackpad.leap_v1.pi05_jax_loss import valid16_mse


def _base_arrays():
    rng = np.random.default_rng(0)
    predicted = rng.normal(size=(2, 3, 32))
    target = rng.normal(size=(2, 3, 32))
    return predicted, target


def test_padded16_errors_do_not_affect_loss():
    predicted, target = _base_arrays()
    expected = np.mean(np.square(predicted[..., :16] - target[..., :16]), axis=-1)
    modified = predicted.copy()
    modified[..., 16:] += 1e6  # arbitrarily large errors only in padded region
    np.testing.assert_allclose(valid16_mse(modified, target), expected)
    np.testing.assert_allclose(valid16_mse(predicted, target), expected)


def test_valid16_errors_do_affect_loss():
    predicted, target = _base_arrays()
    modified = predicted.copy()
    modified[..., :16] += 0.5
    result = valid16_mse(modified, target)
    baseline = valid16_mse(predicted, target)
    assert not np.allclose(result, baseline)
    expected = np.mean(np.square(modified[..., :16] - target[..., :16]), axis=-1)
    np.testing.assert_allclose(result, expected)


def test_preserves_batch_time_axes():
    predicted, target = _base_arrays()
    result = valid16_mse(predicted, target)
    assert result.shape == predicted.shape[:-1] == (2, 3)


def test_rejects_bad_last_dimension():
    predicted, target = _base_arrays()
    with pytest.raises(ValueError, match="last dimension"):
        valid16_mse(predicted[..., :16], target[..., :16])
    with pytest.raises(ValueError, match="last dimension"):
        valid16_mse(np.zeros((2, 3, 33)), np.zeros((2, 3, 33)))


def test_rejects_mismatched_shapes():
    predicted, target = _base_arrays()
    with pytest.raises(ValueError, match="shape mismatch"):
        valid16_mse(predicted, target[:1])
