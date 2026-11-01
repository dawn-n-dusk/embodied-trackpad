"""Focused tests for the shared trace-safe image-layout module and loss wiring.

Dependency doubles (clearly labelled): the local venv has numpy only, so
``jax`` / ``jax.numpy`` / ``openpi.models.{model,pi0}`` are replaced with
recording in-process doubles injected via ``sys.modules``. These verify
dispatch/wiring only — they are NOT a substitute for the real OpenPI/JAX
runtime gate, which the coordinator binds separately.
"""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

from embodied_trackpad.leap_v1 import pi05_image_layout as L
from embodied_trackpad.leap_v1.pi05_jax_loss import pi05_valid16_flow_loss, valid16_mse

PAIR = ("base_0_rgb", "screen_local_0_rgb")


class _Obs:
    def __init__(self, images, masks):
        self.images = images
        self.image_masks = masks
        self.state = np.zeros((1, 32), np.float32)


def _obs(keys=PAIR):
    return _Obs({k: object() for k in keys}, {k: object() for k in keys})


def _install_openpi_double(monkeypatch):
    """Recording double for official preprocess_observation."""
    calls = []
    mod = types.ModuleType("openpi.models.model")

    def preprocess_observation(rng, observation, *, train, image_keys=None):
        calls.append({"rng": rng, "train": train, "image_keys": image_keys})
        return observation

    mod.preprocess_observation = preprocess_observation
    pkg = types.ModuleType("openpi")
    models = types.ModuleType("openpi.models")
    models.model = mod
    monkeypatch.setitem(sys.modules, "openpi", pkg)
    monkeypatch.setitem(sys.modules, "openpi.models", models)
    monkeypatch.setitem(sys.modules, "openpi.models.model", mod)
    return calls


def test_layout_validation_pure_python_before_heavy_imports():
    assert L.validate_image_layout("legacy") == "legacy"
    assert L.validate_image_layout("screen_pair") == "screen_pair"
    for bad in ("screen-pair", "LEGACY", "", None, 3):
        with pytest.raises(ValueError):
            L.validate_image_layout(bad)
    # module imported without jax/openpi present
    assert "jax" not in sys.modules or "numpy" in sys.modules


def test_legacy_default_parity_no_image_keys(monkeypatch):
    calls = _install_openpi_double(monkeypatch)
    obs = _obs(("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"))
    out = L.preprocess_observation_for_layout("RNG", obs, train=False, image_layout="legacy")
    assert out is obs and calls == [{"rng": "RNG", "train": False, "image_keys": None}]


def test_screen_pair_explicit_keys_passthrough(monkeypatch):
    calls = _install_openpi_double(monkeypatch)
    obs = _obs()
    out = L.preprocess_observation_for_layout(None, obs, train=False, image_layout="screen_pair")
    assert out is obs
    assert calls[0]["image_keys"] == L.SCREEN_PAIR_KEYS


@pytest.mark.parametrize("bad", [
    {PAIR[0]: 1},  # missing
    {PAIR[0]: 1, PAIR[1]: 2, "left_wrist_0_rgb": 3},  # extra wrist alias
    {PAIR[1]: 1, PAIR[0]: 2},  # reordered
])
def test_screen_pair_rejects_bad_key_sets(monkeypatch, bad):
    _install_openpi_double(monkeypatch)
    with pytest.raises(ValueError):
        L.preprocess_observation_for_layout(None, _obs(tuple(bad)), train=False, image_layout="screen_pair")


class _NoArray:  # tracer-like: any host coercion explodes
    def __array__(self, *a, **k):
        raise AssertionError("host-side np.asarray on tracer-like value")

    def __bool__(self):
        raise AssertionError("host-side Python bool on tracer-like value")


def test_no_host_coercion_of_tracer_like_values(monkeypatch):
    calls = _install_openpi_double(monkeypatch)
    obs = _obs()
    obs.images[PAIR[0]] = _NoArray()
    obs.image_masks[PAIR[1]] = _NoArray()
    out = L.preprocess_observation_for_layout(None, obs, train=False, image_layout="screen_pair")
    assert out.images[PAIR[0]] is obs.images[PAIR[0]]  # untouched identity, key metadata only
    assert out.image_masks[PAIR[1]] is obs.image_masks[PAIR[1]]
    assert calls[0]["image_keys"] == L.SCREEN_PAIR_KEYS


@pytest.mark.parametrize("bad_masks", [
    {PAIR[0]: True},  # missing
    {PAIR[0]: True, PAIR[1]: True, "left_wrist_0_rgb": False},  # extra wrist alias
    {PAIR[1]: True, PAIR[0]: True},  # reordered
])
def test_screen_pair_rejects_bad_mask_key_sets(monkeypatch, bad_masks):
    """Images hold the exact valid key set; only the MASK mapping is wrong."""
    _install_openpi_double(monkeypatch)
    obs = _Obs({k: object() for k in PAIR}, bad_masks)
    with pytest.raises(ValueError):
        L.preprocess_observation_for_layout(None, obs, train=False, image_layout="screen_pair")


def _install_jax_loss_doubles(monkeypatch):
    """Doubles for jax/jnp/openpi.models.pi0 sufficient for the loss body."""
    calls = _install_openpi_double(monkeypatch)
    jnp = types.ModuleType("jax.numpy")
    jnp.mean = np.mean
    jnp.square = np.square
    jnp.concatenate = np.concatenate
    jnp.cumsum = np.cumsum
    jax = types.ModuleType("jax")
    jax.numpy = jnp
    jax.random = types.SimpleNamespace(
        split=lambda rng, n=3: tuple(f"key{i}" for i in range(n)),
        normal=lambda rng, shape: np.zeros(shape, np.float32),
        beta=lambda rng, a, b, shape: np.full(shape, 0.5, np.float32),
    )
    pi0 = types.ModuleType("openpi.models.pi0")
    pi0.make_attn_mask = lambda m, ar: m
    pkg = sys.modules["openpi.models"]
    pkg.pi0 = pi0
    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", jnp)
    monkeypatch.setitem(sys.modules, "openpi.models.pi0", pi0)

    class _LLM:
        def __call__(self, xs, **kw):
            b = xs[0].shape[0] if xs[0] is not None else xs[1].shape[0]
            suffix = np.zeros((b, 25, 8), np.float32)
            return (None, suffix), None

    model = types.SimpleNamespace(
        action_horizon=25,
        embed_prefix=lambda o: (np.ones((1, 3, 8), np.float32), np.ones((1, 3), bool), np.zeros((3,), bool)),
        embed_suffix=lambda o, x_t, t: (np.ones((1, 25, 8), np.float32), np.ones((1, 25), bool),
                                        np.zeros((25,), bool), None),
        PaliGemma=types.SimpleNamespace(llm=_LLM()),
        action_out_proj=lambda x: np.zeros((1, 25, 32), np.float32),
    )
    return calls, model


def test_loss_legacy_wiring_default_preprocess(monkeypatch):
    calls, model = _install_jax_loss_doubles(monkeypatch)
    obs = _obs(("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"))
    actions = (np.arange(25 * 32, dtype=np.float32).reshape(25, 32) % 7)[None]
    loss = pi05_valid16_flow_loss(model, "rng", obs, actions)
    assert calls[0]["image_keys"] is None
    expected = np.mean(np.square(actions[..., :16]), axis=-1)  # per-(batch,horizon) valid-16 values
    assert np.allclose(loss, expected, atol=1e-6)


def test_loss_screen_pair_wiring_explicit_keys(monkeypatch):
    calls, model = _install_jax_loss_doubles(monkeypatch)
    obs = _obs()
    actions = np.ones((1, 25, 32), np.float32)
    loss = pi05_valid16_flow_loss(model, "rng", obs, actions, image_layout="screen_pair")
    assert calls[0]["image_keys"] == L.SCREEN_PAIR_KEYS
    assert np.allclose(loss, 1.0, atol=1e-6)


def test_loss_unknown_layout_rejected_before_heavy_import():
    with pytest.raises(ValueError):
        pi05_valid16_flow_loss(object(), "rng", _obs(), np.zeros((25, 32), np.float32),
                               image_layout="bogus")


def test_update_core_rejects_unknown_layout_before_heavy_import():
    from embodied_trackpad.leap_v1.pi05_update_core import build_full_update

    with pytest.raises(ValueError):
        build_full_update(object(), 17, image_layout="bogus")
    # keyword-only default keeps the legacy 2-positional-arg call shape
    import inspect

    sig = inspect.signature(build_full_update)
    assert sig.parameters["image_layout"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["image_layout"].default == "legacy"


def test_valid16_mse_reduction_unchanged():
    a = np.ones((2, 3, 32), np.float32)
    b = np.zeros((2, 3, 32), np.float32)
    b[..., 16:] = 1e6  # padded channels must not leak into the loss
    assert np.allclose(valid16_mse(a, b), np.ones((2, 3)))
