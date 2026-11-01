"""Focused tests for pi05_sample_actions with clearly labelled dependency doubles.

The local venv has numpy only (no JAX/OpenPI), so ``jax``/``jax.numpy``/
``openpi.models.pi0`` are replaced with recording in-process doubles. These
tests verify integration-step counting, provided/generated noise, attention
mask routing, and layout dispatch — NOT sampler precision. Real JAX runtime
acceptance is bound separately by the coordinator.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from embodied_trackpad.leap_v1.pi05_sampling import pi05_sample_actions
from embodied_trackpad.leap_v1.pi05_image_layout import SCREEN_PAIR_KEYS


class _Obs:
    def __init__(self, keys=SCREEN_PAIR_KEYS):
        self.images = {k: np.zeros((1, 224, 224, 3), np.float32) for k in keys}
        self.image_masks = {k: np.ones((1,), bool) for k in keys}
        self.state = np.zeros((1, 32), np.float32)


class _LLM:
    def __init__(self, recorder, suffix_v=0.0):
        self.rec = recorder
        self.suffix_v = suffix_v

    def __call__(self, xs, **kw):
        self.rec["llm_calls"].append({
            "has_prefix": xs[0] is not None,
            "mask": np.asarray(kw["mask"]).copy() if "mask" in kw else None,
            "positions": np.asarray(kw["positions"]).copy() if "positions" in kw else None,
            "kv_cache": kw.get("kv_cache"),
            "adarms_cond": kw.get("adarms_cond"),
        })
        x = xs[0] if xs[0] is not None else xs[1]
        b = x.shape[0]
        if xs[0] is not None:  # prefix pass fills KV cache, no suffix out
            return (None, None), ("kv", len(self.rec["llm_calls"]))
        suffix = np.full((b, 25, 8), self.suffix_v, np.float32)
        return (None, suffix), None


def _install_doubles(monkeypatch, suffix_v=0.0):
    rec = {"llm_calls": [], "attn": [], "times": [], "pre": []}
    jnp = types.ModuleType("jax.numpy")
    for name in ("mean", "square", "concatenate", "cumsum", "sum", "broadcast_to", "ones", "zeros"):
        setattr(jnp, name, getattr(np, name))
    jnp.broadcast_to = np.broadcast_to

    def _while_loop(cond, step, carry):
        n = 0
        while cond(carry):
            carry = step(carry)
            n += 1
            assert n < 10_000
        rec["steps"] = n
        return carry

    jax = types.ModuleType("jax")
    jax.numpy = jnp
    jax.random = types.SimpleNamespace(
        normal=lambda rng, shape: np.full(shape, 0.25, np.float32),
    )
    jax.lax = types.SimpleNamespace(while_loop=_while_loop)
    pi0 = types.ModuleType("openpi.models.pi0")

    def _make_attn_mask(mask, ar):
        m = np.asarray(mask)
        b, n = m.shape
        rec["attn"].append((m.copy(), np.asarray(ar).copy()))
        return np.broadcast_to(m[:, None, :], (b, n, n))  # (b, q, kv) like upstream

    pi0.make_attn_mask = _make_attn_mask
    model_mod = types.ModuleType("openpi.models.model")

    def _pre(rng, obs, *, train, image_keys=None):
        rec["pre"].append({"train": train, "image_keys": image_keys})
        return obs

    model_mod.preprocess_observation = _pre
    pkg = types.ModuleType("openpi")
    models = types.ModuleType("openpi.models")
    models.pi0 = pi0
    models.model = model_mod
    pkg.models = models
    for name, mod in (("openpi", pkg), ("openpi.models", models),
                      ("openpi.models.pi0", pi0), ("openpi.models.model", model_mod),
                      ("jax", jax), ("jax.numpy", jnp)):
        monkeypatch.setitem(sys.modules, name, mod)

    model = types.SimpleNamespace(
        action_horizon=25,
        action_dim=32,
        embed_prefix=lambda o: (np.ones((1, 3, 8), np.float32), np.ones((1, 3), bool),
                                np.zeros((1, 3), bool)),
        embed_suffix=lambda o, x_t, t: (rec["times"].append(float(np.asarray(t)[0])),
                                        (np.ones((1, 25, 8), np.float32),
                                         np.ones((1, 25), bool), np.zeros((1, 25), bool), None))[1],
        PaliGemma=types.SimpleNamespace(llm=_LLM(rec, suffix_v)),
        action_out_proj=(lambda v: lambda x: np.full((1, 25, 32), v, np.float32))(suffix_v),
    )
    return rec, model


def test_provided_noise_step_count_and_shape(monkeypatch):
    rec, model = _install_doubles(monkeypatch)
    noise = np.full((1, 25, 32), 3.0, np.float32)
    out = pi05_sample_actions(model, "rng", _Obs(), num_steps=4, noise=noise)
    assert rec["steps"] == 4
    assert out.shape == (1, 25, 32)
    assert np.array_equal(out, noise)  # v_t == 0 -> pure integration of zero drift
    assert rec["times"][:4] == [1.0, 0.75, 0.5, 0.25]  # dt=-1/4, threshold time>=-dt/2
    assert rec["pre"][0]["image_keys"] is None  # legacy default unchanged


def test_generated_noise_and_constant_drift(monkeypatch):
    rec, model = _install_doubles(monkeypatch, suffix_v=1.0)
    out = pi05_sample_actions(model, "rng", _Obs(), num_steps=10)
    # x0 = noise + num_steps*dt*1.0 = 0.25 - 1.0 (full 32-channel output)
    assert rec["steps"] == 10
    assert np.allclose(out, 0.25 - 1.0)
    assert out.shape == (1, 25, 32)


def test_screen_pair_layout_routes_explicit_keys(monkeypatch):
    rec, model = _install_doubles(monkeypatch)
    pi05_sample_actions(model, "rng", _Obs(), num_steps=2, image_layout="screen_pair")
    assert rec["pre"][0]["image_keys"] == SCREEN_PAIR_KEYS


def test_mask_and_kv_routing(monkeypatch):
    """Nontrivial masks: prefix [T,F,T], suffix last token masked (LABELLED DOUBLE:
    records routing of real arrays through the sampler body; not actual JAX)."""
    rec, model = _install_doubles(monkeypatch)
    prefix_mask = np.array([[True, False, True]])
    suffix_mask = np.array([[(True)] * 24 + [False]])
    model.embed_prefix = lambda o: (
        np.ones((1, 3, 8), np.float32), prefix_mask.copy(), np.zeros((1, 3), bool))
    model.embed_suffix = lambda o, x_t, t: (
        np.ones((1, 25, 8), np.float32), suffix_mask.copy(), np.zeros((1, 25), bool), "ADARMS")
    pi05_sample_actions(model, "rng", _Obs(), num_steps=2,
                        noise=np.zeros((1, 25, 32), np.float32))
    pre = rec["llm_calls"][0]
    assert pre["has_prefix"] is True and pre["kv_cache"] is None
    assert np.array_equal(pre["positions"], [[0, 0, 1]])  # cumsum(prefix)-1
    kv_expected = ("kv", 1)  # exact object returned by the prefix pass
    calls = rec["llm_calls"][1:]
    assert len(calls) == 2
    for c in calls:
        assert c["has_prefix"] is False
        assert c["kv_cache"] == kv_expected  # exact prefix KV object reused every step
        assert c["adarms_cond"] == [None, "ADARMS"]
        m = c["mask"]  # (b, suffix_len, prefix_len + suffix_len)
        assert m.shape == (1, 25, 28)
        assert not bool(m[0, :, 1].any())  # invalid prefix column stays False everywhere
        assert bool(m[0, :24, 0].all()) and bool(m[0, :24, 2].all())
        expected_pos = 2 + np.cumsum(suffix_mask, axis=-1) - 1  # 2 valid prefix tokens
        assert np.array_equal(c["positions"], expected_pos)


@pytest.mark.parametrize("bad", [True, 0, -3, 2.5, "10"])
def test_num_steps_must_be_positive_python_int(monkeypatch, bad):
    _rec, model = _install_doubles(monkeypatch)
    with pytest.raises(ValueError):
        pi05_sample_actions(model, "rng", _Obs(), num_steps=bad)


def test_unknown_layout_rejected():
    with pytest.raises(ValueError):
        pi05_sample_actions(object(), "rng", _Obs(), image_layout="nope")
