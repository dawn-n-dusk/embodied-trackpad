"""Tests for pi05_update_core.

The local frozen runtime has no JAX, so JAX-dependent tests use
``pytest.importorskip`` and are reported as skipped here; runtime verification
of the compiled step is a separate later stage. Pure checks (module contract,
optimizer constants, py_compile) always run.
"""

from __future__ import annotations

import py_compile

import pytest

import embodied_trackpad.leap_v1.pi05_update_core as core

PY_FILE = core.__file__


def test_module_compiles():
    py_compile.compile(PY_FILE, doraise=True)


def test_module_imports_without_jax():
    # Heavy deps must be deferred: importing this module must not need jax/flax/optax.
    assert callable(core.build_full_update)
    assert core._ACTIONS_SHAPE == (1, 25, 32)


def test_optimizer_constants_are_full_finetune_adamw():
    kw = core._OPTIMIZER_KW
    assert kw == {"lr": 1e-5, "b1": 0.9, "b2": 0.95, "eps": 1e-8, "weight_decay": 0.0, "clip_norm": 1.0}


def test_leaf_path_rendering():
    assert core._leaf_path(("PaliGemma", "llm", "layers", 0, "kernel")) == "PaliGemma/llm/layers/0/kernel"


# --- JAX-dependent behavior on an explicitly tiny NNX model (skipped locally) ---


def _copy_tree(tree):
    import jax  # noqa: PLC0415

    return jax.tree.map(lambda x: x.copy(), tree)


def test_tiny_model_finite_update_and_nonfinite_rejection(monkeypatch):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    nnx = pytest.importorskip("flax.nnx")
    pytest.importorskip("optax")
    loss_mod = pytest.importorskip("embodied_trackpad.leap_v1.pi05_jax_loss")

    class Tiny(nnx.Module):
        def __init__(self):
            self.w = nnx.Param(jnp.full((4, 4), 0.5, dtype=jnp.float32))
            self.b = nnx.Param(jnp.full((4,), 0.25, dtype=jnp.float32))

    model = Tiny()
    # Small real NNX loss on nonzero data; the observation argument carries a
    # traced 0/1 flag that selects the finite vs NaN branch dynamically, so a
    # single compiled step covers both paths. Real Pi0 loss verification is a
    # later stage; the immutable helper function object is untouched.
    def tiny_loss(mdl, rng, observation, actions):
        del rng
        x = actions[0, 0, :4] + observation  # nonzero data -> nonzero grads
        val = jnp.mean(jnp.square(mdl.w @ x + mdl.b))
        return jnp.where(observation == 0, jnp.full((1, 25), val), jnp.full((1, 25), jnp.nan))

    monkeypatch.setattr(loss_mod, "pi05_valid16_flow_loss", tiny_loss)

    state, step_fn, census = core.build_full_update(model, seed=17)
    assert census["leaf_count"] == 2
    assert census["total_params"] == 20
    assert census["all_fp32"] is True
    assert {leaf["path"] for leaf in census["leaves"]} == {"w", "b"}

    actions = jnp.ones((1, 25, 32), dtype=jnp.float32)
    finite_flag = jnp.zeros(())
    nan_flag = jnp.ones(())

    snapshot0 = _copy_tree(state)
    new_state, metrics = step_fn(state, finite_flag, actions)  # donates `state`
    jax.block_until_ready(metrics["loss"])
    assert metrics["loss"].shape == ()
    assert metrics["loss"].dtype == jnp.float32
    assert float(metrics["accepted"]) == 1.0
    assert int(new_state["step"]) == 1
    assert float(metrics["update_norm"]) > 0.0
    assert float(metrics["param_delta_norm"]) > 0.0
    assert set(metrics["grad_norms"]) == {"w", "b"}
    assert set(metrics["grad_nonfinite_counts"]) == {"w", "b"}
    changed = jax.tree.map(jnp.logical_not, jax.tree.map(jnp.array_equal, new_state["params"], snapshot0["params"]))
    assert any(jax.tree.leaves(changed))  # finite update changed real params

    snapshot1 = _copy_tree(new_state)
    new_state2, metrics2 = step_fn(new_state, nan_flag, actions)  # donates `new_state`
    jax.block_until_ready(metrics2["loss"])
    assert float(metrics2["accepted"]) == 0.0
    assert float(metrics2["update_norm"]) == 0.0
    assert float(metrics2["param_delta_norm"]) == 0.0
    # Rejection must restore params, opt_state, step and RNG exactly.
    for key in ("params", "opt_state", "step", "rng"):
        same = jax.tree.map(jnp.array_equal, new_state2[key], snapshot1[key])
        assert all(jax.tree.leaves(same)), key
