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


def test_initialize_state_kwarg_defaults_true():
    import inspect  # noqa: PLC0415

    sig = inspect.signature(core.build_full_update)
    param = sig.parameters["initialize_state"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is True


@pytest.mark.parametrize("bad_flag", ["false", 1, 0, None, object()])
def test_initialize_state_rejects_nonbool_before_model_access(bad_flag):
    # A non-bool flag must fail before heavy imports and before the model is
    # touched: a plain object() model would raise AttributeError from
    # nnx.split if validation did not come first. Runs without JAX too.
    with pytest.raises(TypeError):
        core.build_full_update(object(), initialize_state=bad_flag)


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


# --- initialize_state continuation admission (JAX-dependent, skipped locally) ---


def _tiny_model():
    import jax.numpy as jnp  # noqa: PLC0415
    from flax import nnx  # noqa: PLC0415

    class Tiny(nnx.Module):
        def __init__(self):
            self.w = nnx.Param(jnp.full((4, 4), 0.5, dtype=jnp.float32))
            self.b = nnx.Param(jnp.full((4,), 0.25, dtype=jnp.float32))

    return Tiny()


def _abstract_like(model):
    """Return an equivalent model whose param leaves are abstract FP32
    ShapeDtypeStruct stand-ins (nnx.eval_shape semantics), as used by the
    continuation-admission path."""
    import jax  # noqa: PLC0415
    from flax import nnx  # noqa: PLC0415

    graphdef, state = nnx.split(model)
    abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state)
    return nnx.merge(graphdef, abstract)


def test_initialize_state_false_skips_all_state_initializers(monkeypatch):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax.nnx")
    optax = pytest.importorskip("optax")
    pytest.importorskip("embodied_trackpad.leap_v1.pi05_jax_loss")

    def _fail_prng(*args, **kwargs):
        raise AssertionError("jax.random.PRNGKey called with initialize_state=False")

    def _fail_zeros(*args, **kwargs):
        raise AssertionError("jnp.zeros called with initialize_state=False")

    real_chain = optax.chain

    def _chain_forbidding_init(*args, **kwargs):
        tx = real_chain(*args, **kwargs)

        class _NoInitTx:
            def init(self, params):
                raise AssertionError("tx.init called with initialize_state=False")

            def update(self, updates, state, params=None):
                return tx.update(updates, state, params)

        return _NoInitTx()

    monkeypatch.setattr(jax.random, "PRNGKey", _fail_prng)
    monkeypatch.setattr(jnp, "zeros", _fail_zeros)
    monkeypatch.setattr(optax, "chain", _chain_forbidding_init)

    model = _abstract_like(_tiny_model())
    state, step_fn, census = core.build_full_update(model, seed=17, initialize_state=False)
    assert state is None
    assert callable(step_fn)
    assert census["leaf_count"] == 2
    assert census["total_params"] == 20
    assert census["all_fp32"] is True
    assert {leaf["path"] for leaf in census["leaves"]} == {"w", "b"}


def test_initialize_state_census_parity_and_default_state():
    pytest.importorskip("jax")
    pytest.importorskip("jax.numpy")
    pytest.importorskip("flax.nnx")
    pytest.importorskip("optax")
    pytest.importorskip("embodied_trackpad.leap_v1.pi05_jax_loss")

    model = _tiny_model()
    state_default, _, census_default = core.build_full_update(model, seed=17)
    state_explicit, _, census_explicit = core.build_full_update(model, seed=17, initialize_state=True)
    state_none, _, census_none = core.build_full_update(model, seed=17, initialize_state=False)
    assert state_none is None
    for st in (state_default, state_explicit):
        assert set(st) == {"params", "opt_state", "step", "rng"}
        assert int(st["step"]) == 0
    assert census_default == census_explicit == census_none


def test_initialize_state_false_both_image_layouts():
    pytest.importorskip("jax")
    pytest.importorskip("jax.numpy")
    pytest.importorskip("flax.nnx")
    pytest.importorskip("optax")
    pytest.importorskip("embodied_trackpad.leap_v1.pi05_jax_loss")

    model = _abstract_like(_tiny_model())
    for layout in ("legacy", "screen_pair"):
        state, step_fn, census = core.build_full_update(
            model, seed=17, image_layout=layout, initialize_state=False
        )
        assert state is None
        assert callable(step_fn)
        assert census["leaf_count"] == 2
        assert census["total_params"] == 20
    # Layout must not affect census on the state-initializing path either.
    _, _, census_legacy = core.build_full_update(_tiny_model(), image_layout="legacy")
    _, _, census_pair = core.build_full_update(_tiny_model(), image_layout="screen_pair")
    assert census_legacy == census_pair
