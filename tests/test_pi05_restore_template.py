"""Tests for pi05_restore_template.

The foundation-only runtime has NumPy and pytest but no JAX/Flax/Optax:
module-contract, fail-closed-boundary, and guard checks run there; the real
tiny-JAX comparisons (traced abstract template vs. abstract of a real tiny
build_full_update state, including the BF16-cast behavior) skip without
jax+flax+optax. No fake JAX, no full model, no restore/weights I/O here.
"""

from __future__ import annotations

import py_compile

import pytest

import embodied_trackpad.leap_v1.pi05_restore_template as rt

PY_FILE = rt.__file__

STATE_KEYS = {"params", "opt_state", "step", "rng"}


def test_module_compiles():
    py_compile.compile(PY_FILE, doraise=True)


def test_module_imports_without_heavy_deps():
    assert callable(rt.build_abstract_update_state)
    assert rt._STATE_KEYS == ("params", "opt_state", "step", "rng")


def test_build_abstract_fail_closed_before_jax():
    with pytest.raises(TypeError):
        rt.build_abstract_update_state(create_model=42)


# --- real tiny-JAX checks (skip without jax+flax+optax) ---


def _tiny_factory(dtype):
    """Return create_model(init_key) building a tiny 2-leaf NNX model in dtype."""

    def create_model(init_key):
        import jax  # noqa: PLC0415
        import jax.numpy as jnp  # noqa: PLC0415
        from flax import nnx  # noqa: PLC0415

        class Tiny(nnx.Module):
            def __init__(self, rngs):
                self.w = nnx.Param(
                    jax.random.normal(rngs.default(), (6, 8), dtype=dtype)
                )
                self.b = nnx.Param(jnp.zeros((8,), dtype=dtype))

        return Tiny(nnx.Rngs(init_key))

    return create_model


def _real_state(factory, seed=17):
    import jax  # noqa: PLC0415

    from embodied_trackpad.leap_v1 import pi05_update_core as core  # noqa: PLC0415

    state, _step_fn, census = core.build_full_update(factory(jax.random.key(1)), seed=seed)
    return state, census


def test_abstract_template_matches_real_tiny_build():
    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    import jax.numpy as jnp  # noqa: PLC0415

    import embodied_trackpad.leap_v1.pi05_checkpoint as ckpt  # noqa: PLC0415

    factory = _tiny_factory(jnp.float32)
    real_state, census = _real_state(factory)
    template = rt.build_abstract_update_state(factory, seed=17)

    # exact four top-level keys, every leaf abstract
    assert set(template.keys()) == STATE_KEYS
    for leaf in jax.tree.leaves(template):
        assert isinstance(leaf, jax.ShapeDtypeStruct)

    # treedef / per-leaf path / shape / dtype equality against the real build
    want = ckpt.abstract_state(real_state)
    assert jax.tree.structure(template) == jax.tree.structure(want)
    t_leaves = jax.tree.leaves_with_path(template)
    w_leaves = jax.tree.leaves_with_path(want)
    assert len(t_leaves) == len(w_leaves)
    for (t_path, t_leaf), (w_path, w_leaf) in zip(t_leaves, w_leaves):
        assert t_path == w_path
        assert tuple(t_leaf.shape) == tuple(w_leaf.shape)
        assert t_leaf.dtype == w_leaf.dtype

    # caller-side census equivalence on the tiny model, params subtree only
    import math  # noqa: PLC0415

    p_leaves = jax.tree.leaves(template["params"])
    assert len(p_leaves) == census["leaf_count"]
    assert sum(math.prod(leaf.shape) for leaf in p_leaves) == census["total_params"]


def test_abstract_template_casts_bf16_params_to_fp32():
    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    import jax.numpy as jnp  # noqa: PLC0415

    from embodied_trackpad.leap_v1 import pi05_update_core as core  # noqa: PLC0415

    bf16_factory = _tiny_factory(jnp.bfloat16)

    # raw BF16 tiny model is rejected by the accepted core (dtype contract)
    with pytest.raises(ValueError):
        core.build_full_update(bf16_factory(jax.random.key(1)), seed=17)

    # traced path casts Param leaves to FP32 and still produces an exact template
    template = rt.build_abstract_update_state(bf16_factory, seed=17)
    assert set(template.keys()) == STATE_KEYS
    for leaf in jax.tree.leaves(template):
        assert isinstance(leaf, jax.ShapeDtypeStruct)
    p_dtype = template["params"].flat_state()
    for var in p_dtype.values():
        assert var.value.dtype == jnp.float32


def test_abstract_template_step_and_rng_dtypes():
    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    import jax.numpy as jnp  # noqa: PLC0415

    factory = _tiny_factory(jnp.float32)
    template = rt.build_abstract_update_state(factory, seed=17)
    assert template["step"].dtype == jnp.int32
    assert template["rng"].dtype == jnp.uint32
    assert tuple(template["rng"].shape) == (2,)
    assert tuple(template["step"].shape) == ()
