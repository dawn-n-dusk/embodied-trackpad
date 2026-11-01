"""Tests for pi05_checkpoint.

The frozen project runtime venv (``.runtime/.../venv-macos``) has NumPy and
pytest but no JAX/Orbax: module-contract, fail-closed-boundary, and NumPy
leaf-record checks run there (the boundary raises also prove fail-closed
behavior before any heavy import); the real tiny JAX/Orbax roundtrip skips
without jax+orbax. No fake JAX or fake full-model acceptance.
"""

from __future__ import annotations

import py_compile

import pytest

import embodied_trackpad.leap_v1.pi05_checkpoint as ckpt

PY_FILE = ckpt.__file__

GOOD_KEYS = {"params": 1, "opt_state": 2, "step": 3, "rng": 4}


def test_module_compiles():
    py_compile.compile(PY_FILE, doraise=True)


def test_module_imports_without_heavy_deps():
    assert callable(ckpt.abstract_state)
    assert callable(ckpt.save_state)
    assert callable(ckpt.restore_state)
    assert callable(ckpt.fingerprint_state)
    assert ckpt._STATE_KEYS == ("params", "opt_state", "step", "rng")


def test_require_state_dict_exact_four_keys():
    ckpt._require_state_dict(GOOD_KEYS)
    for bad in (
        {},
        {"params": 1, "opt_state": 2, "step": 3},
        {**GOOD_KEYS, "ema": 5},
        {**GOOD_KEYS, "full_model": 6},
        ("params", "opt_state", "step", "rng"),
    ):
        with pytest.raises(ValueError):
            ckpt._require_state_dict(bad)


def test_save_state_fail_closed_before_orbax(tmp_path):
    # Relative path and existing path must raise without ever importing Orbax;
    # on this runtime (no orbax installed) reaching these raises proves it.
    with pytest.raises(ValueError):
        ckpt.save_state(GOOD_KEYS, "relative/ckpt")
    existing = tmp_path / "already_here"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        ckpt.save_state(GOOD_KEYS, str(existing))
    with pytest.raises(ValueError):
        ckpt.save_state({"params": 1}, str(tmp_path / "x"))


def test_restore_state_fail_closed_before_orbax(tmp_path):
    with pytest.raises(FileNotFoundError):
        ckpt.restore_state(GOOD_KEYS, str(tmp_path / "missing"), device=None)
    with pytest.raises(ValueError):
        ckpt.restore_state({"params": 1}, str(tmp_path / "missing"), device=None)


def test_abstract_state_fail_closed_before_jax():
    with pytest.raises(ValueError):
        ckpt.abstract_state({"params": 1})


# --- NumPy leaf-record boundaries (skip without numpy) ---


def test_leaf_record_stable_and_value_sensitive():
    np = pytest.importorskip("numpy")
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    r1 = ckpt._leaf_record("params/w", a)
    r2 = ckpt._leaf_record("params/w", a.copy())
    assert r1 == r2
    b = a.copy()
    b[0, 0] = 99.0
    assert ckpt._leaf_record("params/w", b)["sha256"] != r1["sha256"]
    assert r1["numel"] == 6 and r1["bytes"] == 24
    assert r1["shape"] == [2, 3] and r1["dtype"] == "<f4"
    assert isinstance(r1["numel"], int) and isinstance(r1["bytes"], int)


def test_leaf_record_binds_path_shape_and_dtype():
    np = pytest.importorskip("numpy")
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    base = ckpt._leaf_record("params/w", a)["sha256"]
    assert ckpt._leaf_record("params/other", a)["sha256"] != base  # path
    assert ckpt._leaf_record("params/w", a.reshape(3, 2))["sha256"] != base  # shape
    assert ckpt._leaf_record("params/w", a.astype(np.float64))["sha256"] != base  # dtype
    assert ckpt._leaf_record("params/w", a.copy(order="C"))["sha256"] == base  # same bytes, C-contiguous
    with pytest.raises(BufferError):
        ckpt._leaf_record("params/w", a.T)  # helper precondition: caller passes C-contiguous


def test_leaf_record_scalar_shape_preserved():
    np = pytest.importorskip("numpy")
    scalar = np.asarray(np.float32(7.5))  # shape (), not promoted to (1,)
    vec = np.asarray([7.5], dtype=np.float32)
    rs = ckpt._leaf_record("step", scalar)
    rv = ckpt._leaf_record("step", vec)
    assert rs["shape"] == [] and rs["numel"] == 1 and rs["bytes"] == 4
    assert rv["shape"] == [1]
    assert rs["sha256"] != rv["sha256"]  # scalar vs 1-vector are distinct fingerprints


def test_leaf_record_rejects_object_dtype():
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError):
        ckpt._leaf_record("params/bad", np.array([{"a": 1}], dtype=object))


# --- Real tiny JAX/Orbax roundtrip (skip without jax+orbax) ---


def test_tiny_jax_orbax_roundtrip_and_fingerprint_parity(tmp_path):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("orbax.checkpoint")

    state = {
        "params": {"w": jnp.arange(6, dtype=jnp.float32).reshape(2, 3), "b": jnp.full((3,), 0.5)},
        "opt_state": {"count": jnp.zeros((), dtype=jnp.int32), "mu": jnp.ones((2,), dtype=jnp.float32)},
        "step": jnp.array(3, dtype=jnp.int32),
        "rng": jax.random.PRNGKey(17),
    }
    device = jax.devices("cpu")[0]
    template = ckpt.abstract_state(state)
    for leaf in jax.tree.leaves(template):
        assert isinstance(leaf, jax.ShapeDtypeStruct)

    fp_before = ckpt.fingerprint_state(state)
    ckpt_dir = tmp_path / "ckpt"
    info = ckpt.save_state(state, str(ckpt_dir))
    assert info["files"] > 0 and info["bytes"] > 0
    with pytest.raises(FileExistsError):
        ckpt.save_state(state, str(ckpt_dir))

    del state  # restore must work from the abstract template alone
    restored = ckpt.restore_state(template, str(ckpt_dir), device)
    fp_after = ckpt.fingerprint_state(restored)
    assert fp_after == fp_before  # exact payload+structure+path parity
    for leaf in jax.tree.leaves(restored):
        assert isinstance(leaf, jax.Array)
        assert leaf.sharding == jax.sharding.SingleDeviceSharding(device)

    bad = jax.tree.map(lambda s: jax.ShapeDtypeStruct((s.size,), s.dtype), template)
    with pytest.raises((ValueError, RuntimeError)):
        ckpt.restore_state(bad, str(ckpt_dir), device)
