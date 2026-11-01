"""Checkpoint I/O and streaming fingerprint for the reviewed pi0.5 full-update state.

The state dict (``pi05_update_core.build_full_update``) has exactly four keys:
``params`` (flax ``nnx.State`` of ``nnx.Param`` VariableState leaves),
``opt_state`` (optax tuple pytree), ``step`` (int32 scalar), ``rng`` (uint32
PRNG key). Params and Adam moments are FP32; ``step``/``rng`` are integer
arrays (int32 / uint32) and are the explicit non-FP32 exceptions. No EMA and
no second full-model copy: any state dict beyond exactly these four keys is
rejected, so nothing here duplicates the model on disk.

Pinned runtime APIs (source-attributed, Orbax 0.11.13 installed read):
- JAX 0.5.3: ``jax.tree.map``/``jax.tree.leaves_with_path``/``jax.tree.structure``,
  ``jax.ShapeDtypeStruct``, ``jax.device_get``, ``jax.sharding.SingleDeviceSharding``.
- Orbax 0.11.13: ``ocp.PyTreeCheckpointer``; ``ocp.args.PyTreeSave`` /
  ``ocp.args.PyTreeRestore``; ``ocp.ArrayRestoreArgs`` (exported at the
  ``orbax.checkpoint`` top level in 0.11.13 — ``ocp.args.ArrayRestoreArgs``
  does NOT exist; ``ocp.args`` only carries the shorter alias).
  Save/restore are called with the explicit ``args=`` keyword.

JAX/Orbax/NumPy imports are deferred into function bodies, so this module
imports on the foundation-only runtime and all fail-closed validation runs
before any heavy import.

``fingerprint_state`` is a full-payload hash: it reads every array leaf of
the whole four-key state once through device-to-host transfer. For the real
full-parameter state that is a ~40 GiB read, not a cheap metadata hash. It
streams one leaf at a time, retains only small metadata records, and never
writes a second weights copy. Releasing each local host reference between
leaves does NOT bound total host RSS: the pinned JAX 0.5.3 ``ArrayImpl`` may
keep its own host copy of an array after a device-to-host read, and this
one-leaf-at-a-time loop cannot free any JAX-owned cache.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

__all__ = ["abstract_state", "save_state", "restore_state", "fingerprint_state"]

_STATE_KEYS = ("params", "opt_state", "step", "rng")


def _require_state_dict(state: Any) -> None:
    keys = tuple(state.keys()) if isinstance(state, dict) else ()
    if len(keys) != 4 or set(keys) != set(_STATE_KEYS):
        raise ValueError(f"state must have exactly the four keys {_STATE_KEYS}, got {keys}")


def _render_path(path: tuple) -> str:
    return "/".join(str(p) for p in path)


def _close(checkpointer: Any) -> None:
    close = getattr(checkpointer, "close", None)
    if callable(close):
        close()


def abstract_state(state: Any) -> Any:
    """Return a device-free ``jax.ShapeDtypeStruct`` template of ``state``.

    Retains the pytree/NNX structure and per-leaf shape/dtype metadata but no
    device arrays, so the caller may release the full state before restoring.
    """
    _require_state_dict(state)  # fail closed before any heavy import
    import jax  # noqa: PLC0415 - deferred heavy import

    return jax.tree.map(lambda a: jax.ShapeDtypeStruct(tuple(a.shape), a.dtype), state)


def save_state(state: Any, path: str) -> dict:
    """Synchronously save ``state`` to a fresh, non-existing absolute ``path``.

    Refuses overwrite/deletion: anything at ``path`` already is an error.
    Returns the compact actually-created file count/bytes (no copies made).
    Raises on any Orbax failure; the checkpointer is closed either way.
    """
    _require_state_dict(state)
    path = Path(path)
    if not path.is_absolute():
        raise ValueError(f"checkpoint path must be absolute, got {path}")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint path {path}")
    import orbax.checkpoint as ocp  # noqa: PLC0415

    checkpointer = ocp.PyTreeCheckpointer()
    try:
        checkpointer.save(str(path), args=ocp.args.PyTreeSave(item=state))
    finally:
        _close(checkpointer)
    files = [p for p in path.rglob("*") if p.is_file()]
    return {"path": str(path), "files": len(files), "bytes": sum(int(p.stat().st_size) for p in files)}


def restore_state(template: Any, path: str, device: Any) -> Any:
    """Restore exactly the abstract ``template`` from ``path`` onto ``device``.

    Every template leaf must be a ``jax.ShapeDtypeStruct`` (checked before
    Orbax). After restore, the returned treedef and every leaf shape/dtype
    are compared exactly against the template and any mismatch raises — Orbax
    rejection is not assumed to catch every shape/dtype change. No random
    fallback and no model construction happen here.
    """
    _require_state_dict(template)
    path = Path(path)
    if not path.is_absolute() or not path.is_dir():
        raise FileNotFoundError(f"checkpoint path {path} is not an existing directory")
    import jax  # noqa: PLC0415
    import orbax.checkpoint as ocp  # noqa: PLC0415

    template_leaves = jax.tree.leaves_with_path(template)
    for leaf_path, leaf in template_leaves:
        if not isinstance(leaf, jax.ShapeDtypeStruct):
            raise ValueError(
                f"template leaf {_render_path(leaf_path)} is {type(leaf).__name__}, expected jax.ShapeDtypeStruct"
            )
    restore_args = jax.tree.map(
        lambda _: ocp.ArrayRestoreArgs(restore_type=jax.Array, sharding=jax.sharding.SingleDeviceSharding(device)),
        template,
    )
    checkpointer = ocp.PyTreeCheckpointer()
    try:
        restored = checkpointer.restore(
            str(path), args=ocp.args.PyTreeRestore(item=template, restore_args=restore_args)
        )
    finally:
        _close(checkpointer)
    if jax.tree.structure(restored) != jax.tree.structure(template):
        raise ValueError("restored state pytree structure does not match the abstract template")
    for (leaf_path, want), got in zip(jax.tree.leaves_with_path(template), jax.tree.leaves(restored)):
        if tuple(got.shape) != tuple(want.shape) or got.dtype != want.dtype:
            raise ValueError(
                f"restored leaf {_render_path(leaf_path)} has shape {tuple(got.shape)}/{got.dtype}, "
                f"expected {tuple(want.shape)}/{want.dtype}"
            )
    return restored


def _leaf_record(path: str, n: Any) -> dict:
    """Hash one host array with its exact path/shape/dtype (pure NumPy).

    ``n`` must be a C-contiguous non-object numeric ndarray; object dtype is
    rejected because its buffered bytes are not a faithful payload encoding.
    """
    dt = n.dtype
    if dt.hasobject:
        raise ValueError(f"leaf {path} has object dtype {dt}; only numeric array leaves are fingerprintable")
    h = hashlib.sha256()
    header = f"{path}|{tuple(n.shape)}|{dt.str}|{int(n.nbytes)}"
    h.update(header.encode("utf-8"))
    h.update(memoryview(n))
    return {
        "path": path,
        "shape": [int(s) for s in n.shape],
        "dtype": dt.str,
        "numel": int(n.size),
        "bytes": int(n.nbytes),
        "sha256": h.hexdigest(),
    }


def fingerprint_state(state: Any, *, progress: Any = None) -> dict:
    """Streaming SHA-256 fingerprint of the full four-key state payload.

    Iterates ``jax.tree.leaves_with_path``: one device-to-host transfer, one
    local host copy (contiguity fixed only when needed, scalar shape ``()``
    kept), bytes folded into the global digest, then the local host copy is
    released before the next leaf. Releasing that local reference does not
    bound total host RSS: JAX may retain its own host copy of the array after
    the device-to-host read. The global digest binds the pytree structure
    string (``str(jax.tree.structure(state))``), every leaf path, shape,
    dtype and byte payload, plus totals. Totals are Python ints, so no int32
    overflow. This reads the whole payload (~40 GiB at full scale); no leaf is
    skipped or sampled.

    ``progress`` is an optional callable invoked with one dict per event, for
    observability only. Stages: ``before-transfer`` (emitted before every
    device-to-host transfer), ``after-transfer`` (after the host copy of that
    leaf exists), ``after-hash`` (after that leaf's bytes were folded into the
    digest), and ``complete`` (once, after all leaves). Each event carries
    ``stage``, ``leaf_index``, ``leaf_count``, ``path``, ``shape``,
    ``dtype`` (NumPy ``.str`` form), ``nbytes`` (expected leaf bytes),
    ``t_s`` (monotonic seconds since fingerprint start), plus cumulative
    ``transferred_bytes`` and ``hashed_bytes``. Events do not affect the
    digest or the returned record; a raising callback propagates visibly.
    """
    _require_state_dict(state)
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import time  # noqa: PLC0415

    h = hashlib.sha256()

    def _feed(tag: str, blob: bytes) -> None:
        h.update(tag.encode("ascii"))
        h.update(len(blob).to_bytes(8, "big"))
        h.update(blob)

    _feed("struct", str(jax.tree.structure(state)).encode("utf-8"))
    leaves_with_path = jax.tree.leaves_with_path(state)
    leaf_count = len(leaves_with_path)
    t0 = time.monotonic()
    transferred = 0
    hashed = 0

    def _emit(stage: str, index: int, path: str, shape: tuple, dtype: Any, nbytes: int, **extra: Any) -> None:
        if progress is None:
            return
        event = {
            "stage": stage,
            "leaf_index": index,
            "leaf_count": leaf_count,
            "path": path,
            "shape": list(shape),
            "dtype": np.dtype(dtype).str,
            "nbytes": int(nbytes),
            "t_s": round(time.monotonic() - t0, 6),
            "transferred_bytes": int(transferred),
            "hashed_bytes": int(hashed),
        }
        event.update(extra)
        progress(event)  # any callback exception propagates out of fingerprint_state

    leaves: list[dict] = []
    numel = 0
    nbytes = 0
    for index, (path, leaf) in enumerate(leaves_with_path):
        rel = _render_path(path)
        shape = tuple(int(s) for s in leaf.shape)
        leaf_bytes = int(leaf.nbytes)
        _emit("before-transfer", index, rel, shape, leaf.dtype, leaf_bytes)
        host = np.asarray(jax.device_get(leaf))
        if not host.flags.c_contiguous:
            host = np.ascontiguousarray(host)
        transferred += int(host.nbytes)
        _emit("after-transfer", index, rel, shape, leaf.dtype, leaf_bytes)
        rec = _leaf_record(rel, host)
        hashed += int(rec["bytes"])
        _emit("after-hash", index, rel, shape, leaf.dtype, leaf_bytes, sha256=rec["sha256"])
        _feed("leaf", rec["sha256"].encode("ascii"))
        leaves.append(rec)
        numel += rec["numel"]
        nbytes += rec["bytes"]
        del host  # release the local host copy before the next device transfer
    _feed("total", f"leaf_count={len(leaves)};numel={numel};bytes={nbytes}".encode("ascii"))
    _emit("complete", leaf_count, "", (), np.dtype(np.uint8), 0, sha256=h.hexdigest())
    return {"sha256": h.hexdigest(), "leaf_count": len(leaves), "numel": numel, "bytes": nbytes, "leaves": leaves}
