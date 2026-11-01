#!/usr/bin/env python3
"""Strict Pi0.5 model-load smoke against the pinned OpenPI mirror.

Admission gates run before any JAX/Flax/OpenPI import: a SLURM_JOB_ID environment marker,
--deadline-unix within (0, 1800]s, sha256 of the accepted 29-object weight
report, exact upstream git HEAD, and a fresh --output-report path. The load
uses the official model.restore_params / Pi0Config / config.load path with
remove_extra_params=False: any missing or extra key fails, there is no random
fallback, and no optimizer, forward, or backward is built here. The JSON
report -- not the exit code alone -- records acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_HEAD = "215abfb217dbac7d5f1273282331b9b1866c0479"
EXPECTED_VERIFIED_COUNT = 29
EXPECTED_VERIFIED_BYTES = 12441749581
MIN_TOTAL_SCALARS = 1_000_000_000
MAX_DEADLINE_S = 1800
TRAINABLE_NOTE = (
    "all parameter leaves are prospective trainable (no freeze/LoRA); "
    "NO actual gradients or updates measured"
)
CUDA_PLATFORM = "cuda"


def _resolve_cuda_platform():
    """Pin JAX to CUDA before any heavy import; refuse inherited non-CUDA overrides.

    JAX 0.5.3 expands the 'gpu' alias to CUDA+ROCm backends, and the ROCm
    backend then fails at initialization (missing GpuAllocatorConfig), so an
    inherited JAX_PLATFORMS=gpu aborts the load before weight restore. There
    is deliberately no CPU fallback: only an explicit/unset CUDA target is
    admitted.
    """
    requested = os.environ.get("JAX_PLATFORMS")
    if requested is not None and requested.strip().lower() != CUDA_PLATFORM:
        raise RuntimeError(
            f"JAX_PLATFORMS={requested!r} is not supported: this loader is CUDA-only "
            f"with no CPU fallback; set JAX_PLATFORMS={CUDA_PLATFORM} or unset it"
        )
    os.environ["JAX_PLATFORMS"] = CUDA_PLATFORM
    return CUDA_PLATFORM


class _DeadlineExceeded(Exception):
    pass


class _Clock:
    def __init__(self):
        self._t = time.monotonic()
        self.elapsed = {}

    def mark(self, name):
        t = time.monotonic()
        self.elapsed[name] = round(t - self._t, 3)
        self._t = t


def _on_alarm(signum, frame):
    raise _DeadlineExceeded("deadline reached during model load")


def _read_head(root: Path) -> str:
    git = root / ".git"
    if git.is_file():
        git = Path(git.read_text().split(":", 1)[1].strip())
        if not git.is_absolute():
            git = root / git
    head = (git / "HEAD").read_text().strip()
    if not head.startswith("ref:"):
        return head
    ref = head[4:].strip()
    ref_path = git / ref
    if ref_path.exists():
        return ref_path.read_text().strip()
    for line in (git / "packed-refs").read_text().splitlines():
        if line.startswith(("#", "^")):
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[1].strip() == ref:
            return parts[0].strip()
    raise RuntimeError(f"cannot resolve upstream HEAD ref {ref}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--upstream-root", required=True, type=Path)
    p.add_argument("--weights-dir", required=True, type=Path)
    p.add_argument("--weight-report", required=True, type=Path)
    p.add_argument("--expected-weight-report-sha256", required=True)
    p.add_argument("--output-report", required=True, type=Path)
    p.add_argument("--deadline-unix", required=True, type=float)
    return p.parse_args(argv)


def _admit(args):
    if not os.environ.get("SLURM_JOB_ID", "").strip():
        raise RuntimeError("SLURM_JOB_ID is not set; refuse to load on a non-allocated host")
    remaining = args.deadline_unix - time.time()
    if not 0 < remaining <= MAX_DEADLINE_S:
        raise RuntimeError(f"deadline must be future and <= {MAX_DEADLINE_S}s away, got {remaining:.1f}s")
    if args.output_report.exists():
        raise RuntimeError(f"output report already exists: {args.output_report}")
    raw = args.weight_report.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != args.expected_weight_report_sha256:
        raise RuntimeError("weight report sha256 mismatch")
    wr = json.loads(raw)
    totals = wr.get("totals") or {}
    if (totals.get("verified_count"), totals.get("total_count"), totals.get("verified_bytes")) != (
        EXPECTED_VERIFIED_COUNT, EXPECTED_VERIFIED_COUNT, EXPECTED_VERIFIED_BYTES,
    ) or wr.get("stop_reason") != "all_verified":
        raise RuntimeError("weight report totals are not the accepted complete set")
    verified = [r for r in (wr.get("audit") or []) if r.get("state") == "verified"]
    names = {r.get("name") for r in verified}
    sizes = [r.get("actual_size", r.get("size")) for r in verified]
    if (len(verified), len(names)) != (EXPECTED_VERIFIED_COUNT, EXPECTED_VERIFIED_COUNT) or any(
        not isinstance(s, int) or s < 0 for s in sizes
    ) or sum(sizes) != EXPECTED_VERIFIED_BYTES:
        raise RuntimeError("weight report audit rows are not 29 unique verified objects")
    head = _read_head(args.upstream_root)
    if head != EXPECTED_HEAD:
        raise RuntimeError(f"upstream HEAD {head} != pinned {EXPECTED_HEAD}")
    return digest, wr, head, remaining


def _load_and_count(upstream_root: Path, weights_dir: Path):
    resolved = _resolve_cuda_platform()  # before heavy imports
    sys.path.insert(0, str(upstream_root / "src"))
    import jax  # noqa: PLC0415 - deferred: heavy JAX/OpenPI import
    import jax.numpy as jnp  # noqa: PLC0415
    from flax import nnx  # noqa: PLC0415
    from openpi.models import model as upstream_model  # noqa: PLC0415
    from openpi.models import pi0_config as upstream_pi0_config  # noqa: PLC0415

    devices = jax.devices()
    gpus = [d for d in devices if d.platform == "gpu"]
    default_backend = jax.default_backend()
    if default_backend == "cpu" or len(gpus) != 1:
        raise RuntimeError(
            f"require exactly 1 JAX GPU, no CPU fallback; requested={resolved!r} "
            f"resolved_default_backend={default_backend!r} devices={[d.platform for d in devices]}"
        )
    gpu = {
        "platform": gpus[0].platform, "device_kind": gpus[0].device_kind, "device_count": len(devices),
        "requested_platform": resolved, "default_backend": default_backend,
        "devices": [d.platform for d in devices],
    }

    params = upstream_model.restore_params(weights_dir / "params", dtype=jnp.float32)
    config = upstream_pi0_config.Pi0Config(
        pi05=True, action_dim=32, action_horizon=25, max_token_len=200,
        dtype="bfloat16", paligemma_variant="gemma_2b", action_expert_variant="gemma_300m",
    )
    model = config.load(params, remove_extra_params=False)
    jax.block_until_ready(params)

    leaves = [(tuple(p), v.value) for p, v in nnx.iter_graph(model) if isinstance(v, nnx.Param)]
    total = 0
    leaves_by_dtype = {}
    scalars_by_dtype = {}
    top_components = {}
    for path, arr in leaves:
        n = int(arr.size)
        total += n
        key = str(path[0])
        comp = top_components.setdefault(key, {"leaves": 0, "scalars": 0})
        comp["leaves"] += 1
        comp["scalars"] += n
        d = str(arr.dtype)
        leaves_by_dtype[d] = leaves_by_dtype.get(d, 0) + 1
        scalars_by_dtype[d] = scalars_by_dtype.get(d, 0) + n
    if total <= MIN_TOTAL_SCALARS:
        raise RuntimeError(f"loaded total {total} <= {MIN_TOTAL_SCALARS}")
    counts = {
        "leaves": len(leaves), "total_scalars": total,
        "trainable_leaves": len(leaves), "trainable_scalars": total,
        "leaves_by_dtype": leaves_by_dtype, "scalars_by_dtype": scalars_by_dtype,
        "top_component_count": len(top_components), "top_components": top_components,
        "trainable_note": TRAINABLE_NOTE,
    }
    return gpu, counts


def main(argv=None) -> int:
    clock = _Clock()
    started = datetime.now(timezone.utc).isoformat()
    rep = {"schema": "pi05-model-load-smoke/v1", "started_utc": started,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "loaded": False, "error": None}
    args = None
    try:
        args = _parse_args(argv)
        rep["weight_report"] = {"path": str(args.weight_report)}
        rep["upstream"] = {"root": str(args.upstream_root), "head": None}
        digest, wr, head, remaining = _admit(args)
        clock.mark("admission")
        rep["weight_report"].update({"sha256": digest, "stop_reason": wr.get("stop_reason"),
                                     "verified_count": wr["totals"]["verified_count"],
                                     "verified_bytes": wr["totals"]["verified_bytes"]})
        rep["upstream"]["head"] = head
        rep["jax_platform"] = {"requested": os.environ.get("JAX_PLATFORMS")}
        rep["jax_platform"]["resolved"] = _resolve_cuda_platform()
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, max(remaining - 1.0, 0.1))
        gpu, counts = _load_and_count(args.upstream_root, args.weights_dir)
        clock.mark("load_and_count")
        rep.update({"gpu": gpu, "parameters": counts, "loaded": True})
        rep["jax_platform"].update(
            {"runtime_default_backend": gpu["default_backend"], "runtime_devices": gpu["devices"]}
        )
    except Exception as exc:  # noqa: BLE001 - every failure is recorded in the report
        clock.mark("failed")
        rep["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        rep["phases_s"] = clock.elapsed
        rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
        if args is not None and not args.output_report.exists():
            args.output_report.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"loaded": rep["loaded"], "output_report": str(args.output_report) if args else None}))
    return 0 if rep["loaded"] else 1


if __name__ == "__main__":
    sys.exit(main())
