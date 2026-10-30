#!/usr/bin/env python3
"""Bounded single-GPU environment probe for Embodied Trackpad.

Validates, inside one Slurm GPU allocation:
  1. package / runtime / driver / GPU metadata (hostnames, users and paths
     are hashed or omitted),
  2. a small MuJoCo CPU reference contact scene (fingertip pressing a
     trackpad-like surface),
  3. the matching MuJoCo Warp (MJWarp) GPU scene with a tiny world batch,
  4. finite states, contact behavior and selected CPU/GPU tolerances,
  5. warmup vs steady-state stepping cost, measured, not extrapolated.

The script exits nonzero if the GPU path fails. It never silently falls
back to CPU Warp and reports that as a GPU result.

Configuration is via CLI flags or PROBE_* environment variables (see
--help). All inputs are generic; no account, host or user identity is
embedded or emitted.
"""

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

SCENE_XML = """
<mujoco model="fingertip_trackpad_probe">
  <option timestep="0.004166666666666667" iterations="50" tolerance="1e-8"/>
  <default>
    <geom condim="3" friction="0.8 0.02 0.01" solref="0.02 1" solimp="0.9 0.95 0.001"/>
  </default>
  <worldbody>
    <geom name="trackpad" type="box" size="0.05 0.035 0.002" pos="0 0 -0.002"/>
    <body name="finger" pos="0 0 0.045">
      <joint name="finger_z" type="slide" axis="0 0 1" range="-0.045 0.045" damping="0.3"/>
      <geom name="fingertip" type="capsule" size="0.004" fromto="0 0 -0.005  0 0 -0.035" mass="0.02"/>
    </body>
  </worldbody>
  <actuator>
    <position name="press" joint="finger_z" kp="20" ctrlrange="-0.045 0.045"/>
  </actuator>
</mujoco>
"""

# Fingertip capsule lowest point, in body frame (segment end -0.035 minus radius 0.004).
TIP_OFFSET_Z = -0.039
CTRL_TARGET = -0.02  # position-actuator reference driving the fingertip onto the pad
TOL_POS = 5e-3       # m, CPU-vs-GPU final joint position agreement (contact-scale, not bitwise)
TOL_VEL = 5e-2       # m/s, CPU-vs-GPU final joint velocity agreement
TIP_Z_TOL = 2e-3     # m, expected fingertip rest height above pad surface


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha12(text):
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def finite(values):
    import math
    return all(math.isfinite(v) for v in values)


def nvidia_smi_summary():
    """Return sanitized GPU/driver facts; never serial numbers or paths."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return {"available": False, "error": out.stderr.strip()[:200]}
        rows = []
        for line in out.stdout.strip().splitlines():
            name, driver, mem = [p.strip() for p in line.split(",")]
            rows.append({"name": name, "driver_version": driver, "memory_total": mem})
        header = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30)
        cuda = None
        if "CUDA Version:" in header.stdout:
            cuda = header.stdout.split("CUDA Version:")[1].split("|")[0].strip()
        return {"available": True, "gpus_visible": rows, "cuda_version_driver": cuda}
    except Exception as exc:  # noqa: BLE001 - probe must record, not crash, on metadata gaps
        return {"available": False, "error": str(exc)[:200]}


def run_cpu_reference(steps, bench_steps):
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    data.ctrl[:] = CTRL_TARGET
    ncon_trace = []
    t0 = time.perf_counter()
    for i in range(steps):
        mujoco.mj_step(model, data)
        if i % 60 == 0 or i == steps - 1:
            ncon_trace.append(int(data.ncon))
    cpu_seconds = time.perf_counter() - t0

    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    tip_z = model.body("finger").pos[2] + qpos[0] + TIP_OFFSET_Z

    bench = mujoco.MjData(model)
    bench.ctrl[:] = CTRL_TARGET
    t0 = time.perf_counter()
    for _ in range(bench_steps):
        mujoco.mj_step(model, bench)
    bench_seconds = time.perf_counter() - t0

    return {
        "steps": steps,
        "seconds": cpu_seconds,
        "final_qpos": qpos.tolist(),
        "final_qvel": qvel.tolist(),
        "final_ncon": int(data.ncon),
        "max_ncon_seen": max(ncon_trace),
        "ncon_trace_sampled": ncon_trace,
        "fingertip_z_final": float(tip_z),
        "states_finite": bool(finite(qpos.tolist()) and finite(qvel.tolist())),
        "bench": {
            "steps": bench_steps,
            "seconds": bench_seconds,
            "steps_per_s": bench_steps / bench_seconds if bench_seconds > 0 else None,
        },
    }


def _sync(wp):
    if hasattr(wp, "synchronize_device"):
        wp.synchronize_device()
    else:
        wp.synchronize()


def run_gpu(nworlds_list, steps, bench_steps, nconmax, njmax):
    import mujoco
    import mujoco_warp as mjw
    import numpy as np
    import warp as wp

    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("no CUDA device visible to warp; refusing CPU fallback")
    device = wp.get_cuda_device(0)
    runs = []

    mjm = mujoco.MjModel.from_xml_string(SCENE_XML)
    mjd0 = mujoco.MjData(mjm)
    mjd0.ctrl[:] = CTRL_TARGET

    with wp.ScopedDevice(device):
        m = mjw.put_model(mjm)

        for nworld in nworlds_list:
            kwargs = {"nworld": nworld}
            if nconmax is not None:
                kwargs["nconmax"] = nconmax
            if njmax is not None:
                kwargs["njmax"] = njmax
            d = mjw.put_data(mjm, mjd0, **kwargs)

            # Warmup: first step includes kernel/module load for this config.
            _sync(wp)
            t0 = time.perf_counter()
            mjw.step(m, d)
            _sync(wp)
            warmup_s = time.perf_counter() - t0

            # Remaining trajectory steps for the correctness comparison.
            for _ in range(steps - 1):
                mjw.step(m, d)
            _sync(wp)

            # Steady-state timing on a fresh batch, compile already amortized.
            d_bench = mjw.put_data(mjm, mjd0, **kwargs)
            mjw.step(m, d_bench)
            _sync(wp)
            t0 = time.perf_counter()
            for _ in range(bench_steps):
                mjw.step(m, d_bench)
            _sync(wp)
            bench_seconds = time.perf_counter() - t0

            qpos = np.asarray(d.qpos.numpy() if hasattr(d.qpos, "numpy") else d.qpos)
            qvel = np.asarray(d.qvel.numpy() if hasattr(d.qvel, "numpy") else d.qvel)

            # Extract world 0 through the official readback path to inspect
            # contacts with the standard MuJoCo API.
            out = mujoco.MjData(mjm)
            mjw.get_data_into(out, mjm, d, world_id=0)
            ncon_world0 = int(out.ncon)
            naconmax = int(getattr(d, "naconmax", -1))

            runs.append({
                "nworld": nworld,
                "warmup_first_step_s": warmup_s,
                "steps": steps,
                "final_qpos_world0": out.qpos.tolist(),
                "final_qvel_world0": out.qvel.tolist(),
                "qpos_finite_all_worlds": bool(np.isfinite(qpos).all()),
                "qvel_finite_all_worlds": bool(np.isfinite(qvel).all()),
                "ncon_final_world0": ncon_world0,
                "naconmax_allocated": naconmax,
                "bench": {
                    "steps_per_world": bench_steps,
                    "seconds": bench_seconds,
                    "world_steps_per_s": nworld * bench_steps / bench_seconds if bench_seconds > 0 else None,
                    "steps_per_s_per_world": bench_steps / bench_seconds if bench_seconds > 0 else None,
                },
            })

    return {
        "device_name": str(getattr(device, "name", "unknown")),
        "device_is_cuda": bool(getattr(device, "is_cuda", "cuda" in str(device).lower())),
        "warp_version": wp.__version__,
        "mjw_version": getattr(mjw, "__version__", "unknown"),
        "runs": runs,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=os.environ.get("PROBE_OUT", "probe-result.json"))
    p.add_argument("--steps", type=int, default=int(os.environ.get("PROBE_STEPS", "480")))
    p.add_argument("--bench-steps", type=int, default=int(os.environ.get("PROBE_BENCH_STEPS", "480")))
    p.add_argument("--nworlds", default=os.environ.get("PROBE_NWORLDS", "1,32"))
    p.add_argument("--nconmax", type=int, default=None)
    p.add_argument("--njmax", type=int, default=None)
    p.add_argument("--cpu-only", action="store_true",
                   help="smoke-test metadata + CPU scene only; NOT a GPU result")
    args = p.parse_args()

    nworlds_list = [int(x) for x in args.nworlds.split(",") if x.strip()]

    result = {
        "probe": "embodied-trackpad-h800-env",
        "schema_version": "1.0",
        "utc_start": utcnow(),
        "source_commit": os.environ.get("PROBE_SOURCE_COMMIT", "unknown"),
        "mode": "cpu_smoke" if args.cpu_only else "gpu_probe",
        "host_sha256_12": sha12(socket.gethostname()),
        "user_sha256_12": sha12(os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"),
        "slurm_job_sha256_12": sha12(os.environ["SLURM_JOB_ID"]) if os.environ.get("SLURM_JOB_ID") else None,
        "cuda_visible_devices_set": bool(os.environ.get("CUDA_VISIBLE_DEVICES")),
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "python": platform.python_version()},
        "packages": {},
        "nvidia_smi": nvidia_smi_summary(),
        "scene": {
            "description": "single actuated capsule fingertip pressing a static box trackpad",
            "xml_sha256": hashlib.sha256(SCENE_XML.encode()).hexdigest(),
            "timestep_s": 1.0 / 240.0,
            "ctrl_target_m": CTRL_TARGET,
        },
        "tolerances": {"pos_m": TOL_POS, "vel_mps": TOL_VEL, "tip_z_m": TIP_Z_TOL},
        "checks": [],
        "errors": [],
    }

    import importlib.metadata as ilmd
    for pkg in ("mujoco", "mujoco-warp", "warp-lang", "numpy"):
        try:
            result["packages"][pkg] = ilmd.version(pkg)
        except ilmd.PackageNotFoundError:
            result["packages"][pkg] = None

    def check(name, passed, detail):
        result["checks"].append({"name": name, "pass": bool(passed), "detail": detail})

    cpu = None
    try:
        cpu = run_cpu_reference(args.steps, args.bench_steps)
        result["cpu_reference"] = cpu
        check("cpu_states_finite", cpu["states_finite"], "qpos/qvel finite after %d steps" % args.steps)
        check("cpu_contact_observed", cpu["max_ncon_seen"] >= 1,
              "max contacts seen = %d" % cpu["max_ncon_seen"])
        check("cpu_fingertip_rest_height", abs(cpu["fingertip_z_final"]) <= TIP_Z_TOL,
              "tip z = %.6f m" % cpu["fingertip_z_final"])
    except Exception as exc:  # noqa: BLE001
        result["errors"].append("cpu_reference: %s" % exc)
        check("cpu_reference_runs", False, str(exc)[:300])

    if args.cpu_only:
        result["gpu"] = {"skipped": True, "reason": "--cpu-only smoke mode; not a GPU result"}
    else:
        try:
            gpu = run_gpu(nworlds_list, args.steps, args.bench_steps, args.nconmax, args.njmax)
            result["gpu"] = gpu
            check("gpu_device_is_cuda", gpu["device_is_cuda"], gpu["device_name"])
            for run in gpu["runs"]:
                tag = "nworld=%d" % run["nworld"]
                check("gpu_states_finite[%s]" % tag,
                      run["qpos_finite_all_worlds"] and run["qvel_finite_all_worlds"],
                      "all worlds finite after %d steps" % run["steps"])
                if run["ncon_final_world0"] is not None:
                    check("gpu_contact_observed[%s]" % tag,
                          run["ncon_final_world0"] >= 1,
                          "world0 final ncon = %s (naconmax=%s)" % (
                              run["ncon_final_world0"], run["naconmax_allocated"]))
                if cpu is not None and run["nworld"] >= 1:
                    dq = abs(run["final_qpos_world0"][0] - cpu["final_qpos"][0])
                    dv = abs(run["final_qvel_world0"][0] - cpu["final_qvel"][0])
                    check("cpu_gpu_pos_agree[%s]" % tag, dq <= TOL_POS,
                          "max |dq| = %.6e m (tol %.1e)" % (dq, TOL_POS))
                    check("cpu_gpu_vel_agree[%s]" % tag, dv <= TOL_VEL,
                          "max |dv| = %.6e m/s (tol %.1e)" % (dv, TOL_VEL))
        except Exception as exc:  # noqa: BLE001 - GPU failure stays a failure
            import traceback
            result["errors"].append("gpu_probe: %s" % exc)
            result["gpu"] = {"failed": True, "error": str(exc)[:500],
                             "traceback": traceback.format_exc()[-2000:]}
            check("gpu_probe_runs", False, str(exc)[:300])

    result["utc_end"] = utcnow()
    result["overall"] = "pass" if (result["checks"] and all(c["pass"] for c in result["checks"])
                                   and not result["errors"]) else "fail"

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(json.dumps({"out": args.out, "overall": result["overall"],
                      "checks_failed": [c["name"] for c in result["checks"] if not c["pass"]],
                      "errors": result["errors"]}))
    return 0 if result["overall"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
