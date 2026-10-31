# LEAP v1 Full simulation model baseline

2026-10-31. Status: **reference hand-only CPU diagnostic sequence measured locally (the probe executed this sequence twice; the saved result is the second run); pad contact, native/online feedback and training NOT RUN.** The [independent acceptance record](../experiments/records/leap-v1-cpu-baseline-20261031.json) preserves source hashes, measured limitations, timeout/rework and actual checks.

## Source and pin

- Model: MuJoCo Menagerie `leap_hand`, right hand MJCF (`right_hand.xml`) + 10 referenced OBJ meshes.
- Repository: https://github.com/google-deepmind/mujoco_menagerie, pinned commit `a662de603224c47123b0783821c58ad4402ad3e3` (2026-10-31, HEAD of main at verification time). The historical locator pin `f3b6013629c7b94f8ee93fd977b161fa8350f9bc` (2026-10-23) was checked via the GitHub compare API: **zero `leap_hand` file changes** between the two revisions.
- License: MIT, Copyright 2023 Ananye Agarwal (`leap_hand/LICENSE`, verbatim copy kept alongside the assets).
- Assets are NOT redistributed: 14 files (~7.75 MiB, within the 100 MiB budget) are fetched on demand into the git-ignored `assets/downloads/leap_v1_full/` and verified against the committed sha256 manifest `assets/manifests/leap-v1-full.json`. Left hand, scene wrappers, mount mesh and preview PNG are intentionally not downloaded. The noncommercial LEAP_Hand_API Python package is not used.
- Embodiment ID `leap_v1_full_right_menagerie_v1`, calibration ID `uncalibrated_upstream_defaults_v1` — independent of the v2 `leap_sim` stack; nothing v2 is relabeled.

## Compiled inventory (measured by the probe, `inventory.json`)

- `nq = nv = nu = njnt = 16`: exactly 16 independently commanded revolute joints, one position actuator each, one-to-one, upstream order `if_*`, `mf_*`, `rf_*`, `th_*` (index/middle/ring/thumb; `mcp/rot/pip/dip`, thumb `cmc/axl/mcp/ipl`). No joint equalities (v2's four rigid dip=pip couplings do not exist in v1).
- Fixed palm (palm body welded to world), 88 geoms (visual meshes + box-primitive collisions), 18 upstream contact exclusions, 16 upstream jointpos sensors, upstream solver options (implicitfast, elliptic cone, impratio=100, 2 ms timestep).
- Local adaptations (disclosed, nothing else changed): 4 massless fingertip sites (`tip_index/tip_middle/tip_ring/tip_thumb`) added on the distal bodies at the compiled frames of the upstream tip mesh geoms; 1 render light.

## Calibration status

Actuator gains (kp=3.0, kv=0.01), joint damping 0.03, frictionloss 0.001, all link masses/inertias and the box collision geometry are **upstream simulation defaults — not XC330-M288-T motor calibrations and not measured hardware inertias**. v1 Full = 16 XC330-M288-T channels; hardware gain/force/inertia identification is future work.

## Reproduce

```
python tools/fetch_leap_v1_assets.py            # or --check
PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python tools/leap_v1_model_probe.py --out .local/v1-baseline/probe
```

The probe writes `inventory.json`, `result.json`, `motion-metrics.json`, `self-contacts.json`, `trace.npz` and real MuJoCo renders (overview + 16 per-channel sweeps + spread + flex + returned). Motion goes through `ctrl` + `mj_step` only; qpos is written once at the all-zero open reset (measured contact-free).

## First measured baseline (hand-only, see `.local/v1-baseline/probe/result.json`)

- 11,000 steps / 22 simulated seconds; all acceptance gates PASS: finite state, zero solver warnings, joint-limit excursion ≤ 0.01 rad, no negative normal contact force, ctrl always in bounds.
- Measured (reported, not gated): per-channel held tracking maxima are ≤ 0.066 rad for 14 of 16 channels, while `if_rot`/`mf_rot` reach 0.297/0.279 rad — these two maxima coincide with recorded finger-finger collisions at the +0.6·range rot sweep targets, so the 0.10 rad full-tracking reference is **NOT** met across all 16 channels (without per-time contact attribution the 14 channels are not labelled "free-space"). Index/middle spread 0.045 → 0.111 m; recovered-open error 0.026 rad after ≥0.7 s hold; self contacts recorded per event (3382, pairs if_ds|mf_ds, mf_ds|rf_ds, palm|th_ds, if_md|th_ds; none at the flat reset pose).
