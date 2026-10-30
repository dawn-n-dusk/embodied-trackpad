# Embodied Trackpad

Research software for a robot that operates a computer through physical trackpad interaction and visual feedback.

机器人通过触控板使用电脑：连续视觉反馈、灵巧手运动控制与长程任务记忆。

The intended action path is **policy → robot dynamics → touch contacts → device behavior → GUI**. Screen images close the loop. Software click commands are not the robot action space.

**Status:** the LEAP v2 Advanced adapter now runs the complete 17-motor/21-joint hand in MuJoCo CPU. Its 7-second scripted motion passed 34 focused asset/physics checks and 14,000-step numerical, contact and pose-recovery gates, with independent source/trace/image review. This is fixed-base hand motion; pad/GUI integration and policy training remain pending. The foundation (89 prior tests), confined native Mac GUI (47 additional real-app tests and 18 smoke checks), and earlier Allegro H800 replay (43/43 checks) are separate evidence. See [STATUS.md](STATUS.md).

## Run the foundation

From a checkout, using Python 3.10 or newer:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/python -m embodied_trackpad.cli \
  --out runs/demo --seed 7 --steps 8 --created-utc 2026-10-30T00:00:00Z
```

Use a fresh output directory for each episode. This generates a **mock software fixture**, marked `training_eligible=false`; it exercises recording and validation without robot physics or Mac input. See the [foundation guide](docs/foundation.md) for contracts, timing, evidence references, and reproducibility.

## Run the LEAP adapter

The [model guide](docs/leap-model.md) describes the pinned asset downloader, optional CPU dependencies, runtime probe and calibration limits. Models remain in an ignored cache; the published code does not bundle meshes. [Measured results and failed candidates](experiments/records/leap-v2-cpu-20261031.json) retain the distinction between a scripted diagnostic and an autonomous learned task.

## Design

- LEAP v2 Advanced is the selected five-finger research target: 17 motor channels, 21 articulated joints and an uncalibrated Metal hardware target. Existing Allegro results remain a separate regression reference. Independently articulated two-finger hardware is a later comparison.
- MuJoCo CPU reference and NVIDIA GPU batch simulation through MuJoCo Warp, subject to backend validation.
- Visual task reasoning, evidence-backed long-term memory, and continuous motor control with short action chunks.
- Synthetic interactive GUI demonstrations, corrective trajectories, then simulated-hand/real-Mac paired data.
- Current test host: an M5 Mac, starting with a confined native test GUI; a full macOS guest is a separate stage. The M1 remains an optional later testbed. GPU servers handle learning and model inference.

The [latest-model decision](docs/decisions/0006-latest-leap-and-hardware-readiness.zh-CN.md) supersedes the earlier v1 selection with the user's latest/highest-configuration requirement and keeps software simulation independent of hardware arrival. It records coupling, inertial-marker and Metal-calibration qualifications, plus future MacBook/external-Trackpad interfaces. G1/H2 mounting remains custom integration. Earlier [selection research](docs/research/2026-10-31-hand-selection.zh-CN.md), [market prices](docs/research/2026-10-31-hand-market-prices.zh-CN.md) and [humanoid evidence](docs/research/2026-10-31-humanoid-integration.zh-CN.md) remain available as context.

Read the [architecture](docs/architecture.zh-CN.md), [implementation plan](docs/implementation-plan.zh-CN.md), [data plan](docs/data-plan.zh-CN.md), and [Mac setup plan](docs/mac-testbed.zh-CN.md). Earlier designs are retained in [docs/history](docs/history) and are explicitly superseded.

The [initial H800 probe report](docs/experiments/2026-10-30-h800-probe.md) preserves the tiny-scene baseline. The [full-hand H800 report](docs/experiments/2026-10-31-h800-fullhand.zh-CN.md) records the contact-readback correction, actual completed allocation, complete-cycle timing and coverage limits. At 512 repeated worlds this harness reached 67,070 aggregate world-steps/s, about 1.23 times its single-CPU-world reference; this is not an end-to-end training speedup or a comparison with a CPU batch implementation.

The [native Mac test GUI](docs/mac-sandbox.md) accepts pad-contact frames and returns only its own window pixels. Its current results use synthetic contact tests; it has not yet been driven by the articulated robot or a learned policy.

## Reproducibility and responsibilities

The coordinator owns architecture, budgets, read-only review, integration, publication, and evidence reporting. Kimi owns bounded implementation and operational work: installation, VM/server setup, simulation, training, capture/input experiments, and live streaming. A worker's completion message is not an acceptance result. The [live and batch acceptance contract](docs/decisions/0004-live-and-batch-evidence.md) defines what a full-hand stream or comparable H800 result must demonstrate.

Every meaningful change or experiment has a Git commit and an entry in [experiments/ledger.jsonl](experiments/ledger.jsonl), linked to its configuration, source revision, checks, and outcome. Corrections append new entries. [CONTRIBUTING.md](CONTRIBUTING.md) defines the process.

The [initial development report](docs/experiments/2026-10-30-development-report.zh-CN.md) preserves review findings, worker timeouts, corrections, delegation usage, and the next acceptance boundary.

Large datasets, checkpoints, personal screenshots, account details, and raw agent sessions stay outside Git. Publish reviewed manifests and aggregate results; review licenses and privacy before releasing data. Source citations are in [docs/references.md](docs/references.md).

## License

Project-authored code and documentation are licensed under [Apache-2.0](LICENSE). Third-party models, assets, and datasets retain their original licenses and are not relicensed by this repository. See [THIRD_PARTY.md](THIRD_PARTY.md).
