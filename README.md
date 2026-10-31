# Embodied Trackpad

Research software for a robot that operates a computer through physical trackpad interaction and visual feedback.

机器人通过触控板使用电脑：连续视觉反馈、灵巧手运动控制与长程任务记忆。

The intended action path is **policy → robot dynamics → touch contacts → device behavior → GUI**. Screen images close the loop. Software click commands are not the robot action space.

**Current target:** the user has selected **LEAP v1 Full**, four fingers and 16 independent motor channels. Its [hand-only CPU baseline](docs/leap-v1-model.md) has been implemented and reviewed: 11,000 steps / 22 simulated seconds, with explicit collision/tracking limitations. A separate [P1 contact profile](docs/experiments/2026-10-31-v1-contact-p1.zh-CN.md) now passes the fixed-trajectory continuous-contact checks; after three failed controller diagnostics, a [force-governed visual episode](docs/experiments/2026-10-31-v1-governed-pointing.zh-CN.md) now passes one fixed near-target task (24.32→1.67px), with sustained contact and safe release. [ADR 0007](docs/decisions/0007-leap-v1-full-and-learning-priorities.zh-CN.md) records the hardware choice. [ADR 0008](docs/decisions/0008-full-finetuning-and-simulation-first.zh-CN.md) confirms full fine-tuning and simulation-first validation, with physical FSR/conductive-fingertip checks deferred.

**Measured status:** LEAP v1 now has a separately reviewed [1/32-world H800 runtime smoke](docs/leap-v1-gpu-runtime.md), including a diagnostic PyTorch BF16 update. This is hand-only runtime verification, not policy training. V1 P1 contact is separately reviewed for bounded visual validation; the visual action chain has now passed one fixed target, while broader direction coverage, D1 data and learning remain pending. The earlier v2 1/32/128/512-world diagnostics, scripted pad cycle and 105-frame native replay retain their original v2 attribution. See [STATUS.md](STATUS.md).

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

The new [v1 model guide](docs/leap-v1-model.md) describes the separate 16-channel model, asset fetcher and CPU probe. It retains upstream simulation dynamics and has not been calibrated to physical XC330 motors. The following earlier pad/GPU/native results belong to v2.

The [local live viewer](docs/leap-v1-live.md) renders v1 physics with displayed-frame freshness and a bounded runtime. Its reviewed demonstration has ended. The new [training monitor interface](docs/training-monitor.zh-CN.md) reads producer-published frames and metadata, separates test/demo/evaluation/training sources, and follows declared job lifetime. Its finite TEST-source browser checks passed; actual training attachment remains pending.

The [pad-contact guide](docs/leap-pad.md) and [acceptance record](experiments/records/leap-pad-contact-20261031.json) document the first scripted fingertip cycle. The [model guide](docs/leap-model.md) describes the pinned asset downloader, optional CPU dependencies, runtime probe and calibration limits. Models remain in an ignored cache; the published code does not bundle meshes. [Measured results and failed candidates](experiments/records/leap-v2-cpu-20261031.json) retain the distinction between a scripted diagnostic and an autonomous learned task.

## Design

- LEAP v1 Full is the selected four-finger research target, using 16 XC330 motor channels. Optional [FSR fingertips](docs/hardware/leap-v1-fsr.zh-CN.md) are designed but not built. Existing v2 and Allegro results remain separate regression references. Independently articulated two-finger hardware is a later comparison.
- MuJoCo CPU reference and NVIDIA GPU batch simulation through MuJoCo Warp, subject to backend validation.
- Visual task reasoning, evidence-backed long-term memory, and continuous motor control with short action chunks.
- Synthetic interactive GUI demonstrations, corrective trajectories, then simulated-hand/real-Mac paired data.
- Current test host: an M5 Mac, starting with a confined native test GUI; a full macOS guest is a separate stage. The M1 remains an optional later testbed. GPU servers handle learning and model inference.

The [current decision](docs/decisions/0007-leap-v1-full-and-learning-priorities.zh-CN.md) follows the user's final v1 Full choice. The [VLA architecture review](docs/research/2026-10-31-vla-architecture-review.zh-CN.md) compares pi0.5, GR00T N1.7 and LingBot-VLA 2.0; the [pi0.5 empirical review](docs/research/2026-10-31-pi05-finetuning-review.zh-CN.md) explains pretrained full fine-tuning, corrective data and RTC priorities. The [touch-model explanation](docs/simulation-touch-model.zh-CN.md) separates physics contacts, virtual device behavior and actor observations. The older v2 decision and all results retain their original meaning. G1/H2 mounting remains custom integration; prior [selection](docs/research/2026-10-31-hand-selection.zh-CN.md), [prices](docs/research/2026-10-31-hand-market-prices.zh-CN.md) and [humanoid evidence](docs/research/2026-10-31-humanoid-integration.zh-CN.md) remain context.

Read the [architecture](docs/architecture.zh-CN.md), [implementation plan](docs/implementation-plan.zh-CN.md), [data plan](docs/data-plan.zh-CN.md), and [Mac setup plan](docs/mac-testbed.zh-CN.md). Earlier designs are retained in [docs/history](docs/history) and are explicitly superseded.

The [LEAP H800 validation](docs/leap-h800.md) now covers1/32/128/512 repeated worlds with42/42 checks and14,000 steps per world. The successful job completed in7m46s;512-world whole-loop throughput was about47,001 aggregate world-steps/s. This is hand-only backend validation with sampled contact/state checks, separate from learning or GUI performance.

The [initial H800 probe report](docs/experiments/2026-10-30-h800-probe.md) preserves the tiny-scene baseline. The [full-hand H800 report](docs/experiments/2026-10-31-h800-fullhand.zh-CN.md) records the contact-readback correction, actual completed allocation, complete-cycle timing and coverage limits. At 512 repeated worlds this harness reached 67,070 aggregate world-steps/s, about 1.23 times its single-CPU-world reference; this is not an end-to-end training speedup or a comparison with a CPU batch implementation.

The [native Mac test GUI](docs/mac-sandbox.md) accepts pad-contact frames and returns only its own window pixels. A separately reviewed [v2 contact replay](docs/leap-native-bridge.md) connects recorded articulated-hand physics to that GUI. It does not yet close the visual feedback loop. The current next task is the [v1 Full rebaseline](docs/tasks/leap-v1-full-rebaseline.zh-CN.md); online feedback remains a gate before D1 collection.

## Reproducibility and responsibilities

The coordinator owns architecture, budgets, read-only review, integration, publication, and evidence reporting. Kimi owns bounded implementation and operational work: installation, VM/server setup, simulation, training, capture/input experiments, and live streaming. A worker's completion message is not an acceptance result. The [live and batch acceptance contract](docs/decisions/0004-live-and-batch-evidence.md) defines what a full-hand stream or comparable H800 result must demonstrate.

Every meaningful change or experiment has a Git commit and an entry in [experiments/ledger.jsonl](experiments/ledger.jsonl), linked to its configuration, source revision, checks, and outcome. Corrections append new entries. [CONTRIBUTING.md](CONTRIBUTING.md) defines the process.

The [initial development report](docs/experiments/2026-10-30-development-report.zh-CN.md) preserves review findings, worker timeouts, corrections, delegation usage, and the next acceptance boundary.

Large datasets, checkpoints, personal screenshots, account details, and raw agent sessions stay outside Git. Publish reviewed manifests and aggregate results; review licenses and privacy before releasing data. Source citations are in [docs/references.md](docs/references.md).

## License

Project-authored code and documentation are licensed under [Apache-2.0](LICENSE). Third-party models, assets, and datasets retain their original licenses and are not relicensed by this repository. See [THIRD_PARTY.md](THIRD_PARTY.md).
