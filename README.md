# Embodied Trackpad

Research software for a robot that operates a computer through physical trackpad interaction and visual feedback.

机器人通过触控板使用电脑：连续视觉反馈、灵巧手运动控制与长程任务记忆。

The intended action path is **policy → robot dynamics → touch contacts → device behavior → GUI**. Screen images close the loop. Software click commands are not the robot action space.

**Status:** the single-H800 MuJoCo Warp environment probe is verified (12/12 checks on a tiny contact scene). Protocol/data foundations are under implementation review. No trained policy, full-hand environment, real-Mac autonomous task, or hardware result is claimed. See [STATUS.md](STATUS.md) for verified milestones.

## Design

- Full articulated four-finger hand as the primary embodiment; independently articulated two-finger hardware as a later comparison.
- MuJoCo CPU reference and NVIDIA GPU batch simulation through MuJoCo Warp, subject to backend validation.
- Visual task reasoning, evidence-backed long-term memory, and continuous motor control with short action chunks.
- Synthetic interactive GUI demonstrations, corrective trajectories, then simulated-hand/real-Mac paired data.
- A dedicated M1 Mac can run target applications, screen capture, and the input bridge; GPU servers handle learning and model inference.

Read the [architecture](docs/architecture.zh-CN.md), [implementation plan](docs/implementation-plan.zh-CN.md), [data plan](docs/data-plan.zh-CN.md), and [Mac setup plan](docs/mac-testbed.zh-CN.md). Earlier designs are retained in [docs/history](docs/history) and are explicitly superseded.

The [H800 probe report](docs/experiments/2026-10-30-h800-probe.md) includes pinned dependencies, reproduction commands, the first failed attempt and correction, numerical checks, and measured limits. GPU execution is established; a performance advantage over CPU or full-hand readiness is not.

## Reproducibility and responsibilities

The coordinator owns architecture, review, integration, publication, and evidence reporting. Kimi owns bounded implementation tasks. A worker's completion message is not an acceptance result.

Every meaningful change or experiment has a Git commit and an entry in [experiments/ledger.jsonl](experiments/ledger.jsonl), linked to its configuration, source revision, checks, and outcome. Corrections append new entries. [CONTRIBUTING.md](CONTRIBUTING.md) defines the process.

Large datasets, checkpoints, personal screenshots, account details, and raw agent sessions stay outside Git. Publish reviewed manifests and aggregate results; review licenses and privacy before releasing data. Source citations are in [docs/references.md](docs/references.md).

## License

Project-authored code and documentation are licensed under [Apache-2.0](LICENSE). Third-party models, assets, and datasets retain their original licenses and are not relicensed by this repository. See [THIRD_PARTY.md](THIRD_PARTY.md).
