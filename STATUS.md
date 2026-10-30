# Current status

Updated: 2026-10-31 (Asia/Hong_Kong). Status is evidence-based, not inferred from a roadmap item.

| Milestone | State | Evidence / next condition |
|---|---|---|
| Architecture v2 | Accepted for implementation | User accepted the revised architecture on 2026-10-30; public design snapshot in docs |
| Public project | Published | Public code/design repository; publication and integration revisions are recorded in the append-only ledger |
| Protocol, recording, and dry-run pipeline | Independently validated locally | Six initial contracts, strict actor parsing, recording, schemas, CLI; 89 tests, 9 independent boundary checks, 12 CLI rejection cases; deterministic fixture output |
| GitHub CI | Passed on Python 3.10–3.13 | All four Ubuntu jobs passed at `78151af`; [verified run](https://github.com/dawn-n-dusk/embodied-trackpad/actions/runs/41097400783). Earlier native-integration receipt is retained. AppKit tests are explicitly skipped on Linux; native results are separate |
| MuJoCo CPU hand environment | One corrected debug cycle reviewed | New3.6s cycle: clear0.6s hold, intentional middle-tip contact,0.94N peak,0.79mm penetration, clear0.5s lift; [checkpoint and limitations](experiments/records/hand-cpu-debug-checkpoint-20261031.json). Uncalibrated teacher control; source integration review pending |
| H800 / MJWarp | GPU environment probe verified | One H800, tiny 1-DOF contact scene at 1/32 worlds; 12/12 checks; coordinator matched scheduler/script/result evidence. No full-hand or GPU speedup claim |
| Hand candidates | Research reviewed; no embodiment replacement yet | LEAP v1 and Tesollo DG-5F-S 20DoF shortlisted against Allegro V3; [comparison](docs/research/2026-10-31-hand-candidates.zh-CN.md). Real Mac capacitive detection unverified |
| Datasets | No project training dataset yet | Begin dynamic GUI/robot demonstrations; preserve provenance |
| Training | Not started | Wait for environment/data acceptance |
| Mac test host | M5 authorized and inspected | M5/16 GB/macOS 26.4.1; M1 currently unavailable. Lightweight confined native app is being implemented |
| macOS virtual machine | Guest boot and desktop verified | macOS 26.6.2, 4 CPUs/6 GB, fixed 1024×768; no host folder/clipboard sharing. Experimental host VNC listener remains retired. Authenticated1024×768 guest frames plus synthetic fixture click/scroll visually observed; cursor feedback and guest button-release acceptance pending ([receipt](experiments/records/guest-bridge-pixel-review-20261031.json)) |
| Confined native Mac GUI | Corrected software test GUI accepted | Kimi's signed AppKit app: 47 native integration tests, 136 total tests and 18 smoke checks passed. Coordinator reviewed source, raw receipts, checksums and actual drag image. [Record](experiments/records/mac-sandbox-first.json); synthetic contact tests only, physics bridge pending |
| Robot live stream | Full-hand diagnostic stream observed | Kimi deployed live23-actuator CPU physics/EGL,640×480 at~10Hz. Coordinator viewed advancing frames; [receipt](experiments/records/fullhand-live-diagnostic-review-20261031.json). Displayed source is older failed-precision scene; corrected-source update dispatched |
| Full-hand H800 batch simulation | Pending | Existing 12/12 result is only the tiny probe; it is not a full-hand result or measured GPU speedup |
| Real-Mac input and autonomous tasks | Not started | Dedicated testbed, bridge implementation, and measured verification required |

Historical resource audits describe available resource classes; they are not reservations. Initial work is bounded environment and implementation validation, not a full training campaign.

Verified GPU implementation commit: `8022a2db92de2bc577d55b05aebd88906ec4c83b`. The successful probe allocation completed and released its resources. The earlier failed device-API attempt remains recorded in [the report](docs/experiments/2026-10-30-h800-probe.md).

Verified foundation implementation commit: `c1dd62af52b76ec3f1d6fe7ab5863002deb57290`. The first 54-test pass did not establish acceptance; independently found gaps were returned to Kimi. Its correction call timed out after saving source/test changes. The coordinator independently verified those changes and completed documentation/records without modifying production code. See [the record](experiments/records/m0-foundation.json) and [development report](docs/experiments/2026-10-30-development-report.zh-CN.md).

All currently generated episodes are mock software fixtures, explicitly ineligible for training. Full protocol coverage, hand dynamics, dynamic GUI, capture/input bridge, policy training, and long-horizon memory evaluation remain pending. The next milestone is an articulated hand/arm CPU scene with measured control/contact behavior, followed by dynamic GUI integration and a 200–500 trajectory dynamics-data pilot.

The user explicitly clarified operational ownership: Kimi performs installation, VM/server deployment, runs, training, and live streaming. Codex defines architecture/contracts/budgets, coordinates bounded tasks, reviews source/evidence and live views read-only, and integrates documentation/Git changes. Current runtime work is split into CPU scene completion, native GUI correction, and remote compute/live view. No policy training result has been established.
