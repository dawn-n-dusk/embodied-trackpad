# Current status

Updated: 2026-10-30. Status is evidence-based, not inferred from a roadmap item.

| Milestone | State | Evidence / next condition |
|---|---|---|
| Architecture v2 | Accepted for implementation | User accepted the revised architecture on 2026-10-30; public design snapshot in docs |
| Public project | Published | Public code/design repository; publication and integration revisions are recorded in the append-only ledger |
| Protocol, recording, and dry-run pipeline | Independently validated locally | Six initial contracts, strict actor parsing, recording, schemas, CLI; 89 tests, 9 independent boundary checks, 12 CLI rejection cases; deterministic fixture output |
| GitHub CI | Awaiting first run | Workflow for Python 3.10–3.13 and deterministic CLI replay is included; local results do not substitute for CI evidence |
| MuJoCo CPU hand environment | Not validated | Compile selected embodiment and check contact/control |
| H800 / MJWarp | GPU environment probe verified | One H800, tiny 1-DOF contact scene at 1/32 worlds; 12/12 checks; coordinator matched scheduler/script/result evidence. No full-hand or GPU speedup claim |
| Datasets | No project training dataset yet | Begin dynamic GUI/robot demonstrations; preserve provenance |
| Training | Not started | Wait for environment/data acceptance |
| Dedicated Mac | M1 candidate identified | Exact memory/OS not required to start simulation; testbed setup pending |
| Real-Mac input and autonomous tasks | Not started | Dedicated testbed, bridge implementation, and measured verification required |

Historical resource audits describe available resource classes; they are not reservations. Initial work is bounded environment and implementation validation, not a full training campaign.

Verified GPU implementation commit: `8022a2db92de2bc577d55b05aebd88906ec4c83b`. The successful probe allocation completed and released its resources. The earlier failed device-API attempt remains recorded in [the report](docs/experiments/2026-10-30-h800-probe.md).

Verified foundation implementation commit: `c1dd62af52b76ec3f1d6fe7ab5863002deb57290`. The first 54-test pass did not establish acceptance; independently found gaps were returned to Kimi. Its correction call timed out after saving source/test changes. The coordinator independently verified those changes and completed documentation/records without modifying production code. See [the record](experiments/records/m0-foundation.json) and [development report](docs/experiments/2026-10-30-development-report.zh-CN.md).

All currently generated episodes are mock software fixtures, explicitly ineligible for training. Full protocol coverage, hand dynamics, dynamic GUI, capture/input bridge, policy training, and long-horizon memory evaluation remain pending. The next milestone is an articulated hand/arm CPU scene with measured control/contact behavior, followed by dynamic GUI integration and a 200–500 trajectory dynamics-data pilot.
