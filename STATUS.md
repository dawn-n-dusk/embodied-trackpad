# Current status

Updated: 2026-10-30. Status is evidence-based, not inferred from a roadmap item.

| Milestone | State | Evidence / next condition |
|---|---|---|
| Architecture v2 | Accepted for implementation | User accepted the revised architecture on 2026-10-30; public design snapshot in docs |
| Public project | Published | Public repository created; remote main matched `3e43e6c36cd10b315cda60f8f538403cf9ec3565` after initial push |
| Protocol, recording, and dry-run pipeline | Review corrections in progress | Initial 54 tests pass; independent boundary checks found gaps, returned to the same Kimi session |
| MuJoCo CPU hand environment | Not validated | Compile selected embodiment and check contact/control |
| H800 / MJWarp | GPU environment probe verified | One H800, tiny 1-DOF contact scene at 1/32 worlds; 12/12 checks; coordinator matched scheduler/script/result evidence. No full-hand or GPU speedup claim |
| Datasets | No project training dataset yet | Begin dynamic GUI/robot demonstrations; preserve provenance |
| Training | Not started | Wait for environment/data acceptance |
| Dedicated Mac | M1 candidate identified | Exact memory/OS not required to start simulation; testbed setup pending |
| Real-Mac input and autonomous tasks | Not started | Dedicated testbed, bridge implementation, and measured verification required |

Historical resource audits describe available resource classes; they are not reservations. Initial work is bounded environment and implementation validation, not a full training campaign.

Verified GPU implementation commit: `8022a2db92de2bc577d55b05aebd88906ec4c83b`. The successful probe allocation completed and released its resources. The earlier failed device-API attempt remains recorded in [the report](docs/experiments/2026-10-30-h800-probe.md).
