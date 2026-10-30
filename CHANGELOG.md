# Changelog

## Unreleased — 2026-10-30

- Accepted architecture v2: articulated hand, visual closed-loop operation, multi-scale memory, and short action chunks.
- Established a dedicated public project boundary, reproducibility policy, and append-only experiment records.
- Preserved sanitized historical design snapshots and authoritative source links.
- Began implementation planning for protocol/data foundations and a bounded H800 environment probe. Neither is yet an accepted runtime result.
- Published the initial public repository and verified its remote commit. Dispatched isolated Kimi foundation and GPU-probe work.
- Specified staged data acquisition budgets, provenance/quality gates, and a dedicated M1 Mac setup plan.
- Verified and integrated Kimi's single-H800 MuJoCo Warp probe: 12/12 checks, 1/32 worlds, exact executed-source and result hashes matched independently. Retained the initial device-API failure and correction.
- Corrected the probe's interpretation: batching increased GPU aggregate throughput, but no GPU-over-CPU crossover was measured in this tiny scene.
- Reviewed the initial foundation package (54 tests pass) and returned independently reproduced boundary gaps for correction; runtime acceptance remains pending.
- Integrated the corrected protocol/recording foundation after 89 local tests, 9 independent boundary checks, and deterministic mock replay; 12 negative CLI cases are rejected.
- Bound actions to actuator layout and execution windows; enforced observation/evidence references, strict actor contracts, safe episode paths, mock-data exclusion, and explicit interrupted recording.
- Preserved the correction worker's 20-minute timeout and coordinator verification-harness correction. Coordinator completed documentation/provenance; production code remains Kimi's implementation.
- Added runnable setup instructions, protocol-integrity ADR, CI matrix for Python 3.10–3.13, and measured delegation/rework records. CI execution is tracked separately in STATUS.md.
