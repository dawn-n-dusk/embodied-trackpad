# Foundation package (M0)

`embodied_trackpad` is the first runnable foundation package for Embodied
Trackpad. It implements protocol v2 (draft) data contracts, runtime
validation, an append-only episode recorder, and a deterministic mock
dry-run CLI. Requires Python >= 3.10; runtime code uses only the standard
library.

## What this package is not

The dry-run output is an explicit **MOCK / SOFTWARE FIXTURE** with
`training_eligible=false`. It is:

- NOT a MuJoCo dynamics rollout (no physics is simulated);
- NOT robot training data;
- NOT a successful macOS interaction (no screen capture, no input injection);
- NOT a long-horizon task result.

No VLM weights, no training algorithms, no hand assets, no serial/camera
access, and no macOS permission handling exist in this package.

## Layout

- `src/embodied_trackpad/contracts.py` — typed/versioned messages:
  `EpisodeConfig`, `FramePacket`, `MotorIntent`, `ActionChunk`,
  `AttemptOutcome`, `MemoryDelta` (`schema_version = "trackpad.v2-draft"`).
- `src/embodied_trackpad/validation.py` — runtime checks: wrong
  embodiment/actuator layout or shape, nonfinite or out-of-range values,
  expired or wrong-goal/observation/calibration messages, and clock-domain
  handling. Monotonic times are compared only inside one declared
  `clock_id` domain; cross-host frame ages are never invented. Zero-valued
  action rows are valid; executability comes only from `valid_mask`.
  Each chunk binds the canonical `actuator_layout_id` derived from the
  ordered actuator names, so equal-width reordered layouts are rejected.
  Live admission requires `now` inside the half-open interval
  `[start, start + horizon * dt)`, and before any explicit expiry. Omitting
  `now` performs structural/historical checks only; it is not live admission.
  Actions are actuator position references — coordinate-click commands are
  rejected by construction.
- `src/embodied_trackpad/recorder.py` — append-only recorder. Immutable
  `config.json`/`provenance.json`, sequence-numbered actor events,
  separate teacher/evaluator streams, strict actor contract parsing that
  rejects unknown fields in defined structures, explicit
  `evidence_class` (`mock_software_fixture` / `physics_simulation` /
  `real_device`), and a sha256 `manifest.json` (no self-hash). Failed or
  interrupted attempts stay recorded; success is never inferred from event
  emission. Observation and intent references must resolve within the
  episode; `observed_success` requires recorded evidence frame ids.
  Reusing one frame for several decisions is supported; redefining its id
  is rejected. Episode ids must be safe single path components. Mock
  fixtures cannot be marked training eligible.
- `src/embodied_trackpad/cli.py` — deterministic dry-run (`--seed`),
  including built-in negative checks that must all be rejected.
- `schemas/` — JSON Schema (draft 2020-12, one consistent draft) mirrors
  of the contracts.
- `tests/` — behavior tests for the boundaries above.

## Install

```bash
python3 -m venv .venv          # any CPython >= 3.10
.venv/bin/python -m pip install -e '.[dev]'
```

## Test

```bash
.venv/bin/python -m pytest -q
```

## Demo (mock dry-run)

```bash
.venv/bin/python -m embodied_trackpad.cli \
  --out runs/demo --seed 7 --steps 8 --created-utc 2026-10-30T00:00:00Z
```

This writes `runs/demo/dryrun-seed7/` (episode config, provenance, actor /
teacher / evaluator JSONL streams, checksum manifest) and
`runs/demo/dryrun-seed7.fixture_manifest.json` (seed, checksums, negative
check results, disclaimers, `training_eligible=false`).

Reproducibility check (same seed and pinned timestamp → identical bytes):

```bash
.venv/bin/python -m embodied_trackpad.cli --out /tmp/et-a --seed 7 --steps 8 \
  --created-utc 2026-10-30T00:00:00Z
.venv/bin/python -m embodied_trackpad.cli --out /tmp/et-b --seed 7 --steps 8 \
  --created-utc 2026-10-30T00:00:00Z
diff -r /tmp/et-a /tmp/et-b   # no output = byte-identical
```

## CI

`.github/workflows/ci.yml` is configured to run behavior tests on Python
3.10/3.11/3.12/3.13 (CPU only, no GPU) plus the dry-run reproducibility
check above. Consult [current status](../STATUS.md) for observed CI results.

## Recording lifecycle

Every accepted event is flushed. An exception in a recorder context closes
the manifest with `status="interrupted"`; a hard process exit can leave
readable partial JSONL without a finalized manifest. Flush is not a claim
of power-loss durability (`fsync` is not performed). `completed` means
recording closed normally, not that a GUI task succeeded. The seed-7 demo
includes an interrupted attempt inside an otherwise normally closed log.

The corrected package passed 89 local tests and 12 built-in negative
checks. Two seed-7 runs with a fixed timestamp produced identical files.
The first 54-test pass missed important cases; the review and correction
history are retained in [the experiment record](../experiments/records/m0-foundation.json).

## Boundaries and known limitations

- Validation covers message-level semantics only; nothing here proves
  physical executability, contact correctness, or task success.
- `image_ref` is an opaque reference; this package never captures or
  stores real screen images.
- Teacher/evaluator separation is enforced by separate streams and strict
  actor contracts, not by process isolation or semantic inspection of
  image/text contents. A future actor data loader must consume only the
  intended stream.
- Clock domains are identified by string `clock_id`; real cross-host
  synchronization (with uncertainty) is future bridge work.
- The six contracts above are a first protocol subset. Task-context,
  semantic-goal, and motor-observation runtime components from the full
  architecture are still pending; there is no action executor or policy.
