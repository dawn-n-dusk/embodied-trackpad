# LEAP v1 live viewer (local-only)

`tools/leap_v1_live_view.py` serves a small HTTP page bound to **127.0.0.1**
that displays freshly rendered MuJoCo frames of the pinned LEAP Hand v1 Full
model (16 independent channels, four fingers; ADR 0007) while the existing
diagnostics script drives all channels through `ctrl` + `mj_step`.

Scope (shown in the UI at all times): **scripted physics; hand only; no learned
policy**; no trackpad contact; uncalibrated upstream defaults; no prerecorded
frames. This is a visualization aid, not a task demonstration.

## Run

Requires a Python environment with MuJoCo and the `embodied_trackpad` package
importable (the pinned v1 asset cache must be present and manifest-valid):

```
PYTHONPATH=src python tools/leap_v1_live_view.py \
    --port 8768 --lifetime-s 1200 --fps 8 --out-dir .local/leap-v1-live
```

Then open `http://127.0.0.1:8768/` locally. If 8768 is taken, the next ports
are tried (no existing listener is ever killed). `--lifetime-s` is a hard,
server-enforced maximum (validated finite, `(0, 1200]`); at the deadline the
physics stops, the port is released, and `result.json` is written to the
out-dir. SIGTERM/SIGINT trigger the same clean shutdown.

## Endpoints and freshness

- `GET /` — HTML page: viewport plus run id, episode, phase, sim time,
  displayed frame id/age, measured fps, deadline countdown, status badge.
- `GET /frame.png` — newest rendered 640x480 PNG with `X-Frame-Id`,
  `X-Frame-Sha256`, `X-Frame-Rendered-Unix`, `X-Sim-Time-S` headers.
- `GET /state.json` — run identity, episode/sim time/frame/fps, `label`,
  `status` computed from rendered-frame age (2 s staleness threshold).
- `GET /stream` — gone (410): the display transport is per-frame polling.

Freshness is physics-based, not heartbeat-based. The page fetches `/frame.png`
at ≤8 fps with a bounded timeout and no overlapping requests; **only a newly
decoded, displayed frame advances freshness**. An independent watchdog shows
STALLED after >2 s without a new displayed frame (even if `/state.json` keeps
answering) and STOPPED when the server is unreachable; polling retries
automatically. Nothing prerecorded is ever labeled LIVE.

## Physics path

`leap_v1.load_hand` (+ a skybox/headlight spec hook for a bright background),
`open_pose_qpos` at episode reset only (the sole live qpos write), then
`diagnostics.build_script` phases (per-channel sweep, spread, flex/return;
22 s episode, loops with a reset) applied via `apply_ctrl_checked` and
`mj_step` at the model timestep, wall-clock paced to roughly realtime. One
process, numerical-thread env caps forced to 2, render 640x480 at ≤8 fps.

## Receipts

Private run receipts (logs, captured frames, state samples, checksums,
`validation.json`, final `result.json`) live in the git-ignored
`.local/leap-v1-live/` directory. Protocol tests:
`tests/test_leap_v1_live_protocol.py` (status boundaries, state schema, HTML
freshness contract, PNG encoder, script interpolation, arg validation).
