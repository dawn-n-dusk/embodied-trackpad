# LEAP v2 Advanced static-pad contact environment

Physics-only milestone: the accepted 17-motor LEAP v2 Advanced hand performs a
clear → press → hold → tangential-slide → lift cycle against a STATIC
160 × 110 mm pad. The native Mac GUI is NOT launched here; touch.v1 messages
are recorded for a later, separately reviewed bridge.

## Scene

`src/embodied_trackpad/leap_sim/pad_env.py` extends the accepted
`load_hand(spec_hook=...)` with ONE static worldbody box geom
(`trackpad_160x110`, 160 × 110 × 10 mm). The accepted hand model is unchanged:
same URDF, 21 joints / 17 motors / 4 coupling equalities, 8
mechanical-neighbor exclusions, same servo constants; no collision is
disabled. The pad is an ordinary colliding worldbody geom (static by
construction, verified immobile every step).

Fixed mount: pad centre `(0.0, -0.05, -0.177)` → top face at z = −0.172 m.
Measured mesh surface survey (`.local/explore5.py`): the lowest open-pose hand
surfaces (thumb/pip undersides) reach −0.1546 m, so reset clearance is ~17 mm;
the distal index tip (`dip_tip_soft_upward_cut`) is the only body that can
reach the pad when flexed (mcpf ≥ 0.10 with dip 0.45), while the index middle
phalanx stays ~10 mm clear. An earlier mount at −0.152 m intersected the
proximal/middle finger bodies at reset (tiny-002, 30684 unintended contact
events) and was rejected; failed dirs are preserved unchanged under
`.local/leap-pad/` and `.local/superseded/`.

## Episode and gates

`tools/leap_pad_probe.py` runs at most 16 candidates within a 300 s physics
wall-clock budget, fresh output directory per candidate, stopping at the first
PASS. All 17 ctrl channels are commanded every step (piecewise-linear schedule);
the only live qpos write is the reset. After every `mj_step`, `mj_forward`
aligns observations with `d.time`; per-step `t_ctrl` is recorded alongside.

Gates (bounds fixed before execution): finite state/control/forces (raw
contact force/pos/frame/dist validated BEFORE aggregation — NaN can never
vanish through the `> 0` filter), no solver warnings, joint-limit excursion ≤
0.01 rad, coupling residual ≤ 0.01 rad, raw ctrl bounds enforced pre-write,
penetration ≤ 1 mm, total positive normal force over ALL pad contacts ≤ 5 N
(eligible tip force and unintended force reported separately), no
non-tip/edge/backside/hand-self contact, measured ≥ 0.3 s initial and final
clearance windows (never vacuous), one named tip sustaining ≥ 0.05 N for
≥ 0.3 s with ≥ 5 mm measured tangential centroid displacement while contact
is continuous, hold tracking ≤ 0.10 rad for unblocked joints
(contact-constrained joints reported separately), pad immobile, and a
continuous bounded-rate touch.v1 stream (≤ 0.05 s frame interval, monotonic
sequence/time, empty clear/lift frames included). Device thresholds
(DeviceModel: press 1.0 N, release 0.6 N) are measured and reported, not
redefined. Unintended contacts are counted as unique STEPS and as EVENTS
separately.

## Measured result (candidate c01, mcpf 0.12 / dip 0.45 / mcps −0.22)

Executed twice with identical outcomes (`.local/leap-pad/run-c01-c03/cand-c01`
and probe-authored `.local/leap-pad/run-accept-001/cand-c01`): **all gates
PASS**. Peak tip force 1.054 N (crosses the 1 N press threshold, releases to
0), max penetration 0.125 mm, sustained contact 1.8145 s, slide 5.13 mm net,
no unintended contact steps/events, hold tracking (unblocked) ≤ 0.046 rad,
7600/7600 steps, ~1.2 s physics wall per episode. Force and slide margins
above their thresholds are small (0.054 N, 0.13 mm) — reported, not rounded.
Raw per-step contacts (full force6 + frame), touch.v1 stream, sampled full
state trace and initial/press/slide/lift renders from executed state are in
each candidate dir; `.local/pad-report.json` has hashes/versions/budget.

## Limitations

- Physics only: no GUI/GPU/training; touch.v1 recorded, not delivered.
- Placeholder servo gains and CAD inertials: force values are not
  hardware-calibrated. Rigid meshes and 1:1 PIP/DIP coupling remain
  approximations (see `docs/leap-model.md`).
- Single scripted index-finger cycle; hand-only replay is not training data.

## Tests

`tests/test_leap_pad.py` (13 tests): scene integrity vs the accepted hand,
static pad, five-finger tip mapping, contact-free reset, finite-value guard
incl. NaN regression, aggregation/classification math, touch.v1 shape, PNG
roundtrip incl. the broken-writer regression, SMOKE-episode invariants, and
candidate-grid bounds. Skip rule matches `test_leap_sim.py` (skips only when
MuJoCo/assets are genuinely absent).
