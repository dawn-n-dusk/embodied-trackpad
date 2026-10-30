# LEAP v2 Advanced simulation adapter

The selected target is the complete right **LEAP Hand v2 Advanced**: five fingers, 17 motor channels, 21 articulated joints, including two active palm joints. The public CAD export does not establish calibrated Metal-build dynamics. Source selection and future hardware boundaries are in [ADR 0006](decisions/0006-latest-leap-and-hardware-readiness.zh-CN.md).

This is an experimental fixed-base hand adapter. Latest measured acceptance is in [STATUS.md](../STATUS.md) and the experiment ledger. A successful import, a numerical probe, contact/GUI behavior, policy learning and physical deployment are separate results.

## Pinned assets

Upstream: [LEAP_Hand_V2_Adv_API](https://github.com/leap-hand/LEAP_Hand_V2_Adv_API/tree/a0936196e8fa0bcefff045d6fc02523d6da591f2), commit `a0936196e8fa0bcefff045d6fc02523d6da591f2`.

```bash
python3 tools/fetch_leap_assets.py
python3 tools/fetch_leap_assets.py --check
```

The fetcher verifies the committed 13-file manifest and never regenerates its hashes from newly downloaded bytes. Only the explicit right-hand URDF/STL, README and license whitelist is permitted. Paths below the chosen download root are checked before use; downloads use unique temporary files and verified replacement. The selected resolved root is the trusted anchor; concurrent hostile filesystem mutation is outside the tested boundary.

Models and derived files stay under ignored `assets/downloads/`. The nested upstream software license is Apache-2.0; the official CAD page separately lists CC BY-NC-SA. This repository does not infer blanket relicensing or redistribution rights for individual assets. See [THIRD_PARTY.md](../THIRD_PARTY.md).

## Model adaptations

- Preserve the original URDF and matching mesh set. Rewrite local mesh references only in derived files; do not mix the website ZIP geometry with GitHub geometry.
- Convert an STL exceeding MuJoCo's decoder face limit to OBJ with float32-roundtrip coordinate precision. Derived-cache metadata binds source, converter and output hashes. Decimal-to-float64 deviation is distinct from float32 roundtrip identity.
- Replace ten auxiliary `realtip`/`realdip` links with massless sites. Their exported 0.1 kg masses are not retained as physical hand mass.
- Preserve the fixed root as a named body. Corrected CAD mass is 1.505 kg: 0.140 kg fixed root and 1.365 kg moving links. These are export-derived accounting values, not a measured hardware weight.
- Add four explicit per-finger PIP/DIP equality constraints. The 1:1 rigid relation is an approximation of tendon coupling; it does not model soft-finger deformation, slack, hysteresis or backlash.
- Keep 17 position-servo channels in physical motor order. The four PIP joints are coupled, not four additional independently driven motors. Control gains/damping remain simulation placeholders under the upstream effort/range bounds.
- Use eight documented body-pair exclusions to address CAD-convex-hull overlaps at assembly interfaces. These are temporary collision approximations, including relatively moving components. They do not establish accurate geometry throughout the workspace. No exclusion should refer to `world`, because that would also affect subsequently added world-mounted objects.

The final reviewed run passed 34 focused tests and all 14,000-step gates: coupling residual 0.00754 rad, contact penetration 0.449 mm, open-pose error 0.0371 rad, and held-phase tracking error 0.0373 rad. These are debug thresholds and measurements, not hardware accuracy claims. [The complete record](../experiments/records/leap-v2-cpu-20261031.json) includes failed candidates and delegation accounting.

The initial imported collision model had 18 reset contact points with up to approximately 30 mm penetration. A later numerical motion probe removed joint-limit violations and reduced coupling residuals, but weak servos left approximately 0.64 rad of open-pose error. Those results are retained as failures of the stronger motion-readiness contract; see [the debug record](../experiments/records/leap-v2-import-debug-20261031.json).

## Runtime and interpretation

The development CPU runtime uses Python 3.12, MuJoCo 3.17.0 and NumPy 2.5.4. Install optional simulation dependencies from [env/leap-requirements.txt](../env/leap-requirements.txt) in an isolated environment; the foundation package and asset downloader do not require MuJoCo.

```bash
python tools/leap_model_probe.py --out .local/leap-run-001
```

Use a fresh directory for each attempt. The probe applies motor references through `ctrl` and `mj_step`; only reset writes the initial joint state. It saves source/model hashes, state/control traces, contact diagnostics, numerical checks and rendered poses. Read the named gates and coordinator acceptance record rather than interpreting a `PASS` label as general task success.

The stronger hand-control acceptance requires joint-limit excursion and coupling residual at most 0.01 rad, finite states, no solver warnings, valid raw commands, and measured pose tracking. The implemented recovery gate is at most 0.05 rad after at least 0.5 s open hold; held-phase actuator tracking is at most 0.10 rad. Hardware calibration and actual pad contact require additional checks. Current hand-only diagnostic traces are not accepted project training data.

## Next interface

The [native replay milestone](leap-native-bridge.md) now connects recorded fingertip-pad contacts to the confined Mac GUI and verifies actual pixels. The [next online pilot](tasks/leap-online-pilot.zh-CN.md) must use new screen feedback to change subsequent motor actions. Requested motor references, executed states, contacts and observations must share explicit timestamps and model versions. The actor cannot bypass dynamics with software click coordinates. A contact trace or teacher replay is not a learned visual policy.

Future self-built LEAP hardware needs its actual BOM, motor directions/zeros, limits, coupling, latency, backlash and fingertip properties calibrated against this interface. A separate Apple Trackpad can be used with a connected Mac; a completely idle MacBook is not required for the present software work. Neither hardware option is connected or validated by this adapter.
