# LEAP physics contacts → native Mac test GUI

2026-10-31 acceptance: Kimi executed the frozen 105-frame contact stream from the accepted LEAP pad cycle in the signed `TrackpadSandbox` app. The coordinator reviewed source and hashes, every acknowledgement, all four actual own-window PNGs, and an independent pixel-displacement measurement. The [sanitized record](../experiments/records/leap-native-contact-20261031.json) retains timestamps, input/source/binary hashes, outcomes and delegation accounting.

This establishes **offline physics-contact replay**. The physics episode was already complete when replay started; screen observations did not influence its motor actions. No D1 dataset or learned policy is claimed.

## Observed result

| Check | Observed |
|---|---|
| Normal-role wire | 105/105 accepted; acknowledgements contain only `type` and `sequence` |
| Separate evaluator wire | 105/105 accepted in a second app process |
| Captures | Actual 800 × 600 PNGs: before, press, slide, release |
| Press/release | Press held at the mark; one generic release; final button up |
| Cursor displacement | +20.583877 px x, −0.941283 px y in the app's +y-up coordinates |
| Contact vs evaluator delta | Maximum difference < 7e−14 px; common device-model arithmetic, not hardware accuracy |
| Independent visible delta | +20.357143 px x, +0.836134 px y in top-left image coordinates |
| Visible delta error | 0.226735 px x / 0.105148 px y, within the declared 0.5 px raster tolerance |
| Press appearance | Red cursor pixels increased from 116 to 232; released image had 119 |
| Task result | Target hits: 0; drag completion: false |
| Teardown | Both app children exited with code 0 and were reaped |

The pixel analysis used the red ring in crop x=[300,500), y=[200,400), with RGB thresholds R>220, G<110, B<120. Antialiasing limits centroid precision. The coordinator also visually inspected each complete image. The local MuJoCo environment lacked Pillow, so the read-only image analysis used the existing bundled Python runtime; no package was installed.

## Preserved action provenance

The source chain is motor references → CPU dynamics → raw pad contacts → force-weighted `touch.v1` frames → native device model → own-window pixels. Four frozen physics input hashes match the separately accepted pad run. The replay adapter changes only the transport envelope: recorded sequence 0–104 becomes wire sequence 1–105 and `cmd="touch"` is added. Finger identity, coordinates, forces and simulation timestamps are unchanged.

Normal-role evidence contains no evaluator states. A second explicitly privileged process replays the same input and stores evaluator states separately. Actual monotonic capture timestamps and originating simulation timestamps are recorded independently; approximately real-time replay does not synchronize the two clocks or establish latency performance.

## Reproduce this diagnostic

Build the app using the [native GUI guide](mac-sandbox.md) and produce the accepted c01 artifacts using the [pad guide](leap-pad.md). Pass all paths explicitly from a checkout:

```bash
python tools/leap_native_contact_replay.py \
  --cand-dir "$ACCEPTED_PAD_CANDIDATE" \
  --pad-report "$ACCEPTED_PAD_REPORT" \
  --binary "$TRACKPAD_SANDBOX_BINARY" \
  --out .local/native-contact-new-run
```

Use a fresh output directory. The script writes its composite report under `.local/`, normal evidence/captures under the chosen output directory, and privileged state in a separate `evaluator.json`. It verifies signing and checks the recorded gates before launching. The pinned binary hash applies to the original default artifact; a freshly built explicitly supplied binary is recorded with its own hash.

This CLI is deliberately specialized to the accepted 105-frame c01 fixture. Its cursor expectation assumes one finger and this device gain, with no edge clipping. Do not reuse it as a generic multi-finger evaluator. Future online environments need their own reset, observation/action timing and failure gates.

## Remaining boundary

The narrow contact-force margin and limited fixed-palm movement remain properties of the original physics run. Native Apple pinch/rotation/Force Touch, a physical capacitive fingertip, calibrated Metal-hand dynamics and autonomous screen-feedback tasks remain unverified. See the [next online pilot contract](tasks/leap-online-pilot.zh-CN.md).

Three Kimi dispatches took 753.8 summed seconds, including two timeouts before app execution. Two dispatches reported 92,909 uncached input, 805,120 cached input and 24,122 output tokens; the other dispatch's usage is unknown. These counts are not billing or a measured delegation saving. The final corrected dispatch executed both native runs in about 7.65 seconds of combined app-loop time.
