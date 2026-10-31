# Normalized native capture resolution (800x600)

Status: implemented and locally validated on the M5 test host (2026-10-31,
app v0.4). This records the observation-resolution decision for the
visual-control/data plan.

## Problem

The TrackpadSandbox window and all device/window coordinates are **logical**
800x600 units (touch wire, virtual cursor, layout, privileged state). The
first capture implementation (`bitmapImageRepForCachingDisplay` +
`cacheDisplay`) sized the bitmap in **backing-store pixels**, so on the
Retina M5 host (backing scale 2) the emitted PNG was 1600x1200 while the
declared observation contract and all coordinates stayed logical 800x600 —
`test_capture_shape` failed with `width == 1600 != 800`.

## Decision (coordinator-agreed)

Preserve the 800x600 observation API by **deliberately rendering only this
app's own content view to a fixed 800x600 pixel bitmap**, independent of the
host Retina backing scale:

- `capture()` allocates an `NSBitmapImageRep` of exactly 800x600 pixels and
  draws `SandboxView` into it with 1 logical unit = 1 pixel.
- Declared `width`/`height` in the observation message always equal the
  actual PNG IHDR dimensions.
- Scale/aspect and virtual-cursor/target placement in image pixels match
  logical coordinates: image col = logical x, image row = (600 - logical y)
  (the view is y-up; PNG rows run top-down).
- Device/window logical coordinates, the touch.v1 input wire, the actor
  acknowledgement, and the privileged evaluator state are unchanged. No
  host/global capture exists.

This is an intentional **normalized observation** — a fixed-resolution render
of the app's own view for visual-control observations — explicitly **not** a
native full-resolution screen capture.

## Verification (this host, ad-hoc signed sandboxed app)

- Actual PNG IHDR decoded independently of the app: 800x600 for every
  capture (`test_capture_shape`).
- Cursor displaced (+32, +16) logical px via the allowed touch protocol; the
  red cursor-pixel centroid in the decoded bitmap moved (+32, -16) px
  (row axis inverted vs view y), matching the logical delta within 2 px
  (`test_capture_cursor_pixels_track_logical_movement`).
- Real own-window PNGs, binary/source hashes, and run receipts are recorded
  in the private evidence directory `.local/native-capture-normalization/`
  and the coordinator-facing experiment record.

Untested scale conditions: only the M5 Retina (2x) host is exercised; a 1x
(non-Retina) or 3x host is not available here, but the bitmap size is fixed
in code and does not depend on the backing scale factor.
