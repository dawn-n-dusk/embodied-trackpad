"""Standalone screen-local view preparation for LEAP v1 visual-policy input adaptation.

SOURCE-ONLY helper: builds an explicit full view plus a native-pixel 1:1
224x224 local view centred on the cursor from a single decoded confined-app
capture. It exists because a full-frame 224x224 resize loses cursor detail;
the local view copies pixels without any resampling so the mapping stays
invertible via ``origin_col_row``:

    local_rgb[r, c] == full_rgb[origin_row + r, origin_c + c]

wherever ``local_pixel_valid[r, c]`` is True; outside the image the local view
is zero-padded and the mask is False. The crop origin is NEVER shifted near an
edge, so the pixel mapping stays exact even when most of the window is
padding.

Contract and limits:

* Accepts ONLY an ``np.ndarray`` of dtype uint8 with shape (600, 800, 3) — the
  confined TrackpadSandbox capture. Anything else raises TypeError/ValueError.
* The cursor position is derived ONLY from the current RGB pixels via the
  existing frozen fixed-palette extractor
  (:func:`embodied_trackpad.leap_v1.visual_servo.extract_visual_estimate`,
  default config only). Image col/row (top-left origin) are used, not logical
  y. No GUI API, cursor-state oracle, or target/oracle coordinate is read,
  and the target estimate does NOT gate the crop — a missing target leaves the
  local view unchanged as long as the cursor is usable.
* ``local_valid`` requires the extractor's cursor to be found with finite
  col/row/confidence and confidence >= 0.5 (the frozen default colour
  confidence; there is deliberately no tunable permissive threshold). Missing,
  ambiguous/spread-out or low-confidence cursors yield an all-zero local view,
  an all-False mask, ``local_valid=False`` and ``origin_col_row=None`` —
  no stale fallback, while ``full_rgb`` stays available.
* All returned arrays are freshly allocated: independent from the input and
  from each other.
* Limitations inherited from the extractor: this is fixed-threshold colour
  localisation calibrated to this app's systemRed cursor palette only. It is
  NOT general GUI vision, cannot handle arbitrary apps/themes/occlusion, and
  makes no wrist-camera claim. Every estimate carries the extractor
  confidence; low confidence must stop the consumer rather than act.
* This module is NOT wired into the pi0.5 inputs, tokenizer or image masks,
  and it makes no learning-performance claim. It only prepares an explicit
  screen-local view; camera-slot mapping for the visual backbone is a
  separately reviewed integration step owned by the coordinator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from embodied_trackpad.leap_v1.visual_servo import extract_visual_estimate

CAPTURE_WIDTH = 800
CAPTURE_HEIGHT = 600
EXPECTED_SHAPE = (CAPTURE_HEIGHT, CAPTURE_WIDTH, 3)

LOCAL_SIZE = 224  # pi0.5-style native-pixel window; NO resize is applied
HALF_LOCAL = LOCAL_SIZE // 2  # 112
MIN_CURSOR_CONFIDENCE = 0.5  # frozen default colour confidence, not tunable

__all__ = [
    "CAPTURE_HEIGHT",
    "CAPTURE_WIDTH",
    "EXPECTED_SHAPE",
    "HALF_LOCAL",
    "LOCAL_SIZE",
    "MIN_CURSOR_CONFIDENCE",
    "ScreenViews",
    "prepare_screen_views",
]


@dataclass(frozen=True)
class ScreenViews:
    """Full capture plus cursor-centred native-pixel local view.

    ``full_rgb`` is always an exact independent copy of the accepted input.
    ``local_rgb`` is the 224x224 cursor-centred native-pixel copy (zero-padded
    outside the image), valid only when ``local_valid`` is True. When valid,
    ``origin_col_row`` is the integer (col, row) of the local view's top-left
    corner in full-image pixels and ``cursor_col_row`` is the pixel-derived
    float (col, row) of the cursor; both are None when invalid.
    """

    full_rgb: np.ndarray  # (600, 800, 3) uint8, exact independent copy
    local_rgb: np.ndarray  # (224, 224, 3) uint8; zeros when invalid
    local_valid: bool
    local_pixel_valid: np.ndarray  # (224, 224) bool; False where padded/invalid
    origin_col_row: tuple[int, int] | None  # integer (col, row) in full image
    cursor_col_row: tuple[float, float] | None  # pixel-derived (col, row)
    cursor_confidence: float  # extractor confidence (0.0 when not found)


def _reject_invalid_input(rgb: object) -> np.ndarray:
    if not isinstance(rgb, np.ndarray):
        raise TypeError(f"rgb must be a numpy ndarray, got {type(rgb).__name__}")
    if rgb.dtype != np.uint8:
        raise ValueError(f"rgb must have dtype uint8, got {rgb.dtype}")
    if rgb.shape != EXPECTED_SHAPE:
        raise ValueError(f"rgb must have shape {EXPECTED_SHAPE}, got {rgb.shape}")
    return rgb


def _cursor_usable(cursor: object) -> bool:
    return (
        bool(cursor.found)
        and cursor.col is not None
        and cursor.row is not None
        and math.isfinite(float(cursor.col))
        and math.isfinite(float(cursor.row))
        and math.isfinite(float(cursor.confidence))
        and float(cursor.confidence) >= MIN_CURSOR_CONFIDENCE
    )


def prepare_screen_views(rgb: np.ndarray) -> ScreenViews:
    """Build the full + screen-local views for one decoded 800x600x3 capture.

    Derives the cursor centre from the current RGB pixels only (frozen
    fixed-palette extractor, default config) and copies a native-pixel 1:1
    224x224 window centred at ``(floor(col) - 112, floor(row) - 112)``. The
    crop is never shifted at image edges: out-of-image area is zero-padded and
    masked False, keeping the explicit origin invertible. A missing,
    spread-out or low-confidence cursor yields an all-zero local view with
    ``local_valid=False`` and no cursor/origin reported — never a stale
    fallback. No GUI API is called and no target/oracle coordinate is read.
    """
    rgb = _reject_invalid_input(rgb)
    full_rgb = np.array(rgb, dtype=np.uint8, copy=True)

    est = extract_visual_estimate(rgb)  # frozen extractor, default config only
    cursor = est.cursor  # target estimate intentionally does NOT gate the crop

    if not _cursor_usable(cursor):
        return ScreenViews(
            full_rgb=full_rgb,
            local_rgb=np.zeros((LOCAL_SIZE, LOCAL_SIZE, 3), dtype=np.uint8),
            local_valid=False,
            local_pixel_valid=np.zeros((LOCAL_SIZE, LOCAL_SIZE), dtype=bool),
            origin_col_row=None,
            cursor_col_row=None,
            cursor_confidence=float(cursor.confidence),
        )

    col = float(cursor.col)
    row = float(cursor.row)
    origin_col = math.floor(col) - HALF_LOCAL
    origin_row = math.floor(row) - HALF_LOCAL

    local_rgb = np.zeros((LOCAL_SIZE, LOCAL_SIZE, 3), dtype=np.uint8)
    local_pixel_valid = np.zeros((LOCAL_SIZE, LOCAL_SIZE), dtype=bool)

    # Overlap of the explicit crop window with the image; the origin is never
    # clamped or shifted, so dst indices recover the full-image pixels exactly.
    src_c0 = max(0, origin_col)
    src_c1 = min(CAPTURE_WIDTH, origin_col + LOCAL_SIZE)
    src_r0 = max(0, origin_row)
    src_r1 = min(CAPTURE_HEIGHT, origin_row + LOCAL_SIZE)
    if src_c1 > src_c0 and src_r1 > src_r0:
        dst_c0 = src_c0 - origin_col
        dst_r0 = src_r0 - origin_row
        dst_c1 = dst_c0 + (src_c1 - src_c0)
        dst_r1 = dst_r0 + (src_r1 - src_r0)
        # Native-pixel direct copy; no resampling anywhere in this module.
        local_rgb[dst_r0:dst_r1, dst_c0:dst_c1] = rgb[src_r0:src_r1, src_c0:src_c1]
        local_pixel_valid[dst_r0:dst_r1, dst_c0:dst_c1] = True

    return ScreenViews(
        full_rgb=full_rgb,
        local_rgb=local_rgb,
        local_valid=True,
        local_pixel_valid=local_pixel_valid,
        origin_col_row=(origin_col, origin_row),
        cursor_col_row=(col, row),
        cursor_confidence=float(cursor.confidence),
    )
