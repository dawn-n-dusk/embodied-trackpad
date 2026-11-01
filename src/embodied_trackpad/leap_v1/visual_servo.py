"""LEAP v1 Full visual-servo INPUT UTILITIES (pixel/clock/gate layer only).

This module currently implements ONLY the observation-side utilities of the
planned v1 Full online visual-servo loop (four fingers, 16 independent motor
channels, upstream uncalibrated gains):

* :func:`decode_png_bytes` — pure-stdlib+numpy decoder for the confined
  TrackpadSandbox's own-window PNG captures (8-bit RGB/RGBA, non-interlaced,
  all five per-row filter types). Dimensions and decompressed length are
  bounded; truncation, CRC mismatch and size mismatch are rejected.
* :func:`extract_visual_estimate` — cursor/target localisation from decoded
  pixels via FIXED-THRESHOLD colour masks calibrated to this app's known
  palette (systemRed cursor, systemBlue target in a deviceRGB render). This
  is NOT general GUI vision: it cannot handle arbitrary apps, overlapping
  same-colour content, theme changes, or occlusion, so every estimate carries
  a confidence and spread-out/too-small matches report ``found=False``.
* :class:`ClockReconciler` — the app capture clock (``CACurrentMediaTime``)
  is reconciled against Python ``time.monotonic`` by ONE explicit measured
  offset with a measured round-trip uncertainty; the two clocks are never
  assumed to share a basis. The calibration timestamp is validated and the
  uncertainty is quantified, never dropped.
* :class:`FrameGate` — accepts only fresh, strictly newer, non-future frames
  whose capture metadata matches the decoded image dimensions and whose
  cursor/target estimates are found, finite and confident enough. Frame-age
  acceptance uses the worst-case age INCLUDING the clock uncertainty, so the
  uncertainty is never silently ignored. Rejection reasons are returned, not
  raised. There is exactly one reference clock contract here (the measured
  offset above); no GUI/cursor state truth is consumed anywhere.

Planned but NOT implemented in this module (no controller, runner, CLI or
episodes exist yet): actor-observation building and recording, the 16-channel
joint reference and 3x4 damped least-squares plan, physics stepping and
contact recording, wall-clock pacing, episode budgets and cleanup, and the
coordinator-accepted frozen contact manifest gate. ``CONTACT_PROFILE`` and
``ACCEPTANCE_*`` below are the frozen CONSTANTS that planned gate will hash;
they are data only — nothing here enforces them yet, and this module must
not be cited as evidence that any episode ran.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import time
import zlib
from contextlib import closing
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

# Pixel/logical mapping of the normalized 800x600 capture (see
# docs/native-capture-resolution.md): image col = logical x, image
# row = (height - logical y) because the view is y-up and PNG rows run
# top-down. One logical unit == one pixel.
CAPTURE_WIDTH = 800
CAPTURE_HEIGHT = 600

# Existing software device gain of the confined app (tools/mac_sandbox_client
# GAIN; Sources/DeviceModel.swift PadConfig): 4000 px per pad metre. This is a
# device CONFIG value read from source, not runtime GUI state truth.
DEVICE_GAIN_PX_PER_M = 4000.0
PAD_HALF_W_M = 0.080
PAD_HALF_H_M = 0.055

EXTRACTION_LIMITATIONS = (
    "fixed-threshold colour localisation calibrated to this app's systemRed "
    "cursor / systemBlue target palette only; NOT general GUI vision",
    "fails on occlusion, overlapping same-colour content, palette/theme "
    "changes and non-default cursor/target sizes; low confidence must stop "
    "the loop rather than act",
)

# Files whose frozen hashes the coordinator-accepted contact manifest must
# match. Paths are repository-relative; no machine-local paths are stored.
ACCEPTANCE_SOURCE_FILES = (
    "src/embodied_trackpad/leap_v1/visual_servo.py",
    "src/embodied_trackpad/leap_v1/pad_env.py",
    "src/embodied_trackpad/leap_v1/model.py",
    "src/embodied_trackpad/leap_sim/pad_env.py",
    "tools/leap_v1_visual_probe.py",
    "tools/mac_sandbox_client.py",
    "macos/TrackpadSandbox/Sources/main.swift",
    "macos/TrackpadSandbox/Sources/DeviceModel.swift",
)
ACCEPTANCE_MANIFEST_REL = Path("assets/manifests/leap-v1-full.json")
CONTACT_ACCEPTANCE_REL = Path(".local/v1-visual/contact-acceptance.json")

# Frozen contact/safety profile the coordinator accepts or rejects. Any edit
# changes the fingerprint and invalidates an existing acceptance manifest.
# ``scene_contact`` mirrors the reviewed P1 selection (.local/v1-visual/
# contact-profile-selection.json, allow_episode_execution=false): these are
# the ACTUAL physics parameters the loaded scene must carry — the acceptance
# receipt hashes them and the post-gate scene check compares them against the
# compiled model, so hashing thresholds alone is never enough.
CONTACT_PROFILE: dict[str, Any] = {
    "profile_id": "leap_v1_visual_servo_contact_v1",
    "touch_schema": "touch.v1",
    "press_force_n": 1.0,  # existing software device threshold (not an Apple measurement)
    "release_force_n": 0.6,
    "engage_force_min_n": 0.15,  # light sliding contact, below press threshold
    "engage_force_max_n": 0.8,
    "max_pad_normal_force_n": 5.0,
    "penetration_tol_m": 0.001,
    "limit_excursion_tol_rad": 0.01,
    "control_hz_max": 20.0,
    "max_frame_age_ms": 250.0,
    "max_future_frame_ms": 50.0,
    "min_colour_confidence": 0.5,
    "scene_contact": {
        "pad_geom": "trackpad_160x110",
        "solref": [0.02, 1.0],  # P1 selection (reviewed, uncalibrated)
        "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
        "friction": [0.5, 0.005, 0.0001],  # unchanged upstream geom defaults
        "timestep_s": 0.002,
        "selection_source": ".local/v1-visual/contact-profile-selection.json (P1)",
        "selection_allows_episode_execution": False,
    },
}


# ---------------------------------------------------------------------------
# PNG decoding. The app emits real AppKit PNGs (8-bit RGB or RGBA,
# non-interlaced, any per-row filter). Both backends share the strict
# structural + bounded-inflate validation in _validated_png_scanlines; the
# default "stdlib" backend reconstructs pixels in pure numpy, while the
# optional "pillow" backend (Pillow imported only when explicitly chosen)
# re-inflates via Pillow and skips the per-pixel Python loops. Pillow is an
# optional dependency (project "vision" extra); there is never a silent
# fallback between backends.
# ---------------------------------------------------------------------------

PNG_MAX_DIM = 4096


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _validated_png_scanlines(data: bytes) -> tuple[int, int, int, bytes]:
    """Strict structural + bounded-inflate validation shared by both backends.

    Returns ``(width, height, bpp, raw)`` where ``raw`` is the inflated
    scanline stream (one filter byte per row, then ``width * bpp`` bytes).
    Every existing validation is preserved: signature/type, chunk bounds and
    CRC, 8-bit RGB/RGBA non-interlaced encoding only, dimension cap,
    IHDR/IDAT/IEND presence, bounded decompress (``expected + 1``), exact
    decompressed size, zlib EOF and no trailing compressed data. In addition
    every per-row filter byte must be in 0..4 — this runs BEFORE the Pillow
    backend hands anything to the image library.
    """
    if not isinstance(data, (bytes, bytearray)) or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("bad PNG signature")
    pos = 8
    width = height = None
    bpp = None
    idat = b""
    saw_iend = False
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        if len(payload) != length or pos + 12 + length > len(data):
            raise ValueError("truncated PNG chunk")
        (crc,) = struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])
        if zlib.crc32(tag + payload) & 0xFFFFFFFF != crc:
            raise ValueError(f"bad CRC in {tag!r}")
        if tag == b"IHDR":
            width, height, depth, color, comp, filt, inter = struct.unpack(">IIBBBBB", payload)
            if (depth, comp, filt, inter) != (8, 0, 0, 0):
                raise ValueError("unsupported PNG encoding")
            if not (1 <= width <= PNG_MAX_DIM and 1 <= height <= PNG_MAX_DIM):
                raise ValueError(f"PNG dimensions out of bounds: {width}x{height}")
            if color == 2:
                bpp = 3
            elif color == 6:
                bpp = 4
            else:
                raise ValueError(f"unsupported PNG colour type {color}")
        elif tag == b"IDAT":
            idat += payload
        elif tag == b"IEND":
            saw_iend = True
            break
        pos += 12 + length
    if width is None or height is None or not idat or bpp is None:
        raise ValueError("missing IHDR/IDAT")
    if not saw_iend:
        raise ValueError("missing IEND (truncated PNG)")
    stride = width * bpp
    expected = height * (stride + 1)
    decomp = zlib.decompressobj()
    try:
        raw = decomp.decompress(idat, expected + 1)
    except zlib.error as exc:
        raise ValueError(f"corrupt PNG image data: {exc}") from exc
    if decomp.unconsumed_tail or len(raw) != expected:
        raise ValueError(
            f"decompressed size mismatch: got {len(raw)}, want {expected} (bounded)"
        )
    if not decomp.eof:
        # A truncated deflate stream can still yield exactly `expected` output
        # bytes; without an EOF check it would be silently accepted.
        raise ValueError("truncated PNG image data: zlib stream never reaches EOF")
    if decomp.unused_data:
        raise ValueError("trailing compressed data after zlib stream end")
    for y in range(height):
        ftype = raw[y * (stride + 1)]
        if ftype > 4:
            raise ValueError(f"unknown PNG filter type {ftype}")
    return width, height, bpp, raw


def decode_png_bytes(data: bytes, *, backend: str = "stdlib") -> np.ndarray:
    """Decode an 8-bit RGB/RGBA non-interlaced PNG to an HxWx3 uint8 array.

    Raises ValueError on any structural mismatch. Pure stdlib + numpy by
    default; ``backend="pillow"`` additionally requires the OPTIONAL Pillow
    dependency (imported only when explicitly chosen) and reuses the same
    strict validation before letting Pillow inflate again — the Python
    per-pixel reconstruction loop is skipped. Unknown backends are rejected
    before any decoding; there is never a silent fallback.
    """
    if backend == "stdlib":
        pass
    elif backend == "pillow":
        pass
    else:
        raise ValueError(f"unknown PNG decode backend {backend!r}")
    width, height, bpp, raw = _validated_png_scanlines(data)
    if backend == "pillow":
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "PNG backend 'pillow' requested but Pillow is not installed; "
                "install the 'vision' extra (embodied-trackpad[vision]) or use "
                "backend='stdlib'"
            ) from exc
        try:
            with closing(Image.open(BytesIO(bytes(data)))) as im:
                with closing(im.convert("RGB")) as rgb_im:
                    arr = np.array(rgb_im, dtype=np.uint8)
        except Exception as exc:
            raise ValueError(f"Pillow failed to decode validated PNG: {exc}") from exc
        if arr.shape != (height, width, 3):
            raise ValueError(
                f"Pillow decoded shape {arr.shape}, want {(height, width, 3)}"
            )
        return arr
    stride = width * bpp
    img = np.zeros((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    for y in range(height):
        line = raw[y * (stride + 1) : (y + 1) * (stride + 1)]
        ftype = line[0]
        cur = np.frombuffer(line[1:], dtype=np.uint8).copy()
        if ftype == 0:
            pass
        elif ftype == 1:  # Sub: cumsum within each byte lane
            cur = (cur.reshape(width, bpp).cumsum(axis=0) & 0xFF).reshape(stride).astype(np.uint8)
        elif ftype == 2:  # Up
            cur = ((cur.astype(np.uint16) + prev.astype(np.uint16)) & 0xFF).astype(np.uint8)
        elif ftype == 3:  # Average (sequential; left=0 for the first bpp bytes)
            b = bytearray(cur)
            p = bytes(prev)
            for i in range(stride):
                left = b[i - bpp] if i >= bpp else 0
                b[i] = (b[i] + ((left + p[i]) >> 1)) & 0xFF
            cur = np.frombuffer(bytes(b), dtype=np.uint8).copy()
        elif ftype == 4:  # Paeth (sequential; left/upper-left=0 for first bpp bytes)
            b = bytearray(cur)
            p = bytes(prev)
            for i in range(stride):
                a = b[i - bpp] if i >= bpp else 0
                c = p[i - bpp] if i >= bpp else 0
                b[i] = (b[i] + _paeth(a, p[i], c)) & 0xFF
            cur = np.frombuffer(bytes(b), dtype=np.uint8).copy()
        else:
            raise ValueError(f"unknown PNG filter type {ftype}")
        img[y] = cur
        prev = cur
    rgb = img.reshape(height, width, bpp)[:, :, :3]
    return np.ascontiguousarray(rgb)


# ---------------------------------------------------------------------------
# Colour-based cursor/target extraction from decoded pixels.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtractorConfig:
    # systemRed (255,59,48) cursor ring+dot; strict so the pastel scroll rows
    # and the orange/purple/green task decorations cannot pass.
    cursor_min_pixels: int = 24
    cursor_ref_pixels: int = 120  # ~2 px ring + filled dot at the default cursor size
    cursor_max_bbox_px: int = 32
    # systemBlue (0,122,255) click target (default 90x60 rect or the explicit
    # near-target 20x20 square); green flash after a hit removes the blue and
    # must read as target-lost.
    target_min_pixels: int = 100
    target_ref_pixels: int = 400
    target_max_bbox_px: int = 140


@dataclass
class PixelEstimate:
    found: bool
    x_logical: float | None  # app logical px, +x right / +y up (device coords)
    y_logical: float | None
    col: float | None  # image pixel column (top-left origin)
    row: float | None  # image pixel row (top-left origin)
    confidence: float
    n_pixels: int
    bbox: tuple[int, int, int, int] | None


@dataclass
class VisualEstimate:
    cursor: PixelEstimate
    target: PixelEstimate
    image_shape: tuple[int, int]
    limitations: tuple[str, ...] = EXTRACTION_LIMITATIONS

    @property
    def error_px(self) -> tuple[float, float] | None:
        """(target - cursor) in logical px; None when either side is lost."""
        if not (self.cursor.found and self.target.found):
            return None
        return (
            self.target.x_logical - self.cursor.x_logical,
            self.target.y_logical - self.cursor.y_logical,
        )


def _colour_mask(rgb: np.ndarray, kind: str) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    if kind == "cursor_red":
        return (r >= 200) & (g <= 120) & (b <= 120) & (r - g >= 110) & (r - b >= 110)
    if kind == "target_blue":
        return (b >= 220) & (r <= 100) & (g >= 60) & (g <= 190) & (b - r >= 130) & (b - g >= 50)
    raise ValueError(kind)


def _estimate_from_mask(mask: np.ndarray, kind: str, cfg: ExtractorConfig) -> PixelEstimate:
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    n = int(xs.size)
    if kind == "cursor_red":
        min_px, ref_px, max_bbox = cfg.cursor_min_pixels, cfg.cursor_ref_pixels, cfg.cursor_max_bbox_px
    else:
        min_px, ref_px, max_bbox = cfg.target_min_pixels, cfg.target_ref_pixels, cfg.target_max_bbox_px
    if n < min_px:
        return PixelEstimate(False, None, None, None, None, 0.0, n, None)
    c0, c1 = int(xs.min()), int(xs.max())
    r0, r1 = int(ys.min()), int(ys.max())
    if (c1 - c0 + 1) > max_bbox or (r1 - r0 + 1) > max_bbox:
        # Spread-out matches mean the colour is not one localised marker.
        return PixelEstimate(False, None, None, None, None, 0.0, n, (c0, r0, c1, r1))
    col = float(xs.mean())
    row = float(ys.mean())
    confidence = min(1.0, n / float(ref_px))
    return PixelEstimate(
        True,
        col,  # logical x == pixel col
        float(h) - row,  # logical y == height - pixel row (y-up view)
        col,
        row,
        confidence,
        n,
        (c0, r0, c1, r1),
    )


def extract_visual_estimate(rgb: np.ndarray, cfg: ExtractorConfig | None = None) -> VisualEstimate:
    """Locate the systemRed cursor and systemBlue target in a decoded capture."""
    cfg = cfg or ExtractorConfig()
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 array, got {rgb.shape}")
    h, w = rgb.shape[:2]
    return VisualEstimate(
        cursor=_estimate_from_mask(_colour_mask(rgb, "cursor_red"), "cursor_red", cfg),
        target=_estimate_from_mask(_colour_mask(rgb, "target_blue"), "target_blue", cfg),
        image_shape=(w, h),
    )


# ---------------------------------------------------------------------------
# Clock reconciliation and frame gating. The app's capture clock
# (CACurrentMediaTime) and Python's time.monotonic are NEVER assumed to share
# a basis: the offset is measured once with a round-trip uncertainty, and
# future-dated frames are rejected like over-age ones.
# ---------------------------------------------------------------------------

@dataclass
class ClockReconciler:
    """Measured offset: app_capture_clock = time.monotonic() + offset_s.

    ``uncertainty_s`` is the measured round-trip half-width of the
    calibration sample; consumers must treat frame ages as known only to
    within +/- this value (see :class:`FrameGate`).
    """

    offset_s: float
    uncertainty_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.offset_s):
            raise ValueError(f"clock offset not finite: {self.offset_s!r}")
        if not math.isfinite(self.uncertainty_s) or self.uncertainty_s < 0.0:
            raise ValueError(f"clock uncertainty invalid: {self.uncertainty_s!r}")

    @classmethod
    def sample(cls, capture_callable) -> "ClockReconciler":
        """Measure the offset with one discarded capture round trip.

        The calibration timestamp is validated: a missing or non-finite
        ``monotonic_time_s`` in the probe raises ValueError instead of
        silently poisoning every later frame age.
        """
        py0 = time.monotonic()
        probe = capture_callable()
        py1 = time.monotonic()
        ts = probe.get("monotonic_time_s") if isinstance(probe, dict) else None
        if not isinstance(ts, (int, float)) or isinstance(ts, bool) or not math.isfinite(ts):
            raise ValueError(f"calibration capture timestamp invalid: {ts!r}")
        mid = 0.5 * (py0 + py1)
        return cls(
            offset_s=float(ts) - mid,
            uncertainty_s=0.5 * (py1 - py0),
        )

    @property
    def uncertainty_ms(self) -> float:
        """Round-trip clock uncertainty in milliseconds (always >= 0)."""
        return self.uncertainty_s * 1000.0

    def frame_age_ms(self, frame: dict, now_monotonic_s: float | None = None) -> float:
        now = time.monotonic() if now_monotonic_s is None else now_monotonic_s
        app_now = now + self.offset_s
        return (app_now - float(frame["monotonic_time_s"])) * 1000.0


@dataclass(frozen=True)
class FrameGateConfig:
    max_frame_age_ms: float = CONTACT_PROFILE["max_frame_age_ms"]
    max_future_ms: float = CONTACT_PROFILE["max_future_frame_ms"]
    min_confidence: float = CONTACT_PROFILE["min_colour_confidence"]
    expect_width: int = CAPTURE_WIDTH
    expect_height: int = CAPTURE_HEIGHT
    # ClockReconciler.uncertainty_ms; frame-age acceptance below uses the
    # WORST-CASE age within +/- this bound, so the measured clock uncertainty
    # is never silently ignored. 0.0 keeps the point-estimate behaviour.
    clock_uncertainty_ms: float = 0.0


@dataclass
class GateDecision:
    ok: bool
    reasons: list[str]
    frame_age_ms: float | None


class FrameGate:
    """Accepts only fresh, strictly newer, non-future frames with usable estimates."""

    def __init__(self, cfg: FrameGateConfig | None = None):
        self.cfg = cfg or FrameGateConfig()
        self.last_frame_id = 0
        self.last_capture_clock_s = -math.inf

    def check(self, frame: dict, est: VisualEstimate, frame_age_ms: float) -> GateDecision:
        reasons: list[str] = []
        unc_ms = self.cfg.clock_uncertainty_ms
        if not math.isfinite(unc_ms) or unc_ms < 0.0:
            reasons.append(f"invalid clock uncertainty config {unc_ms!r}")
            unc_ms = 0.0
        img_w, img_h = est.image_shape
        if frame.get("width") != self.cfg.expect_width or frame.get("height") != self.cfg.expect_height:
            reasons.append(
                f"unexpected capture shape {frame.get('width')}x{frame.get('height')}"
            )
        if frame.get("width") != img_w or frame.get("height") != img_h:
            reasons.append(
                f"capture metadata {frame.get('width')}x{frame.get('height')} "
                f"!= decoded image {img_w}x{img_h}"
            )
        fid = frame.get("frame_id")
        if not isinstance(fid, int) or isinstance(fid, bool) or fid <= self.last_frame_id:
            reasons.append(f"stale frame_id {fid!r} (last accepted {self.last_frame_id})")
        mono = frame.get("monotonic_time_s")
        if not isinstance(mono, (int, float)) or isinstance(mono, bool) or not math.isfinite(mono):
            reasons.append("missing capture monotonic clock")
        elif mono <= self.last_capture_clock_s:
            reasons.append(f"non-increasing capture clock {mono}")
        if not math.isfinite(frame_age_ms):
            reasons.append("frame age not finite")
        else:
            # Worst case within +/- uncertainty: the frame may be OLDER than
            # the point estimate, or further in the FUTURE than it.
            worst_age_ms = frame_age_ms + unc_ms
            worst_future_ms = frame_age_ms - unc_ms
            if worst_age_ms > self.cfg.max_frame_age_ms:
                reasons.append(
                    f"frame age {frame_age_ms:.1f}+{unc_ms:.1f} ms uncertainty "
                    f"exceeds {self.cfg.max_frame_age_ms:.1f} ms"
                )
            if worst_future_ms < -self.cfg.max_future_ms:
                reasons.append(
                    f"frame timestamp {(-frame_age_ms):.1f}+{unc_ms:.1f} ms uncertainty "
                    f"in the FUTURE"
                )
        for name, e in (("cursor", est.cursor), ("target", est.target)):
            if not e.found:
                reasons.append(f"{name} not located in pixels")
                continue
            if not math.isfinite(e.confidence):
                reasons.append(f"{name} confidence not finite: {e.confidence!r}")
            elif e.confidence < self.cfg.min_confidence:
                reasons.append(f"{name} confidence {e.confidence:.2f} < {self.cfg.min_confidence:.2f}")
            coords = (e.x_logical, e.y_logical, e.col, e.row)
            if any(c is None or not math.isfinite(c) for c in coords):
                reasons.append(f"{name} pixel estimate not finite: {coords!r}")
        if not reasons:
            self.last_frame_id = fid
            self.last_capture_clock_s = float(mono)
        return GateDecision(not reasons, reasons, frame_age_ms)
