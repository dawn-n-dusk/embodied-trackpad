"""Focused pure tests for the LEAP v1 visual-servo input utilities.

Covers ONLY what exists in src/embodied_trackpad/leap_v1/visual_servo.py:
std-lib PNG decoding (filters 0..4, RGB/RGBA, CRC/truncation/size rejection),
colour-mask cursor/target extraction, ClockReconciler calibration validation
and uncertainty, and FrameGate freshness/metadata/finiteness checks. No
controller, CLI, episodes, GUI truth or hardware are exercised.
"""
from __future__ import annotations

import math
import struct
import time
import zlib

import pytest

# Repository CI installs only .[dev] (pytest/jsonschema); keep collection
# working there. The pinned local runtime has numpy and runs every test.
np = pytest.importorskip("numpy", reason="numpy not installed")

from embodied_trackpad.leap_v1.visual_servo import (
    ClockReconciler,
    ExtractorConfig,
    FrameGate,
    FrameGateConfig,
    PixelEstimate,
    VisualEstimate,
    decode_png_bytes,
    extract_visual_estimate,
)

# ---------------------------------------------------------------------------
# Minimal independent PNG encoder (test-side reference implementation).
# ---------------------------------------------------------------------------


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def _paeth_ref(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def encode_png(pixels: np.ndarray, filters) -> bytes:
    """Encode an HxWx3 (RGB) or HxWx4 (RGBA) uint8 image, one filter per row."""
    h, w, c = pixels.shape
    assert c in (3, 4)
    color = 2 if c == 3 else 6
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color, 0, 0, 0)
    stride = w * c
    if isinstance(filters, int):
        filters = [filters] * h
    assert len(filters) == h
    prev = [0] * stride
    raw = bytearray()
    for y in range(h):
        cur = [int(v) for v in pixels[y].reshape(-1)]
        f = filters[y]
        out = bytearray(stride)
        for i in range(stride):
            a = cur[i - c] if i >= c else 0
            b = prev[i]
            cc = prev[i - c] if i >= c else 0
            x = cur[i]
            if f == 0:
                out[i] = x
            elif f == 1:
                out[i] = (x - a) & 0xFF
            elif f == 2:
                out[i] = (x - b) & 0xFF
            elif f == 3:
                out[i] = (x - ((a + b) >> 1)) & 0xFF
            elif f == 4:
                out[i] = (x - _paeth_ref(a, b, cc)) & 0xFF
            else:
                raise AssertionError(f)
        raw.append(f)
        raw += out
        prev = cur
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw)))
        + _chunk(b"IEND", b"")
    )


def known_pixels(h: int = 6, w: int = 9, channels: int = 3, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, channels), dtype=np.uint8)


# ---------------------------------------------------------------------------
# PNG decoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filter_type", [0, 1, 2, 3, 4])
def test_png_rgb_known_pixels_each_filter(filter_type):
    px = known_pixels()
    decoded = decode_png_bytes(encode_png(px, filter_type))
    np.testing.assert_array_equal(decoded, px)


def test_png_mixed_filters_per_row():
    px = known_pixels(h=10, w=7)
    decoded = decode_png_bytes(encode_png(px, [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]))
    np.testing.assert_array_equal(decoded, px)


def test_png_filter3_later_rows_first_channels():
    # Rows after the first have no left neighbour for the first pixel: the
    # Average predictor there must use floor(up/2) with left=0.
    px = known_pixels(h=8, w=5)
    decoded = decode_png_bytes(encode_png(px, 3))
    np.testing.assert_array_equal(decoded[1:, 0, :], px[1:, 0, :])
    np.testing.assert_array_equal(decoded, px)


def test_png_rgba_decodes_to_rgb():
    px = known_pixels(h=5, w=6, channels=4, seed=99)
    for f in (0, 1, 2, 3, 4):
        decoded = decode_png_bytes(encode_png(px, f))
        assert decoded.shape == (5, 6, 3)
        np.testing.assert_array_equal(decoded, px[:, :, :3])


def test_png_rejects_bad_signature_and_non_bytes():
    with pytest.raises(ValueError):
        decode_png_bytes(b"not a png at all")
    with pytest.raises(ValueError):
        decode_png_bytes("not bytes")


def test_png_rejects_crc_corruption():
    data = bytearray(encode_png(known_pixels(), 0))
    idx = data.find(b"IDAT")
    assert idx > 0
    data[idx + 6] ^= 0xFF  # flip a payload byte, CRC now wrong
    with pytest.raises(ValueError, match="CRC"):
        decode_png_bytes(bytes(data))


def test_png_rejects_truncation():
    png = encode_png(known_pixels(), 2)
    with pytest.raises(ValueError):
        decode_png_bytes(png[:30])  # cut mid-IDAT
    with pytest.raises(ValueError, match="IEND"):
        decode_png_bytes(png[:-12])  # drop the IEND chunk


def test_png_rejects_decompressed_size_mismatch():
    px = known_pixels(h=4, w=4)
    h, w, _ = px.shape
    # Claim one extra row: decompressed length no longer matches IHDR.
    ihdr = struct.pack(">IIBBBBB", w, h + 1, 8, 2, 0, 0, 0)
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw += px[y].reshape(-1).tobytes()
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw)))
        + _chunk(b"IEND", b"")
    )
    with pytest.raises(ValueError, match="size mismatch"):
        decode_png_bytes(png)


def test_png_rejects_oversized_decompression():
    px = known_pixels(h=4, w=4)
    h, w, _ = px.shape
    ihdr = struct.pack(">IIBBBBB", w, h - 1, 8, 2, 0, 0, 0)  # claim fewer rows
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(
            b"".join(b"\x00" + px[y].reshape(-1).tobytes() for y in range(h))
        ))
        + _chunk(b"IEND", b"")
    )
    with pytest.raises(ValueError, match="size mismatch"):
        decode_png_bytes(png)


def test_png_rejects_truncated_zlib_stream_without_eof():
    # Coordinator counterexample: a 1x1 RGB image whose IDAT is a zlib stream
    # cut 4 bytes short still inflates to exactly the expected 4 output bytes,
    # so size matching alone silently accepted it. The stream never reaches
    # EOF and must be rejected; the intact stream must still decode.
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    full = zlib.compress(b"\x00\x11\x22\x33")  # filter 0, pixel (17, 34, 51)
    good = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", full)
        + _chunk(b"IEND", b"")
    )
    decoded = decode_png_bytes(good)  # valid decoding unchanged
    np.testing.assert_array_equal(decoded, np.array([[[17, 34, 51]]], dtype=np.uint8))
    truncated = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", full[:-4])  # CRC recomputed over the truncated payload
        + _chunk(b"IEND", b"")
    )
    with pytest.raises(ValueError, match="EOF"):
        decode_png_bytes(truncated)


def test_png_rejects_trailing_compressed_data():
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    full = zlib.compress(b"\x00\x11\x22\x33")
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", full + b"junk-after-stream")
        + _chunk(b"IEND", b"")
    )
    with pytest.raises(ValueError, match="trailing"):
        decode_png_bytes(png)


def test_png_rejects_unknown_filter_and_bad_encoding():
    px = known_pixels(h=3, w=3)
    h, w, _ = px.shape
    raw = bytearray()
    for y in range(h):
        raw.append(5 if y == 1 else 0)  # invalid filter on row 1
        raw += px[y].reshape(-1).tobytes()
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw)))
        + _chunk(b"IEND", b"")
    )
    with pytest.raises(ValueError, match="filter"):
        decode_png_bytes(png)
    ihdr16 = struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)  # 16-bit depth
    png16 = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr16)
        + _chunk(b"IDAT", zlib.compress(b"\x00"))
        + _chunk(b"IEND", b"")
    )
    with pytest.raises(ValueError, match="unsupported PNG encoding"):
        decode_png_bytes(png16)


# ---------------------------------------------------------------------------
# Colour extraction
# ---------------------------------------------------------------------------

RED = (255, 59, 48)  # app systemRed cursor
BLUE = (0, 122, 255)  # app systemBlue target


def painted_image(h: int = 100, w: int = 200) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[20:28, 50:62] = RED  # 8x12 = 96 px cursor
    img[60:80, 120:140] = BLUE  # 20x20 = 400 px target
    return img


def test_extract_localizes_cursor_and_target():
    est = extract_visual_estimate(painted_image())
    assert est.image_shape == (200, 100)
    assert est.cursor.found and est.target.found
    assert est.cursor.col == pytest.approx(55.5)
    assert est.cursor.row == pytest.approx(23.5)
    assert est.cursor.x_logical == pytest.approx(55.5)
    assert est.cursor.y_logical == pytest.approx(100 - 23.5)  # y-up logical
    assert est.cursor.confidence == pytest.approx(min(1.0, 96 / 120.0))
    assert est.target.confidence == pytest.approx(1.0)
    assert est.target.x_logical == pytest.approx(129.5)
    assert est.error_px == pytest.approx((129.5 - 55.5, (100 - 69.5) - (100 - 23.5)))


def test_extract_missing_markers():
    est = extract_visual_estimate(np.zeros((60, 80, 3), dtype=np.uint8))
    assert not est.cursor.found and not est.target.found
    assert est.cursor.confidence == 0.0 and est.target.confidence == 0.0
    assert est.error_px is None


def test_extract_ambiguous_spread_colour_rejected():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[10:14, 5:15] = RED
    img[80:84, 170:180] = RED  # two far-apart clusters -> huge bbox
    est = extract_visual_estimate(img)
    assert not est.cursor.found
    assert est.cursor.confidence == 0.0
    assert est.cursor.bbox is not None  # evidence of the spread kept


def test_extract_too_few_pixels_rejected():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[40:42, 90:98] = RED  # 16 px < cursor_min_pixels
    est = extract_visual_estimate(img, ExtractorConfig())
    assert not est.cursor.found


def test_extract_rejects_bad_array_shape():
    with pytest.raises(ValueError):
        extract_visual_estimate(np.zeros((10, 10), dtype=np.uint8))


# ---------------------------------------------------------------------------
# ClockReconciler
# ---------------------------------------------------------------------------


def test_reconciler_constructor_validation():
    ClockReconciler(offset_s=0.5, uncertainty_s=0.0)  # ok
    with pytest.raises(ValueError):
        ClockReconciler(offset_s=float("nan"), uncertainty_s=0.0)
    with pytest.raises(ValueError):
        ClockReconciler(offset_s=float("inf"), uncertainty_s=0.0)
    with pytest.raises(ValueError):
        ClockReconciler(offset_s=0.0, uncertainty_s=-0.001)
    with pytest.raises(ValueError):
        ClockReconciler(offset_s=0.0, uncertainty_s=float("nan"))


def test_reconciler_sample_validates_calibration_timestamp():
    with pytest.raises(ValueError, match="timestamp invalid"):
        ClockReconciler.sample(lambda: {"monotonic_time_s": float("nan")})
    with pytest.raises(ValueError, match="timestamp invalid"):
        ClockReconciler.sample(lambda: {})  # missing key
    with pytest.raises(ValueError, match="timestamp invalid"):
        ClockReconciler.sample(lambda: {"monotonic_time_s": True})
    with pytest.raises(ValueError, match="timestamp invalid"):
        ClockReconciler.sample(lambda: "not a frame dict")


def test_reconciler_sample_measures_offset_and_uncertainty():
    rec = ClockReconciler.sample(lambda: {"monotonic_time_s": time.monotonic() + 5.0})
    assert rec.offset_s == pytest.approx(5.0, abs=0.2)
    assert 0.0 <= rec.uncertainty_s < 0.5  # quantified, never dropped
    assert rec.uncertainty_ms == pytest.approx(rec.uncertainty_s * 1000.0)


def test_reconciler_frame_age_uses_single_reference_clock():
    rec = ClockReconciler(offset_s=2.0, uncertainty_s=0.01)
    frame = {"monotonic_time_s": 10.0}
    # app_now = 10.5 + 2.0 = 12.5 -> age 2.5 s
    assert rec.frame_age_ms(frame, now_monotonic_s=10.5) == pytest.approx(2500.0)


# ---------------------------------------------------------------------------
# FrameGate
# ---------------------------------------------------------------------------


def gate_and_est(w: int = 200, h: int = 100, **cfg_kw):
    cfg = FrameGateConfig(expect_width=w, expect_height=h, **cfg_kw)
    return FrameGate(cfg), extract_visual_estimate(painted_image(h=h, w=w))


def make_frame(fid=1, mono=100.0, w=200, h=100):
    return {"frame_id": fid, "monotonic_time_s": mono, "width": w, "height": h}


def test_gate_accepts_fresh_frame():
    gate, est = gate_and_est()
    d = gate.check(make_frame(), est, frame_age_ms=10.0)
    assert d.ok, d.reasons
    assert d.frame_age_ms == 10.0


def test_gate_rejects_stale_and_non_increasing_frames():
    gate, est = gate_and_est()
    assert gate.check(make_frame(fid=1, mono=100.0), est, 5.0).ok
    d = gate.check(make_frame(fid=1, mono=100.4), est, 5.0)
    assert not d.ok and any("stale frame_id" in r for r in d.reasons)
    d = gate.check(make_frame(fid=2, mono=100.0), est, 5.0)
    assert not d.ok and any("non-increasing" in r for r in d.reasons)
    # A genuinely newer frame still passes afterwards.
    assert gate.check(make_frame(fid=3, mono=100.6), est, 5.0).ok


def test_gate_rejects_future_and_overage_frames():
    gate, est = gate_and_est()
    d = gate.check(make_frame(), est, frame_age_ms=-100.0)  # 100 ms in future
    assert not d.ok and any("FUTURE" in r for r in d.reasons)
    d = gate.check(make_frame(), est, frame_age_ms=300.0)  # > 250 ms default
    assert not d.ok and any("frame age" in r for r in d.reasons)
    d = gate.check(make_frame(), est, frame_age_ms=float("nan"))
    assert not d.ok and any("not finite" in r for r in d.reasons)


def test_gate_rejects_invalid_frame_fields():
    gate, est = gate_and_est()
    d = gate.check(make_frame(mono=float("nan")), est, 5.0)
    assert not d.ok and any("monotonic clock" in r for r in d.reasons)
    d = gate.check(make_frame(mono=True), est, 5.0)
    assert not d.ok and any("monotonic clock" in r for r in d.reasons)
    d = gate.check(make_frame(fid="abc"), est, 5.0)
    assert not d.ok and any("stale frame_id" in r for r in d.reasons)
    d = gate.check(make_frame(fid=True), est, 5.0)
    assert not d.ok and any("stale frame_id" in r for r in d.reasons)


def test_gate_rejects_metadata_mismatch():
    gate, est = gate_and_est()
    # Capture metadata disagrees with the decoded image dimensions.
    d = gate.check(make_frame(w=201), est, 5.0)
    assert not d.ok and any("!= decoded image" in r for r in d.reasons)
    # Image itself is not the expected capture size.
    gate2 = FrameGate(FrameGateConfig(expect_width=640, expect_height=480))
    d = gate2.check(make_frame(), est, 5.0)
    assert not d.ok and any("unexpected capture shape" in r for r in d.reasons)


def test_gate_rejects_missing_and_low_confidence_estimates():
    gate, _ = gate_and_est()
    blank = extract_visual_estimate(np.zeros((100, 200, 3), dtype=np.uint8))
    d = gate.check(make_frame(), blank, 5.0)
    assert not d.ok
    assert any("cursor not located" in r for r in d.reasons)
    assert any("target not located" in r for r in d.reasons)
    img = painted_image()
    img[20:28, 50:62] = 0
    img[20:25, 50:56] = RED  # 30 px -> confidence 0.25 < 0.5
    est = extract_visual_estimate(img)
    d = gate.check(make_frame(), est, 5.0)
    assert not d.ok and any("cursor confidence" in r for r in d.reasons)


def _finite_est_with(cursor=None, target=None) -> VisualEstimate:
    good = PixelEstimate(True, 55.0, 76.0, 55.0, 24.0, 0.9, 96, (50, 20, 61, 27))
    return VisualEstimate(
        cursor=cursor if cursor is not None else good,
        target=target if target is not None else good,
        image_shape=(200, 100),
    )


def test_gate_rejects_nonfinite_confidence_and_pixels():
    gate, _ = gate_and_est()
    # NaN < threshold is False: without an explicit finiteness check this
    # confidence would silently pass.
    est = _finite_est_with(
        cursor=PixelEstimate(True, 55.0, 76.0, 55.0, 24.0, float("nan"), 96, None)
    )
    d = gate.check(make_frame(), est, 5.0)
    assert not d.ok and any("confidence not finite" in r for r in d.reasons)
    est = _finite_est_with(
        target=PixelEstimate(True, float("nan"), 30.0, 129.0, 70.0, 0.9, 400, None)
    )
    d = gate.check(make_frame(), est, 5.0)
    assert not d.ok and any("pixel estimate not finite" in r for r in d.reasons)
    est = _finite_est_with(
        cursor=PixelEstimate(True, 55.0, float("inf"), 55.0, 24.0, 0.9, 96, None)
    )
    d = gate.check(make_frame(), est, 5.0)
    assert not d.ok and any("pixel estimate not finite" in r for r in d.reasons)


def test_gate_age_acceptance_includes_clock_uncertainty():
    # With 30 ms of measured clock uncertainty, a point-estimate age of
    # 230 ms may really be 260 ms old and must NOT be accepted at a 250 ms
    # bound; likewise a slightly-future frame may be further in the future.
    gate, est = gate_and_est(clock_uncertainty_ms=30.0)
    d = gate.check(make_frame(), est, frame_age_ms=230.0)
    assert not d.ok and any("uncertainty" in r for r in d.reasons)
    assert gate.check(make_frame(fid=2, mono=100.5), est, frame_age_ms=200.0).ok
    d = gate.check(make_frame(fid=3, mono=101.0), est, frame_age_ms=-30.0)
    assert not d.ok and any("FUTURE" in r for r in d.reasons)
    assert gate.check(make_frame(fid=4, mono=101.5), est, frame_age_ms=-10.0).ok
    # Zero uncertainty keeps the old point-estimate boundary exactly.
    gate0, est0 = gate_and_est(clock_uncertainty_ms=0.0)
    assert gate0.check(make_frame(), est0, frame_age_ms=230.0).ok


def test_gate_rejects_invalid_uncertainty_config():
    gate, est = gate_and_est(clock_uncertainty_ms=-1.0)
    d = gate.check(make_frame(), est, 5.0)
    assert not d.ok and any("invalid clock uncertainty" in r for r in d.reasons)
    gate, est = gate_and_est(clock_uncertainty_ms=float("nan"))
    d = gate.check(make_frame(), est, 5.0)
    assert not d.ok and any("invalid clock uncertainty" in r for r in d.reasons)


def test_gate_with_reconciler_end_to_end_ages():
    # One reference clock contract: reconciler-measured offset feeds the gate.
    rec = ClockReconciler(offset_s=0.0, uncertainty_s=0.02)
    gate, est = gate_and_est(clock_uncertainty_ms=rec.uncertainty_ms)
    frame = make_frame(mono=200.0)
    age = rec.frame_age_ms(frame, now_monotonic_s=200.01)  # 10 ms old
    assert age == pytest.approx(10.0)
    assert gate.check(frame, est, age).ok
