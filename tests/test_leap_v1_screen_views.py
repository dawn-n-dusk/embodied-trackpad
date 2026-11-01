"""Pure NumPy software tests for the standalone screen-view preparation helper.

Covers ONLY the contract of
:mod:`embodied_trackpad.leap_v1.screen_views`: input validation, exact
native-pixel 1:1 crop mapping (horizontal/vertical and all four edges with
zero padding and an explicit unshifted origin), cursor-only gating (the target
estimate never gates the crop), rejection of missing/spread-out/low-confidence
cursors, and output-array independence. All images are synthesized in code
(deterministic patterned background plus painted systemRed cursor /
systemBlue target blobs); they are unit-test fixtures, not training data. No
GUI capture, oracle/target coordinates, model slots or pi0.5 wiring are
exercised.
"""
from __future__ import annotations

import pytest

# Repository CI installs only .[dev] (pytest/jsonschema); keep collection
# working there. The pinned local runtime has numpy and runs every test.
np = pytest.importorskip("numpy", reason="numpy not installed")

from embodied_trackpad.leap_v1.screen_views import (
    EXPECTED_SHAPE,
    LOCAL_SIZE,
    MIN_CURSOR_CONFIDENCE,
    ScreenViews,
    prepare_screen_views,
)

RED = (255, 59, 48)  # app systemRed cursor
BLUE = (0, 122, 255)  # app systemBlue target


def background_image() -> np.ndarray:
    """Deterministic patterned 800x600x3 background.

    Channel values stay far below both colour-mask thresholds (r < 200,
    b < 220), so the fixed-palette extractor sees no spurious cursor/target
    pixels; the pattern still makes any non-1:1 pixel mapping visible.
    """
    rows = np.arange(600, dtype=np.uint16)[:, None]
    cols = np.arange(800, dtype=np.uint16)[None, :]
    img = np.empty((600, 800, 3), dtype=np.uint8)
    img[:, :, 0] = (40 + (cols % 16)).astype(np.uint8)
    img[:, :, 1] = (60 + (rows % 16)).astype(np.uint8)
    img[:, :, 2] = (80 + ((cols + rows) % 16)).astype(np.uint8)
    return img


def paint_cursor(img: np.ndarray, row0: int, col0: int, h: int = 12, w: int = 12):
    """Paint a compact systemRed cursor blob whose centre is (col0+w/2, row0+h/2)."""
    img[row0 : row0 + h, col0 : col0 + w] = RED
    return img


def cursor_image(row0: int, col0: int, h: int = 12, w: int = 12) -> np.ndarray:
    return paint_cursor(background_image(), row0, col0, h=h, w=w)


def expected_origin(col0: int, row0: int, w: int = 12, h: int = 12):
    # blob covers cols [col0, col0+w): mean col = col0 + (w-1)/2, so
    # floor(mean) = col0 + (w-1)//2; origin uses floor(col/row) - 112
    return (col0 + (w - 1) // 2 - 112, row0 + (h - 1) // 2 - 112)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rejects_non_ndarray_input():
    with pytest.raises(TypeError):
        prepare_screen_views([[0] * 800 * 600])
    with pytest.raises(TypeError):
        prepare_screen_views(np.zeros((600, 800, 3), dtype=np.uint8).tolist())


@pytest.mark.parametrize(
    "shape",
    [
        (600, 800),  # 2-D grayscale
        (600, 800, 4),  # RGBA
        (600, 800, 1),
        (224, 224, 3),  # already-resized frame is not a raw capture
        (600, 799, 3),
        (599, 800, 3),
    ],
)
def test_rejects_bad_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        prepare_screen_views(np.zeros(shape, dtype=np.uint8))


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int16, np.bool_])
def test_rejects_bad_dtype(dtype):
    with pytest.raises(ValueError, match="uint8"):
        prepare_screen_views(np.zeros(EXPECTED_SHAPE, dtype=dtype))


# ---------------------------------------------------------------------------
# Exact native-pixel crop mapping
# ---------------------------------------------------------------------------


def test_full_rgb_is_exact_independent_copy():
    img = cursor_image(300, 400)
    before = img.copy()
    views = prepare_screen_views(img)
    np.testing.assert_array_equal(views.full_rgb, before)
    assert views.full_rgb.dtype == np.uint8
    assert views.full_rgb.shape == EXPECTED_SHAPE
    # In-place mutation of the input must not leak into the returned views.
    img[:] = 0
    np.testing.assert_array_equal(views.full_rgb, before)


@pytest.mark.parametrize(
    "col0,row0",
    [
        (200, 300),  # horizontal: mid-left
        (600, 300),  # horizontal: mid-right
        (400, 150),  # vertical: upper
        (400, 450),  # vertical: lower
        (400, 300),  # centre
    ],
)
def test_interior_crop_native_pixel_1to1_mapping(col0, row0):
    img = cursor_image(row0, col0)
    views = prepare_screen_views(img)
    origin_col, origin_row = expected_origin(col0, row0)
    assert views.local_valid is True
    assert views.origin_col_row == (origin_col, origin_row)
    assert views.local_rgb.shape == (LOCAL_SIZE, LOCAL_SIZE, 3)
    assert views.local_rgb.dtype == np.uint8
    assert views.local_pixel_valid.dtype == bool
    # Exact elementwise equality against the correct source window: any
    # resize, interpolation or shifted window would fail this.
    np.testing.assert_array_equal(
        views.local_rgb,
        img[origin_row : origin_row + LOCAL_SIZE, origin_col : origin_col + LOCAL_SIZE],
    )
    assert views.local_pixel_valid.all()
    # centre pixel maps back to floor(cursor) in full-image coordinates
    np.testing.assert_array_equal(
        views.local_rgb[LOCAL_SIZE // 2, LOCAL_SIZE // 2],
        img[row0 + 5, col0 + 5],
    )


def test_cursor_reporting_uses_pixel_col_row_not_logical_y():
    img = cursor_image(300, 400)  # 12x12 blob: cursor centre (col 405.5, row 305.5)
    views = prepare_screen_views(img)
    assert views.cursor_col_row == pytest.approx((405.5, 305.5))
    assert views.cursor_confidence == pytest.approx(1.0)
    assert isinstance(views.cursor_confidence, float)


@pytest.mark.parametrize(
    "col0,row0,origin,src_slice,dst_slice",
    [
        (10, 10, (-97, -97), (slice(0, 127), slice(0, 127)), (slice(97, 224), slice(97, 224))),
        (770, 10, (663, -97), (slice(0, 127), slice(663, 800)), (slice(97, 224), slice(0, 137))),
        (10, 570, (-97, 463), (slice(463, 600), slice(0, 127)), (slice(0, 137), slice(97, 224))),
        (770, 570, (663, 463), (slice(463, 600), slice(663, 800)), (slice(0, 137), slice(0, 137))),
    ],
    ids=["top_left", "top_right", "bottom_left", "bottom_right"],
)
def test_edge_padding_mask_and_unshifted_origin(col0, row0, origin, src_slice, dst_slice):
    img = cursor_image(row0, col0)
    views = prepare_screen_views(img)
    assert views.local_valid is True
    # Origin stays explicit (negative or large); the crop is NEVER shifted.
    assert views.origin_col_row == origin
    # Copied window is exact; padded area is black with mask False.
    np.testing.assert_array_equal(views.local_rgb[dst_slice], img[src_slice])
    assert views.local_pixel_valid[dst_slice].all()
    pad_mask = np.ones((LOCAL_SIZE, LOCAL_SIZE), dtype=bool)
    pad_mask[dst_slice] = False
    assert not views.local_pixel_valid[pad_mask].any()
    assert views.local_rgb[pad_mask].sum() == 0
    # The cursor still maps back to its full-image pixel via the origin.
    fc, fr = origin[0] + LOCAL_SIZE // 2, origin[1] + LOCAL_SIZE // 2
    np.testing.assert_array_equal(
        views.local_rgb[LOCAL_SIZE // 2, LOCAL_SIZE // 2], img[fr, fc]
    )


# ---------------------------------------------------------------------------
# Cursor-only gating: the target estimate never gates the crop
# ---------------------------------------------------------------------------


def test_target_absent_does_not_change_cursor_crop():
    col0, row0 = 300, 400
    origin = expected_origin(col0, row0)
    with_target = cursor_image(row0, col0)
    with_target[40:80, 40:100] = BLUE  # outside the local window
    without_target = cursor_image(row0, col0)
    views_t = prepare_screen_views(with_target)
    views_n = prepare_screen_views(without_target)
    assert views_t.local_valid and views_n.local_valid
    assert views_t.origin_col_row == origin == views_n.origin_col_row
    np.testing.assert_array_equal(views_t.local_rgb, views_n.local_rgb)
    np.testing.assert_array_equal(views_t.local_pixel_valid, views_n.local_pixel_valid)
    # The target is still visible in the full view.
    assert (views_t.full_rgb[40:80, 40:100] == np.array(BLUE, dtype=np.uint8)).all()


def test_target_change_alters_only_copied_pixels_not_origin():
    col0, row0 = 300, 400
    origin = expected_origin(col0, row0)
    img_a = cursor_image(row0, col0)
    img_a[330:350, 230:260] = BLUE  # target A inside the local window
    img_b = cursor_image(row0, col0)
    img_b[450:470, 340:380] = BLUE  # target B inside the local window
    views_a = prepare_screen_views(img_a)
    views_b = prepare_screen_views(img_b)
    assert views_a.origin_col_row == origin == views_b.origin_col_row
    assert views_a.local_valid and views_b.local_valid
    diff = np.any(views_a.local_rgb != views_b.local_rgb, axis=2)
    expected = np.zeros((LOCAL_SIZE, LOCAL_SIZE), dtype=bool)
    expected[330 - origin[1] : 350 - origin[1], 230 - origin[0] : 260 - origin[0]] = True
    expected[450 - origin[1] : 470 - origin[1], 340 - origin[0] : 380 - origin[0]] = True
    np.testing.assert_array_equal(diff, expected)


# ---------------------------------------------------------------------------
# Rejection: missing / spread-out / low-confidence cursor, no stale fallback
# ---------------------------------------------------------------------------


def _assert_invalid(views: ScreenViews, confidence: float):
    assert views.local_valid is False
    assert views.origin_col_row is None
    assert views.cursor_col_row is None
    assert views.cursor_confidence == pytest.approx(confidence)
    assert views.local_rgb.shape == (LOCAL_SIZE, LOCAL_SIZE, 3)
    assert views.local_rgb.dtype == np.uint8
    assert views.local_rgb.sum() == 0  # all-zero local view, no stale fallback
    assert views.local_pixel_valid.dtype == bool
    assert not views.local_pixel_valid.any()
    assert views.full_rgb.shape == EXPECTED_SHAPE  # full view stays available


def test_missing_cursor_rejected():
    views = prepare_screen_views(background_image())
    _assert_invalid(views, confidence=0.0)


def test_spread_out_cursor_rejected():
    img = background_image()
    paint_cursor(img, 20, 20)  # two compact blobs but one huge colour bbox
    paint_cursor(img, 500, 700)
    views = prepare_screen_views(img)
    _assert_invalid(views, confidence=0.0)


def test_low_confidence_cursor_rejected():
    img = background_image()
    paint_cursor(img, 300, 400, h=6, w=4)  # 24 px -> found, 24/120 = 0.2 < 0.5
    views = prepare_screen_views(img)
    _assert_invalid(views, confidence=pytest.approx(24 / 120))
    assert MIN_CURSOR_CONFIDENCE == 0.5


# ---------------------------------------------------------------------------
# Output array independence (no aliasing anywhere)
# ---------------------------------------------------------------------------


def test_returned_arrays_share_no_memory():
    img = cursor_image(300, 400)
    views = prepare_screen_views(img)
    assert not np.shares_memory(views.full_rgb, img)
    assert not np.shares_memory(views.local_rgb, img)
    assert not np.shares_memory(views.local_pixel_valid, img)
    assert not np.shares_memory(views.full_rgb, views.local_rgb)
    assert not np.shares_memory(views.full_rgb, views.local_pixel_valid)
    assert not np.shares_memory(views.local_rgb, views.local_pixel_valid)
    local_before = views.local_rgb.copy()
    views.full_rgb[:] = 0  # mutating one output must not touch the others
    np.testing.assert_array_equal(views.local_rgb, local_before)


def test_invalid_views_also_share_no_memory_with_input():
    img = background_image()
    views = prepare_screen_views(img)
    before = views.full_rgb.copy()
    img[:] = 255
    np.testing.assert_array_equal(views.full_rgb, before)
    assert not np.shares_memory(views.local_rgb, views.full_rgb)
