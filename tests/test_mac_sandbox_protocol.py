"""Real-app protocol tests for TrackpadSandbox (post-review semantics).

These launch the actual signed sandboxed app (built by
macos/TrackpadSandbox/build.sh into .local/build/) and exercise the JSONL
wire over its stdin/stdout. No mocks of the app: rejection paths, gesture
semantics, watchdog behavior, and capture shape run against the live GUI.
"""
import json
import os
import stat
import struct
import subprocess
import sys
import time
import zlib

import pytest

from tools.mac_sandbox_client import (
    APP_BINARY,
    NEAR_PROFILE_ID,
    SandboxClient,
    SandboxError,
    _slide_to,
    near_target_rect,
)

# These are real AppKit integration tests: they compile and launch the signed
# native app, so they can only run on macOS. Repository CI (Ubuntu) collects
# but skips this module; on macOS the real-app tests are mandatory and must
# never be mocked or skipped for a missing binary (the fixture builds it).
if sys.platform != "darwin":
    pytest.skip("real AppKit tests require macOS", allow_module_level=True)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(REPO_ROOT, "macos", "TrackpadSandbox", "build.sh")


@pytest.fixture(scope="module", autouse=True)
def built_app():
    if not os.path.exists(APP_BINARY):
        subprocess.run(["bash", BUILD], check=True, capture_output=True)


@pytest.fixture()
def client():
    c = SandboxClient(evaluator=True)  # tests explicitly hold the evaluator role
    try:
        yield c
    finally:
        c._teardown()


def contact(fid="a", x=0.0, y=0.0, force=0.2):
    return {"finger_id": fid, "x_m": x, "y_m": y, "normal_force_n": force}


def touch_msg(seq, t, contacts):
    return {"cmd": "touch", "schema_version": "touch.v1", "sequence": seq,
            "simulation_time_s": t, "contacts": contacts}


# ---- wire validation ---------------------------------------------------------
def test_malformed_json_rejected(client):
    r = client.raw_line("this is not json")
    assert r["type"] == "error" and r["code"] == "malformed"


def test_unknown_cmd_rejected(client):
    r = client.raw({"cmd": "teleport", "x": 1})
    assert r["type"] == "error" and r["code"] == "unknown_cmd"


def test_unknown_touch_field_rejected(client):
    msg = touch_msg(1, 0.016, [])
    msg["target_window"] = "Finder"  # input cannot name windows/apps/coords
    r = client.raw(msg)
    assert r["type"] == "error" and r["code"] == "unknown_field"


def test_unknown_contact_field_rejected(client):
    r = client.raw(touch_msg(1, 0.016, [dict(contact(), pressure=0.9)]))
    assert r["type"] == "error" and "unknown field" in r["detail"]


def test_wrong_schema_rejected(client):
    msg = touch_msg(1, 0.016, [])
    msg["schema_version"] = "touch.v9"
    r = client.raw(msg)
    assert r["type"] == "error" and r["code"] == "touch_rejected"


def test_nonfinite_rejected(client):
    client.proc.stdin.write(
        b'{"cmd":"touch","schema_version":"touch.v1","sequence":1,'
        b'"simulation_time_s":NaN,"contacts":[]}\n')
    client.proc.stdin.flush()
    r = client._read()
    assert r["type"] == "error"


def test_boolean_sequence_rejected(client):
    msg = touch_msg(1, 0.016, [])
    msg["sequence"] = True  # NSNumber must not bridge this to 1
    r = client.raw(msg)
    assert r["type"] == "error" and "sequence" in r["detail"]


def test_boolean_time_rejected(client):
    msg = touch_msg(1, 0.016, [])
    msg["simulation_time_s"] = True
    r = client.raw(msg)
    assert r["type"] == "error"


def test_boolean_force_rejected(client):
    r = client.raw(touch_msg(1, 0.016, [dict(contact(), normal_force_n=True)]))
    assert r["type"] == "error"


def test_real_sequence_still_accepted(client):
    r = client.raw(touch_msg(1, 0.016, [dict(contact(), normal_force_n=1)]))
    assert r["type"] == "touch_ok"  # int-typed JSON numbers remain valid


def test_replayed_sequence_rejected(client):
    client.touch([])
    r = client.raw(touch_msg(1, 0.032, []))
    assert r["type"] == "error" and "sequence" in r["detail"]


def test_out_of_order_sequence_rejected(client):
    r = client.raw(touch_msg(5, 0.016, []))
    assert r["type"] == "error" and "sequence" in r["detail"]


def test_nonmonotonic_time_rejected(client):
    client.touch([], dt=0.1)
    r = client.raw(touch_msg(2, 0.05, []))
    assert r["type"] == "error" and "time" in r["detail"]


def test_too_many_contacts_rejected(client):
    contacts = [contact(fid=f"f{i}") for i in range(11)]
    r = client.raw(touch_msg(1, 0.016, contacts))
    assert r["type"] == "error" and "too many" in r["detail"]


def test_duplicate_finger_id_rejected(client):
    r = client.raw(touch_msg(1, 0.016, [contact(fid="a"), contact(fid="a", x=0.01)]))
    assert r["type"] == "error" and "duplicate" in r["detail"]


def test_empty_finger_id_rejected(client):
    r = client.raw(touch_msg(1, 0.016, [contact(fid="")]))
    assert r["type"] == "error" and "empty" in r["detail"]


def test_out_of_pad_range_rejected(client):
    r = client.raw(touch_msg(1, 0.016, [contact(x=5.0)]))
    assert r["type"] == "error" and "range" in r["detail"]


def test_half_extent_enforced(client):
    # x/y are pad-centered on the shared 160 x 110 mm pad contract:
    # |x| <= 0.08 and |y| <= 0.055.
    r = client.raw(touch_msg(1, 0.016, [contact(x=0.09)]))
    assert r["type"] == "error" and "range" in r["detail"]
    r = client.raw(touch_msg(1, 0.016, [contact(x=0.079)]))
    assert r["type"] == "touch_ok"
    # y boundary on the 110 mm dimension: |y| <= 0.055 (fresh fingers avoid
    # cross-frame delta limits).
    r = client.raw(touch_msg(2, 0.032, [contact(fid="b", y=0.056)]))
    assert r["type"] == "error" and "range" in r["detail"]
    r = client.raw(touch_msg(2, 0.032, [contact(fid="b", y=0.055)]))
    assert r["type"] == "touch_ok"
    r = client.raw(touch_msg(3, 0.048, [contact(fid="c", y=-0.055)]))
    assert r["type"] == "touch_ok"
    r = client.raw(touch_msg(4, 0.064, [contact(fid="d", y=-0.056)]))
    assert r["type"] == "error" and "range" in r["detail"]


def test_excessive_delta_rejected_and_not_consumed(client):
    client.touch([("a", 0.0, 0.0, 0.2)])
    r = client.raw(touch_msg(2, 0.032, [contact(x=0.049)]))
    assert r["type"] == "touch_ok"
    r = client.raw(touch_msg(3, 0.048, [contact(x=-0.049)]))  # 0.098 m jump
    assert r["type"] == "error" and "excessive" in r["detail"]
    r = client.raw(touch_msg(3, 0.048, [contact(x=0.049)]))  # seq 3 still open
    assert r["type"] == "touch_ok"


# ---- gesture semantics ---------------------------------------------------------
def test_first_touch_does_not_teleport(client):
    client.reset(seed=3)
    before = client.state()["cursor"]
    client.touch([("a", 0.07, 0.05, 0.2)])
    assert client.state()["cursor"] == before


def test_returning_finger_does_not_jump(client):
    # a+b down, a lifts (b stays), a returns 6 mm away: no origin survives, so
    # the cursor must not move (was +24 px pre-fix).
    client.reset(seed=1)
    before = client.state()["cursor"]
    client.touch([("a", 0.0, 0.0, 0.2), ("b", 0.02, 0.0, 0.2)])
    client.touch([("a", 0.001, 0.0, 0.2), ("b", 0.02, 0.0, 0.2)])
    client.touch([("b", 0.02, 0.0, 0.2)])  # a omitted; b stays put
    client.touch([("b", 0.02, 0.0, 0.2), ("a", 0.007, 0.0, 0.2)])  # a returns +6 mm
    after = client.state()["cursor"]
    assert after["x"] == before["x"] and after["y"] == before["y"]


def test_cursor_clamped_inside_content(client):
    client.reset(seed=1)
    for i in range(1, 60):
        client.touch([("a", min(i * 0.02, 0.079), 0.0, 0.2)])
    st = client.state()["cursor"]
    assert st["x"] <= client.content_size[0] and st["y"] <= client.content_size[1]


def test_lift_releases_as_click_and_counts_release(client):
    client.touch([("a", 0.0, 0.0, 1.5)])
    client.touch([("a", 0.0, 0.0, 0.1)])
    st = client.state()
    assert st["button_down"] is False and st["release_count"] == 1


def test_press_outside_then_release_inside_target_is_no_hit(client):
    client.reset(seed=1)
    st0 = client.state()
    tgt = st0["click_target"]
    cx, cy = tgt["x"] + tgt["w"] / 2, tgt["y"] + tgt["h"] / 2
    # Park the cursor just outside the target, then press (outside) and make
    # a short held slide into it - within pad bounds.
    px, py = cx, max(40.0, cy - 120.0)
    _slide_to(client, px, py)
    client.touch([("a", 0.0, 0.0, 1.5)])          # press OUTSIDE target
    n = 20
    dx = (cx - px) / 4000.0
    dy = (cy - py) / 4000.0
    for i in range(1, n + 1):                     # slide into target held
        client.touch([("a", dx * i / n, dy * i / n, 1.5)])
    client.touch([("a", dx, dy, 0.1)])            # release inside target
    st = client.state()
    assert st["release_count"] == st0["release_count"] + 1
    assert st["target_hits"] == st0["target_hits"]  # no target activation


def test_press_and_release_inside_target_hits(client):
    client.reset(seed=1)
    st0 = client.state()
    tgt = st0["click_target"]
    cx, cy = tgt["x"] + tgt["w"] / 2, tgt["y"] + tgt["h"] / 2
    _slide_to(client, cx, cy)                    # hover into target
    client.touch([])                              # lift, fresh origin
    client.touch([("a", 0.0, 0.0, 1.5)])          # press INSIDE target
    client.touch([("a", 0.0, 0.0, 0.1)])          # release inside target
    st = client.state()
    assert st["target_hits"] == st0["target_hits"] + 1


def test_state_requires_evaluator_role():
    with SandboxClient() as actor:  # normal actor launch: no evaluator role
        r = actor.raw({"cmd": "state"})
        assert r["type"] == "error" and r["code"] == "evaluator_required"
        actor.quit()
    with SandboxClient(evaluator=True) as ev:
        st = ev.state()
        assert st["type"] == "state" and st["privileged"] is True
        ev.quit()


def test_outside_target_click_counts_release_but_no_hit(client):
    client.reset(seed=5)
    st0 = client.state()
    client.touch([("a", 0.0, 0.0, 1.5)])   # cursor at center: outside target
    client.touch([("a", 0.0, 0.0, 0.1)])
    st = client.state()
    assert st["release_count"] == st0["release_count"] + 1
    assert st["target_hits"] == st0["target_hits"]


def test_hysteresis_no_release_between_thresholds(client):
    client.touch([("a", 0.0, 0.0, 1.5)])
    client.touch([("a", 0.0, 0.0, 0.8)])
    assert client.state()["button_down"] is True


def test_finger_set_change_cancels_press_without_click(client):
    client.reset(seed=1)
    client.touch([("a", 0.0, 0.0, 1.5)])
    assert client.state()["button_down"] is True
    rc = client.state()["release_count"]
    client.touch([("a", 0.001, 0.0, 1.5), ("b", 0.02, 0.0, 0.3)])  # second finger joins
    st = client.state()
    assert st["button_down"] is False
    assert st["release_count"] == rc  # silent cancel, no fake click
    assert st["press_suppressed"] is True


def test_multi_contact_cannot_inherit_one_finger_press(client):
    client.reset(seed=1)
    client.touch([("a", 0.0, 0.0, 1.5)])
    rc = client.state()["release_count"]
    client.touch([("a", 0.001, 0.0, 1.5), ("b", 0.02, 0.0, 0.3), ("c", -0.02, 0.0, 0.3)])
    st = client.state()
    assert st["button_down"] is False and st["release_count"] == rc


def test_expiry_applies_before_lift(client):
    # Press at t~0, then empty contacts at t=2 s: the sim-time gap cancels the
    # press BEFORE the lift is evaluated - no release, no hit, no drag success.
    client.reset(seed=1)
    st0 = client.state()
    client.touch([("a", 0.0, 0.0, 1.5)], dt=0.016)
    assert client.state()["button_down"] is True
    client.touch([], dt=1.984)  # empty contacts after a ~2 s sim gap
    st = client.state()
    assert st["button_down"] is False
    assert st["release_count"] == st0["release_count"]
    assert st["target_hits"] == st0["target_hits"]
    assert st["drag_done"] is False
    # A fresh lift afterwards clears suppression; a new press works.
    client.touch([])
    client.touch([("a", 0.0, 0.0, 1.5)])
    assert client.state()["button_down"] is True


def test_idle_expiry_releases_and_suppresses_until_lift(client):
    client.reset(seed=1)
    client.touch([("a", 0.0, 0.0, 1.5)], dt=0.016)
    client.touch([("a", 0.01, 0.0, 1.5)], dt=1.0)  # sim-time gap > 0.5 s
    st = client.state()
    assert st["button_down"] is False and st["press_suppressed"] is True
    client.touch([])  # lift clears suppression
    client.touch([("a", 0.0, 0.0, 1.5)])
    assert client.state()["button_down"] is True


def test_wall_clock_watchdog_cancels_stalled_press(client):
    client.reset(seed=1)
    client.touch([("a", 0.0, 0.0, 1.5)])
    assert client.state()["button_down"] is True
    rc = client.state()["release_count"]
    time.sleep(1.1)  # > 0.8 s wall-clock silence: stalled transport
    st = client.state()
    assert st["button_down"] is False
    assert st["release_count"] == rc  # cancel is not a click
    assert st["press_suppressed"] is True
    client.touch([("a", 0.01, 0.0, 1.5)])  # still suppressed before a lift
    assert client.state()["button_down"] is False
    client.touch([])
    client.touch([("a", 0.0, 0.0, 1.5)])  # fresh gesture after lift works
    assert client.state()["button_down"] is True


def test_stop_clears_all_gesture_state(client):
    client.touch([("a", 0.0, 0.0, 1.5)])
    client.stop()
    st = client.state()
    assert st["button_down"] is False
    assert st["drag_done"] is False


def test_drag_arms_only_on_press_inside_object(client):
    client.reset(seed=2)
    st = client.state()
    src, dst = st["drag_source"], st["drag_destination"]
    obj0 = dict(st["drag_object"])
    # Press outside the object, slide through it while held, release at dst.
    client.touch([("a", 0.0, 0.0, 0.2)])  # origin at center (outside object)
    client.touch([("a", 0.0, 0.0, 1.5)])  # press outside object
    cur = client.state()["cursor"]
    n = 20
    dx = (src["x"] + src["w"] / 2 - cur["x"]) / 4000.0
    dy = (src["y"] + src["h"] / 2 - cur["y"]) / 4000.0
    for i in range(1, n + 1):  # slide through the object while held
        client.touch([("a", dx * i / n, dy * i / n, 1.5)])
    ddx = (dst["x"] + dst["w"] / 2 - src["x"] - src["w"] / 2) / 4000.0
    ddy = (dst["y"] + dst["h"] / 2 - src["y"] - src["h"] / 2) / 4000.0
    for i in range(1, n + 1):  # continue to destination
        client.touch([("a", dx + ddx * i / n, dy + ddy * i / n, 1.5)])
    client.touch([])
    st2 = client.state()
    assert st2["drag_done"] is False
    assert st2["drag_object"] == obj0  # object never moved


def test_drag_completes_by_object_placement(client):
    client.reset(seed=2)
    st = client.state()
    dst, obj0 = st["drag_destination"], st["drag_object"]
    client.touch([("a", 0.0, 0.0, 0.2)])  # hover origin (cursor starts center)
    # move cursor onto the object without pressing
    cur = client.state()["cursor"]
    n = 20
    dx = (obj0["x"] - cur["x"]) / 4000.0
    dy = (obj0["y"] - cur["y"]) / 4000.0
    for i in range(1, n + 1):
        client.touch([("a", dx * i / n, dy * i / n, 0.2)])
    client.touch([])  # lift, reposition at origin, press on the object
    client.touch([("a", 0.0, 0.0, 1.5)])
    ddx = (dst["x"] + dst["w"] / 2 - obj0["x"]) / 4000.0
    ddy = (dst["y"] + dst["h"] / 2 - obj0["y"]) / 4000.0
    for i in range(1, n + 1):
        client.touch([("a", ddx * i / n, ddy * i / n, 1.5)])
    client.touch([])
    st2 = client.state()
    o = st2["drag_object"]
    assert st2["drag_done"] is True
    assert o != obj0  # object actually moved and stayed at its final position
    assert dst["x"] <= o["x"] <= dst["x"] + dst["w"]
    assert dst["y"] <= o["y"] <= dst["y"] + dst["h"]


# ---- channel separation ---------------------------------------------------------
def _png_ihdr(png):
    """Actual PNG IHDR (width, height), decoded independently of the app."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    length, ctype = struct.unpack(">I4s", png[8:16])
    assert ctype == b"IHDR" and length == 13
    return struct.unpack(">II", png[16:24])


def _png_rows(png):
    """Unfilter an 8-bit RGB/RGBA PNG to raw scanline bytes (stdlib only)."""
    w, h = _png_ihdr(png)
    bit_depth, color_type, interlace = png[24], png[25], png[28]
    assert bit_depth == 8 and color_type in (2, 6) and interlace == 0
    ch = 3 if color_type == 2 else 4
    idat = b""
    off = 8
    while off < len(png):
        ln, typ = struct.unpack(">I4s", png[off:off + 8])
        if typ == b"IDAT":
            idat += png[off + 8:off + 8 + ln]
        off += 12 + ln
    raw = zlib.decompress(idat)
    stride = w * ch
    rows, prev, pos = [], bytearray(stride), 0
    for _ in range(h):
        filt = raw[pos]
        pos += 1
        line = bytearray(raw[pos:pos + stride])
        pos += stride
        for i in range(stride):
            a = line[i - ch] if i >= ch else 0
            b = prev[i]
            cc = prev[i - ch] if i >= ch else 0
            if filt == 1:
                line[i] = (line[i] + a) & 0xFF
            elif filt == 2:
                line[i] = (line[i] + b) & 0xFF
            elif filt == 3:
                line[i] = (line[i] + (a + b) // 2) & 0xFF
            elif filt == 4:
                p = a + b - cc
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - cc)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc
                                      else (b if pb <= pc else cc))) & 0xFF
        rows.append(line)
        prev = line
    return w, h, ch, rows


def _cursor_centroid(png):
    """Centroid (col, row) of the red virtual-cursor pixels in the image.

    Only the cursor uses saturated red (systemRed); targets/rows are blue,
    orange, purple, green, or pastel, so a strict threshold isolates it.
    """
    w, h, ch, rows = _png_rows(png)
    cols, rws = [], []
    for y in range(h):
        row = rows[y]
        for x in range(w):
            i = x * ch
            if row[i] > 200 and row[i + 1] < 90 and row[i + 2] < 90:
                cols.append(x)
                rws.append(y)
    assert cols, "no red cursor pixels found in capture"
    return sum(cols) / len(cols), sum(rws) / len(rws)


def test_touch_ack_carries_no_truth(client):
    r = client.touch([("a", 0.0, 0.0, 0.2)])
    assert r["type"] == "touch_ok"
    for key in ("cursor", "button_down", "click_count", "target_hits",
                "drag_done", "scroll_offset_px", "layout"):
        assert key not in r


def test_capture_shape(client):
    r1 = client.capture()
    r2 = client.capture()
    # Declared observation dimensions are the normalized logical size...
    assert r1["width"] == 800 and r1["height"] == 600
    assert r2["width"] == 800 and r2["height"] == 600
    # ...and the ACTUAL PNG IHDR matches them (not Retina backing pixels).
    assert _png_ihdr(r1["png"]) == (800, 600)
    assert _png_ihdr(r2["png"]) == (800, 600)
    assert r2["frame_id"] == r1["frame_id"] + 1
    assert r2["monotonic_time_s"] >= r1["monotonic_time_s"]  # declared clock
    assert r1["png"][:8] == b"\x89PNG\r\n\x1a\n"
    for key in ("cursor", "release_count", "target_hits", "drag_done",
                "scroll_offset_px", "click_target"):
        assert key not in r1


def test_capture_cursor_pixels_track_logical_movement(client):
    # Real bitmap/coordinate behavior: a known logical cursor move shifts the
    # rendered cursor by the same delta in image pixels (1 logical px = 1
    # image px; PNG rows run top-down against the y-up view).
    client.reset(seed=1)
    cap0 = client.capture()
    c0 = _cursor_centroid(cap0["png"])
    st0 = client.state()["cursor"]
    client.touch([("a", 0.0, 0.0, 0.2)])  # origin contact: no teleport
    for i in range(1, 9):                 # light hover, below press force
        client.touch([("a", i * 0.001, i * 0.0005, 0.2)])
    client.touch([])
    st1 = client.state()["cursor"]
    cap1 = client.capture()
    c1 = _cursor_centroid(cap1["png"])
    dx, dy = st1["x"] - st0["x"], st1["y"] - st0["y"]
    assert (dx, dy) == (32.0, 16.0)  # 4000 px/m device gain, 8 frames
    assert abs((c1[0] - c0[0]) - dx) < 2.0
    assert abs((c1[1] - c0[1]) + dy) < 2.0  # row axis is inverted vs view y


def test_state_query_is_explicitly_privileged(client):
    st = client.state()
    assert st["type"] == "state" and st["privileged"] is True


def test_reset_seed_deterministic_layout(client):
    client.reset(seed=42)
    a = client.state()["click_target"]
    client.reset(seed=42)
    assert client.state()["click_target"] == a
    client.reset(seed=43)
    assert client.state()["click_target"] != a


def test_clean_exit_on_eof(client):
    assert client.quit() == 0


# ---- near_pointing_v1 fixed reset profile -----------------------------------
def _blue_stats(png):
    """(count, centroid col, centroid row) of strict systemBlue-mask pixels,
    matching the fixed-threshold extractor thresholds in
    embodied_trackpad.leap_v1.visual_servo (_colour_mask target_blue)."""
    w, h, ch, rows = _png_rows(png)
    n, sx, sy = 0, 0.0, 0.0
    for y in range(h):
        row = rows[y]
        for x in range(w):
            i = x * ch
            r, g, b = row[i], row[i + 1], row[i + 2]
            if b >= 220 and r <= 100 and 60 <= g <= 190 and b - r >= 130 and b - g >= 50:
                n += 1
                sx += x
                sy += y
    return n, (sx / n if n else None), (sy / n if n else None)


def _red_count(png):
    """Strict systemRed-mask pixel count (visual_servo _colour_mask cursor_red)."""
    w, h, ch, rows = _png_rows(png)
    n = 0
    for y in range(h):
        row = rows[y]
        for x in range(w):
            i = x * ch
            r, g, b = row[i], row[i + 1], row[i + 2]
            if r >= 200 and g <= 120 and b <= 120 and r - g >= 110 and r - b >= 110:
                n += 1
    return n


def test_near_profile_deterministic_offsets(client):
    # All 12 published offsets land exactly, reproducibly, by seed % 12.
    for seed in range(12):
        r = client.reset(seed=seed, task_profile=NEAR_PROFILE_ID)
        assert r["profile"] == NEAR_PROFILE_ID
        st = client.state()
        assert st["task_profile"] == NEAR_PROFILE_ID
        assert st["click_target"] == near_target_rect(seed), f"seed {seed}"
    # Wraparound: seed 12 selects the same offset as seed 0.
    client.reset(seed=12, task_profile=NEAR_PROFILE_ID)
    assert client.state()["click_target"] == near_target_rect(0)
    # Same seed twice: identical fixed placement (never drifts between resets).
    client.reset(seed=5, task_profile=NEAR_PROFILE_ID)
    a = client.state()["click_target"]
    client.reset(seed=5, task_profile=NEAR_PROFILE_ID)
    assert client.state()["click_target"] == a


def test_near_profile_does_not_leak_into_default_reset(client):
    client.reset(seed=42)
    default_a = client.state()["click_target"]
    assert (default_a["w"], default_a["h"]) == (90.0, 60.0)  # existing layout
    client.reset(seed=7, task_profile=NEAR_PROFILE_ID)
    assert client.state()["click_target"] == near_target_rect(7)
    # A later plain reset restores the exact default layout and profile.
    client.reset(seed=42)
    st = client.state()
    assert st["click_target"] == default_a
    assert st["task_profile"] == "default"


def test_near_profile_invalid_fails_atomically(client):
    client.reset(seed=7)
    before = client.state()
    for bad in ("teleport_please", "", 123, True, 12.5):
        r = client.raw({"cmd": "reset", "seed": 9, "task_profile": bad})
        assert r["type"] == "error" and r["code"] == "bad_task_profile", r
    after = client.state()
    # Nothing was reset: seed, layout, cursor and counters are untouched.
    assert after["seed"] == before["seed"] == 7
    assert after["click_target"] == before["click_target"]
    assert after["cursor"] == before["cursor"]
    assert after["release_count"] == before["release_count"]


def test_near_profile_actor_role_stays_bounded():
    with SandboxClient() as actor:  # normal actor launch, no evaluator role
        r = actor.reset(seed=3, task_profile=NEAR_PROFILE_ID)
        assert r["type"] == "reset_ok" and r["profile"] == NEAR_PROFILE_ID
        for key in ("cursor", "click_target", "layout", "drag_source"):
            assert key not in r
        r = actor.raw({"cmd": "state"})  # actor still cannot query truth
        assert r["type"] == "error" and r["code"] == "evaluator_required"
        cap = actor.capture()  # capture stays the actual 800x600 own-window PNG
        assert (cap["width"], cap["height"]) == (800, 600)
        assert _png_ihdr(cap["png"]) == (800, 600)
        actor.quit()


def test_near_profile_target_never_follows_cursor(client):
    client.reset(seed=0, task_profile=NEAR_PROFILE_ID)  # offset (+12, 0)
    rect0 = client.state()["click_target"]
    _slide_to(client, 200.0, 450.0)  # hover far away from the target
    assert client.state()["click_target"] == rect0  # fixed, not following
    cap = client.capture()
    n, col, row = _blue_stats(cap["png"])
    # Pixel-derived centre stays at the fixed target centre: image col = x,
    # image row = 600 - logical y (y-up view, top-down PNG rows).
    assert n >= 200
    assert abs(col - (rect0["x"] + rect0["w"] / 2)) < 2.0
    assert abs(row - (600 - (rect0["y"] + rect0["h"] / 2))) < 2.0


def test_near_profile_marker_extractable_under_pressed_cursor(client):
    client.reset(seed=1, task_profile=NEAR_PROFILE_ID)  # offset (+18, 0)
    cap = client.capture()
    n_open, _, _ = _blue_stats(cap["png"])
    assert n_open >= 800  # unobstructed 32x32 marker (rounded corners + AA)
    rect = client.state()["click_target"]
    _slide_to(client, rect["x"] + rect["w"] / 2, rect["y"] + rect["h"] / 2)
    client.touch([("a", 0.0, 0.0, 1.5)])  # pressed red ring over the centre
    pressed = client.capture()
    n_under, col, row = _blue_stats(pressed["png"])
    # Marker stays extractable: >=200 blue pixels survive the pressed ring,
    # and the red cursor itself remains clearly visible.
    assert n_under >= 200, f"only {n_under} blue pixels under pressed ring"
    assert _red_count(pressed["png"]) >= 24
    assert abs(col - (rect["x"] + rect["w"] / 2)) < 2.0
    assert abs(row - (600 - (rect["y"] + rect["h"] / 2))) < 2.0
    client.touch([])


# ---- diagnostics gating ---------------------------------------------------------
def test_probes_unavailable_in_normal_mode(client):
    r = client.raw({"cmd": "probe_file"})
    assert r["type"] == "error" and r["code"] == "diagnostics_disabled"
    r = client.raw({"cmd": "probe_net", "port": 9})
    assert r["type"] == "error" and r["code"] == "diagnostics_disabled"


def test_probe_port_validated(tmp_path):
    with SandboxClient(diagnostics=True, canary_root=str(tmp_path)) as c:
        r = c.raw({"cmd": "probe_net", "port": 70000})  # no UInt16 trap
        assert r["type"] == "error" and r["code"] == "bad_probe"
        r = c.raw({"cmd": "probe_net", "port": True})
        assert r["type"] == "error" and r["code"] == "bad_probe"
        with pytest.raises(SandboxError):
            c.probe_net(0)
        c.quit()


def test_probe_file_fixed_canary_only(tmp_path):
    with SandboxClient(diagnostics=True, canary_root=str(tmp_path)) as c:
        r = c.probe_file()
        assert r["type"] == "probe_file"
        assert r["path"].endswith("sandbox-canary.txt")
        # Arbitrary paths cannot be supplied: probe_file takes no path field.
        r2 = c.raw({"cmd": "probe_file", "path": "/tmp/evil"})
        assert r2["path"].endswith("sandbox-canary.txt")
        c.quit()


# ---- bounded client reads / teardown (fake unresponsive apps) ------------------
def _fake_app(tmp_path, body):
    p = tmp_path / "fake.sh"
    p.write_text(f"#!/bin/sh\n{body}\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(p)


def test_startup_timeout_bounded(tmp_path):
    fake = _fake_app(tmp_path, "sleep 30")
    t0 = time.monotonic()
    with pytest.raises(SandboxError):
        SandboxClient(binary=fake, startup_timeout=2.0)
    assert time.monotonic() - t0 < 10


def test_partial_output_bounded(tmp_path, monkeypatch):
    fake = _fake_app(tmp_path, "printf '{\"type\":\"hel'\nsleep 30")
    procs = []
    real_popen = subprocess.Popen
    monkeypatch.setattr(subprocess, "Popen",
                        lambda *a, **k: procs.append(real_popen(*a, **k)) or procs[-1])
    with pytest.raises(SandboxError):
        SandboxClient(binary=fake, startup_timeout=2.0)
    assert all(p.poll() is not None for p in procs)  # no zombie child


def test_read_timeout_mid_session(client):
    # The app answers every command; asking for an extra response must time
    # out boundedly rather than block forever.
    t0 = time.monotonic()
    with pytest.raises(SandboxError):
        client._read(timeout=1.0)
    assert time.monotonic() - t0 < 5
