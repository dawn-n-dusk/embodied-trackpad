"""Device adapter tests: contact -> recognition -> cursor -> tap -> GUI.

These run anywhere (no Isaac). They pin the lawful path: cursor and button
state can ONLY be produced from contact events, never set by the actor.
"""
import math

import pytest

from embodied_trackpad.sharpa_sim.device import (
    RECOGNITION_PROXY, ContactEvent, RelativeCursorTracker,
    SimulatedCapacitiveRecognizer, TrackpadDeviceAdapter, VirtualGUI,
)


def ev(finger="index", x=0.0, y=0.0, f=1.0, t=0.0, pen=0.0):
    return ContactEvent(finger=finger, x=x, y=y, normal_force=f, t=t,
                        penetration=pen)


def test_recognizer_threshold_and_hysteresis():
    r = SimulatedCapacitiveRecognizer(on_threshold=0.15, off_threshold=0.08)
    assert r.update([ev(f=0.10)]) == []          # below on-threshold
    touches = r.update([ev(f=0.20)])
    assert len(touches) == 1                      # recognized
    assert r.update([ev(f=0.10)]) != []           # hysteresis: stays on
    assert r.update([ev(f=0.05)]) == []           # below off-threshold: ends


def test_recognizer_rejects_non_finite():
    r = SimulatedCapacitiveRecognizer()
    bad = ContactEvent("index", float("nan"), 0.0, 5.0, 0.0)
    assert r.update([bad]) == []


def test_recognizer_proxy_label():
    r = SimulatedCapacitiveRecognizer()
    (t,) = r.update([ev(f=1.0)])
    assert t.recognition == RECOGNITION_PROXY


def test_cursor_is_relative_not_absolute():
    c = RelativeCursorTracker(gain=1000.0, width=640, height=480)
    class T:  # minimal touch
        def __init__(self, x, y):
            self.finger, self.x, self.y = "index", x, y
    c.update([T(0.0, 0.0)], 0.01)   # first frame: no delta
    dx, dy = c.update([T(0.01, 0.0)], 0.01)
    assert dx > 0 and abs(dy) < 1e-12


def test_cursor_speed_clamped():
    c = RelativeCursorTracker(gain=1e9, max_speed_px_s=1000.0)
    class T:
        def __init__(self, x, y):
            self.finger, self.x, self.y = "index", x, y
    c.update([T(0, 0)], 0.01)
    dx, _ = c.update([T(1.0, 0.0)], 0.01)   # huge real displacement
    assert dx <= 1000.0 * 0.01 + 1e-9


def test_tap_short_press_clicks():
    gui = VirtualGUI(seed=0)
    a = TrackpadDeviceAdapter(gui, physics_dt=1 / 120)
    for i in range(10):
        a.step([ev(f=1.0, t=i / 120)], i / 120)
    a.step([], 10 / 120)  # release after ~83 ms < 350 ms
    assert gui.click_registered == 1
    assert len(a.click_events) == 1


def test_long_press_is_not_a_click():
    gui = VirtualGUI(seed=0)
    a = TrackpadDeviceAdapter(gui, physics_dt=1 / 120)
    for i in range(60):  # 500 ms hold
        a.step([ev(f=1.0, t=i / 120)], i / 120)
    a.step([], 60 / 120)
    assert gui.click_registered == 0


def test_moving_touch_is_not_a_tap():
    gui = VirtualGUI(seed=0)
    a = TrackpadDeviceAdapter(gui, physics_dt=1 / 120)
    for i in range(10):
        a.step([ev(f=1.0, x=i * 0.002, t=i / 120)], i / 120)  # 18 mm travel
    a.step([], 10 / 120)
    assert gui.click_registered == 0   # moved beyond 4 mm


def test_gui_seed_determinism_and_reset():
    g1, g2 = VirtualGUI(seed=7), VirtualGUI(seed=7)
    assert g1.target == g2.target
    g3 = VirtualGUI(seed=8)
    assert g1.target != g3.target
    g1.reset(7)
    assert g1.click_registered == 0
    assert g1.target == g2.target


def test_click_on_target_counts_separately():
    gui = VirtualGUI(seed=0)
    gui.reset()
    # place cursor exactly on the target then click
    gui.cursor.x, gui.cursor.y = gui.target
    gui.register_click(t=0.0)
    assert gui.click_registered == 1
    assert gui.click_on_target == 1


def test_action_path_no_touch_no_motion_no_click():
    # VirtualGUI/ContactEvent are plain mutable dataclasses: Python access
    # control does NOT enforce the actor boundary. What these tests pin is
    # the pipeline property: without a trusted recognized touch, the adapter
    # produces zero cursor motion and zero clicks, whatever the raw events
    # claimed (empty frame, missing frame, invalid event).
    gui = VirtualGUI(seed=0)
    a = TrackpadDeviceAdapter(gui, physics_dt=0.1)
    x0, y0 = gui.cursor.x, gui.cursor.y
    a.step([], 0.0)
    a.step(None, 0.1)
    a.step([ContactEvent("index", float("nan"), 0.0, 1.0, 0.2)], 0.2)
    assert gui.cursor.x == x0 and gui.cursor.y == y0
    assert gui.click_registered == 0


def test_adapter_reports_proxy_and_counts():
    gui = VirtualGUI(seed=0)
    a = TrackpadDeviceAdapter(gui, physics_dt=1 / 120)
    s = a.step([ev(f=1.0, t=0.0)], 0.0)
    assert s["recognition"] == RECOGNITION_PROXY
    assert s["touch_count"] == 1
    assert s["contact_count"] == 1


def test_empty_frame_releases_touch():
    r = SimulatedCapacitiveRecognizer()
    assert len(r.update([ev(f=1.0)])) == 1
    assert r.update([]) == []            # observed empty frame releases
    # a fresh strong force after release is a NEW press (not a ghost)
    assert len(r.update([ev(f=1.05)])) == 1


def test_missing_frame_keeps_state():
    r = SimulatedCapacitiveRecognizer()
    assert len(r.update([ev(f=1.0)])) == 1
    assert r.update(None) == []          # missing data: keep state, emit none
    assert len(r.update([ev(f=1.0, t=0.01)])) == 1  # still pressed


def test_out_and_back_slide_does_not_click():
    gui = VirtualGUI(seed=0)
    a = TrackpadDeviceAdapter(gui, physics_dt=1 / 120)
    xs = [0.0, 0.004, 0.008, 0.004, 0.0]  # out-and-back within 4mm endpoints
    for i, x in enumerate(xs):
        a.step([ev(f=1.0, x=x, t=i / 120)], i / 120)
    a.step([], 5 / 120)
    assert gui.click_registered == 0     # max excursion exceeded


def test_multitouch_cancels_tap():
    gui = VirtualGUI(seed=0)
    a = TrackpadDeviceAdapter(gui, physics_dt=1 / 120)
    a.step([ev(f=1.0, t=0.0)], 0.0)
    a.step([ev(f=1.0, t=1 / 120), ev(finger="middle", f=1.0, x=0.05, t=1 / 120)], 1 / 120)
    a.step([ev(f=1.0, t=2 / 120)], 2 / 120)
    a.step([], 3 / 120)
    assert gui.click_registered == 0


def test_finger_identity_change_cancels_tap():
    gui = VirtualGUI(seed=0)
    a = TrackpadDeviceAdapter(gui, physics_dt=1 / 120)
    a.step([ev(f=1.0, t=0.0)], 0.0)
    a.step([ev(finger="middle", f=1.0, x=0.0, t=1 / 120)], 1 / 120)
    a.step([], 2 / 120)
    assert gui.click_registered == 0


def test_nonfinite_time_never_clicks():
    gui = VirtualGUI(seed=0)
    a = TrackpadDeviceAdapter(gui, physics_dt=1 / 120)
    a.step([ev(f=1.0, t=0.0)], 0.0)
    x0, y0 = gui.cursor.x, gui.cursor.y
    # non-finite frame time fails closed: full observation, tap cancelled,
    # no motion, cursor clock unchanged
    s = a.step([], float("nan"))
    assert s["events_valid"] is False
    assert s["sensor_missing"] is False
    assert s["touch_count"] == 0 and s["contact_count"] == 0
    assert "recognition" in s and "cursor" in s
    assert gui.cursor.x == x0 and gui.cursor.y == y0
    assert gui.cursor.t == 0.0
    assert gui.click_registered == 0
    a.step([], 0.05)  # later valid empty frame must not resurrect a click
    assert gui.click_registered == 0


def test_mixed_valid_invalid_frame_fails_closed():
    gui = VirtualGUI(seed=0)
    a = TrackpadDeviceAdapter(gui, physics_dt=0.1)
    a.step([ev(f=1.0, t=0.0)], 0.0)
    x0, y0 = gui.cursor.x, gui.cursor.y
    # one valid finger and one NaN finger in the same frame: the WHOLE frame
    # is untrusted — the valid part must not move the cursor, tap cancelled
    s = a.step([ev(f=1.0, x=0.01, t=0.05),
                ContactEvent("middle", float("nan"), 0.0, 1.0, 0.05)], 0.05)
    assert s["events_valid"] is False
    assert s["touch_count"] == 0
    assert gui.cursor.x == x0 and gui.cursor.y == y0
    a.step([], 0.1)  # trusted empty: no fabricated click
    assert gui.click_registered == 0
    # trusted recontact after the explicit empty restarts normally
    a.step([ev(f=1.0, t=0.2)], 0.2)
    a.step([], 0.25)
    assert gui.click_registered == 1


def test_two_fingers_on_first_frame_no_click():
    gui = VirtualGUI(seed=0)
    a = TrackpadDeviceAdapter(gui, physics_dt=0.1)
    s = a.step([ev(f=1.0, t=0.0),
                ev(finger="middle", f=1.0, x=0.03, t=0.0)], 0.0)
    assert s["touch_count"] == 2  # first observed frame is already multitouch
    a.step([], 0.1)
    assert gui.click_registered == 0


def test_invalid_contact_cancels_pending_tap_no_release():
    gui = VirtualGUI(seed=0)
    a = TrackpadDeviceAdapter(gui, physics_dt=0.1)
    a.step([ev(f=1.0, t=0.0)], 0.0)
    s = a.step([ContactEvent("index", float("nan"), 0.0, 1.0, 0.1)], 0.1)
    assert s["events_valid"] is False
    assert gui.click_registered == 0
    # the invalid sample is not a trusted lift: explicit empty keeps it at 0
    a.step([], 0.2)
    assert gui.click_registered == 0
    # a fresh press after the explicit empty can restart normally
    a.step([ev(f=1.0, t=0.3)], 0.3)
    a.step([], 0.35)
    assert gui.click_registered == 1


def test_nonmonotonic_frame_time_cancels_tap():
    gui = VirtualGUI(seed=0)
    a = TrackpadDeviceAdapter(gui, physics_dt=0.1)
    a.step([ev(f=1.0, t=1.0)], 1.0)
    s = a.step([], 0.9)  # frame time went backwards
    assert s["events_valid"] is False
    assert gui.click_registered == 0
    a.step([], 1.1)  # a later valid empty frame must not resurrect a click
    assert gui.click_registered == 0


def test_event_time_must_match_frame_time():
    gui = VirtualGUI(seed=0)
    a = TrackpadDeviceAdapter(gui, physics_dt=0.1)
    a.step([ev(f=1.0, t=0.0)], 0.0)
    s = a.step([ev(f=1.0, t=0.2)], 0.1)  # future event
    assert s["events_valid"] is False
    a.step([], 0.2)
    assert gui.click_registered == 0


def test_stale_event_in_long_gap_is_rejected():
    gui = VirtualGUI(seed=0)
    a = TrackpadDeviceAdapter(gui, physics_dt=0.1)
    a.step([ev(f=1.0, t=0.0)], 0.0)
    # event timestamp matches neither this frame nor "newer than last";
    # inside a long gap it is still stale, not a fresh observation
    s = a.step([ev(f=1.0, t=0.05)], 0.1)
    assert s["events_valid"] is False
    assert gui.click_registered == 0


def test_finger_moving_away_moves_cursor_up():
    class T:
        def __init__(self, x, y):
            self.finger, self.x, self.y = "index", x, y
    c = RelativeCursorTracker(gain=1000.0)
    c.update([T(0.0, 0.0)], 0.01)
    dx, dy = c.update([T(0.0, 0.01)], 0.01)
    assert abs(dx) < 1e-12 and dy < 0  # pad +y away => pixel dy negative
    gui = VirtualGUI(seed=0)
    a = TrackpadDeviceAdapter(gui, physics_dt=0.1)
    a.step([ev(f=1.0, y=0.0, t=0.0)], 0.0)
    y0 = gui.cursor.y
    a.step([ev(f=1.0, y=0.01, t=0.1)], 0.1)
    assert gui.cursor.y < y0
    a.step([ev(f=1.0, y=0.02, t=0.2)], 0.2)
    assert gui.cursor.y < y0 - 10  # keeps moving up, not clamped at 0


def test_multitouch_does_not_sum_cursor_deltas():
    gui = VirtualGUI(seed=0)
    a = TrackpadDeviceAdapter(gui, physics_dt=0.1)
    a.step([ev(f=1.0, x=0.0, t=0.0)], 0.0)
    x0 = gui.cursor.x
    a.step([ev(f=1.0, x=0.01, t=0.1),
            ev(finger="middle", f=1.0, x=0.02, t=0.1)], 0.1)
    assert gui.cursor.x == x0          # no summed all-finger delta
    a.step([ev(f=1.0, x=0.01, t=0.2)], 0.2)  # back to a single finger
    assert gui.cursor.x == x0          # re-baselined: no phantom jump


def test_finger_identity_change_rebaselines_cursor():
    gui = VirtualGUI(seed=0)
    a = TrackpadDeviceAdapter(gui, physics_dt=0.1)
    a.step([ev(f=1.0, x=0.0, t=0.0)], 0.0)
    x0 = gui.cursor.x
    a.step([ev(finger="middle", f=1.0, x=0.05, t=0.1)], 0.1)
    assert gui.cursor.x == x0  # a different finger is a new baseline, no jump


def test_missing_frame_cancels_tap_and_rebaselines_without_jump():
    gui = VirtualGUI(seed=0)
    a = TrackpadDeviceAdapter(gui, physics_dt=0.1)
    a.step([ev(f=1.0, x=0.0, t=0.0)], 0.0)
    a.step([ev(f=1.0, x=0.001, t=0.1)], 0.1)
    x_mid = gui.cursor.x
    a.step(None, 0.2)
    s = a.step([ev(f=1.0, x=0.01, t=0.3)], 0.3)
    assert gui.cursor.x == x_mid  # gap motion is not flung: fresh baseline
    assert s["events_valid"] is True
    a.step([], 0.35)
    assert gui.click_registered == 0  # the missing frame cancelled the tap
