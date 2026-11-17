"""Trackpad device adapter: contact -> recognition -> cursor -> tap -> GUI.

This module is pure Python and deliberately independent of Isaac Sim. The
*only* input from the physics side is a stream of :class:`ContactEvent`
objects produced by the pad contact sensor. The actor can never set the
cursor or the button directly: all cursor motion is derived from recognized
touches, and all clicks come from the tap state machine.

Simulated capacitive recognition is an explicit PROXY: a real Magic
Trackpad does capacitive imaging; we instead threshold contact force/area
from the physics contact sensor. Calibration to a real device is pending and
flagged in every emitted observation via ``recognition="sim_force_proxy"``.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

RECOGNITION_PROXY = "sim_force_proxy"

# Simulator runs on a single clock, so event timestamps should match the
# frame time exactly; the small tolerance only absorbs float jitter.
TIME_TOL = 1e-6  # s


@dataclass
class ContactEvent:
    """One fingertip<->pad contact sample from the physics contact sensor.

    x/y are pad-local coordinates in meters, origin at the pad centre,
    +x to the right of the operator, +y away from the operator.
    """

    finger: str
    x: float
    y: float
    normal_force: float  # N, >= 0
    t: float  # sim time, s
    # derived-from-geometry pseudo penetration (m, >0 means below surface)
    penetration: float = 0.0


@dataclass
class RecognizedTouch:
    finger: str
    x: float
    y: float
    force: float
    t: float
    recognition: str = RECOGNITION_PROXY


@dataclass
class CursorState:
    x: float  # GUI pixels
    y: float
    pressed: bool
    t: float


class SimulatedCapacitiveRecognizer:
    """Threshold + hysteresis contact recognizer (explicit proxy).

    A touch starts when force exceeds ``on_threshold`` and ends when it
    drops below ``off_threshold``. Forces in between keep the previous
    state. Non-finite inputs are rejected (never recognized); a rejected
    sample for an active finger also terminates that touch as untrusted
    and sets ``invalid_observed`` — an invalid observation is distinct
    from an observed empty frame (``[]``). Per-finger event times must be
    monotonic within ``TIME_TOL``.
    """

    def __init__(self, on_threshold: float = 0.15, off_threshold: float = 0.08):
        if not 0 < off_threshold < on_threshold:
            raise ValueError("need 0 < off < on")
        self.on_threshold = float(on_threshold)
        self.off_threshold = float(off_threshold)
        self._active: dict[str, RecognizedTouch] = {}
        self.invalid_observed = False

    def update(self, events: list[ContactEvent] | None) -> list[RecognizedTouch]:
        """Advance recognition.

        ``events=None`` means the sensor frame is MISSING (no data): state is
        kept, nothing is emitted, ``invalid_observed`` is False. ``events=[]``
        is an observed empty frame: all active touches are released
        immediately (no decay timeout). Events with non-finite fields or
        non-monotonic timestamps are dropped and flagged on
        ``invalid_observed``; they never produce, keep, or release a touch.
        """
        self.invalid_observed = False
        if events is None:
            return []
        out: list[RecognizedTouch] = []
        seen = set()
        for ev in events:
            if not all(math.isfinite(v) for v in (ev.x, ev.y, ev.normal_force)):
                self.invalid_observed = True
                self._active.pop(ev.finger, None)  # invalid kills the touch
                continue
            if not math.isfinite(ev.t):
                self.invalid_observed = True
                self._active.pop(ev.finger, None)
                continue
            active = self._active.get(ev.finger)
            if active is not None and ev.t < active.t - TIME_TOL:
                self.invalid_observed = True
                del self._active[ev.finger]
                continue
            seen.add(ev.finger)
            if active is None:
                if ev.normal_force >= self.on_threshold:
                    self._active[ev.finger] = RecognizedTouch(
                        ev.finger, ev.x, ev.y, ev.normal_force, ev.t)
                    out.append(self._active[ev.finger])
            else:
                if ev.normal_force < self.off_threshold:
                    del self._active[ev.finger]
                else:
                    active.x, active.y = ev.x, ev.y
                    active.force, active.t = ev.normal_force, ev.t
                    out.append(active)
        # fingers not seen this frame decay out by timeout (no ghost touches)
        now = max((e.t for e in events), default=None)
        if now is not None:
            for f in [f for f, a in self._active.items() if f not in seen]:
                if now - self._active[f].t > 0.25:
                    del self._active[f]
        return out


class RelativeCursorTracker:
    """Touch movement -> relative cursor motion (trackpad semantics).

    The cursor never teleports to the touch point: a recognized touch
    contributes a *delta* proportional to touch displacement since the last
    frame, scaled by ``gain`` (px per meter), and clamped to
    ``max_speed_px_s`` so a physics glitch cannot fling the cursor.

    Coordinate contract: pad +y is away from the operator, GUI pixel +y is
    down, so a finger moving away moves the cursor UP (negative pixel dy).

    Stage1 is single-finger. Multitouch frames contribute no cursor delta
    and clear the baseline, and a single finger whose identity differs from
    the previous frame re-baselines instead of jumping; either way the next
    trusted touch starts from a fresh baseline.
    """

    def __init__(self, gain: float = 3200.0, max_speed_px_s: float = 2500.0,
                 width: int = 640, height: int = 480):
        self.gain = float(gain)
        self.max_speed_px_s = float(max_speed_px_s)
        self.width = int(width)
        self.height = int(height)
        self._last: dict[str, tuple[float, float]] = {}

    def reset(self) -> None:
        self._last.clear()

    def update(self, touches: list[RecognizedTouch], dt: float) -> tuple[float, float]:
        dx = dy = 0.0
        if dt <= 0:
            return 0.0, 0.0
        if len(touches) > 1:
            # stage1 single-finger: multitouch is not a trusted gesture.
            # Never sum per-finger deltas; drop the baseline so returning to
            # one finger re-baselines instead of producing a phantom jump.
            self._last.clear()
            return 0.0, 0.0
        current = {t.finger: (t.x, t.y) for t in touches}
        if len(current) == 1 and self._last:
            finger = next(iter(current))
            if finger not in self._last:
                # finger identity discontinuity: re-baseline, no phantom jump
                self._last = current
                return 0.0, 0.0
        for finger, (x, y) in current.items():
            if finger in self._last:
                dx += (x - self._last[finger][0]) * self.gain
                # pad +y away from operator; GUI pixel +y down => cursor up
                dy -= (y - self._last[finger][1]) * self.gain
        self._last = current
        cap = self.max_speed_px_s * dt
        dx = max(-cap, min(cap, dx))
        dy = max(-cap, min(cap, dy))
        return dx, dy


@dataclass
class VirtualGUI:
    """Deterministic point-and-click GUI (teacher truth, not actor input).

    The target is a disc at a fixed, seeded position. Success fields are
    independent: ``cursor_over_target`` tracks hover, ``click_registered``
    counts valid taps, ``click_on_target`` counts taps released on target.
    """

    width: int = 640
    height: int = 480
    target_radius_px: float = 26.0
    target: tuple[float, float] = (448.0, 160.0)
    seed: int = 0
    cursor: CursorState = field(default_factory=lambda: CursorState(320, 240, False, 0.0))
    cursor_over_target: bool = False
    click_registered: int = 0
    click_on_target: int = 0
    history: list = field(default_factory=list)

    def __post_init__(self):
        self.reset(self.seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        # deterministic target layout from seed (LCG, stable across runs)
        s = (1103515245 * (self.seed + 7) + 12345) & 0x7FFFFFFF
        self.target = (float(120 + (s % 400)), float(100 + ((s >> 8) % 280)))
        self.cursor = CursorState(self.width / 2, self.height / 2, False, 0.0)
        self.cursor_over_target = False
        self.click_registered = 0
        self.click_on_target = 0
        self.history.clear()

    def step(self, t: float) -> dict:
        dx = self.cursor.x - self.target[0]
        dy = self.cursor.y - self.target[1]
        self.cursor_over_target = (dx * dx + dy * dy) ** 0.5 <= self.target_radius_px
        state = {
            "t": t,
            "cursor": (round(self.cursor.x, 3), round(self.cursor.y, 3)),
            "pressed": self.cursor.pressed,
            "target": self.target,
            "cursor_over_target": self.cursor_over_target,
            "click_registered": self.click_registered,
            "click_on_target": self.click_on_target,
        }
        self.history.append(state)
        return state

    def register_click(self, t: float) -> None:
        self.click_registered += 1
        dx = self.cursor.x - self.target[0]
        dy = self.cursor.y - self.target[1]
        if (dx * dx + dy * dy) ** 0.5 <= self.target_radius_px:
            self.click_on_target += 1


class TrackpadDeviceAdapter:
    """Composes recognizer + cursor tracker + tap FSM + GUI.

    ``step`` consumes raw physics contact events for one frame and advances
    the whole device pipeline. This is the ONLY lawful path from physics to
    GUI state in the environment. Each returned state exposes
    ``sensor_missing`` and ``events_valid``. Fail closed: a missing frame, a
    non-finite/backwards frame time, or ANY invalid event (time/coords/force)
    cancels the pending tap, resets the cursor delta baseline, contributes no
    cursor motion, and returns the full observation with
    ``events_valid=False`` while the recognizer keeps its last valid state.
    A fresh trusted touch after an explicit empty frame re-baselines.
    """

    def __init__(self, gui: VirtualGUI, physics_dt: float,
                 tap_max_duration: float = 0.35,
                 tap_max_movement_m: float = 0.004,
                 on_threshold: float = 0.15,
                 off_threshold: float = 0.08):
        self.gui = gui
        self.dt = float(physics_dt)
        self.recognizer = SimulatedCapacitiveRecognizer(on_threshold, off_threshold)
        self.tracker = RelativeCursorTracker(width=gui.width, height=gui.height)
        self.tap_max_duration = float(tap_max_duration)
        self.tap_max_movement = float(tap_max_movement_m)
        self._press_t: float | None = None
        self._press_xy: tuple[float, float] | None = None
        self._press_last_xy: tuple[float, float] | None = None
        self._press_finger: str | None = None
        self._press_max_excursion: float = 0.0
        self._tap_cancelled: bool = False
        self._was_pressed = False
        self._last_frame_t: float | None = None
        self.pressed = False
        self.last_touches: list[RecognizedTouch] = []
        self.click_events: list[dict] = []

    def reset(self, seed: int | None = None) -> None:
        self.gui.reset(seed)
        self.tracker.reset()
        self.recognizer._active.clear()
        self._press_t = None
        self._press_xy = None
        self._press_last_xy = None
        self._press_finger = None
        self._press_max_excursion = 0.0
        self._tap_cancelled = False
        self._was_pressed = False
        self._last_frame_t = None
        self.recognizer.invalid_observed = False
        self.pressed = False
        self.last_touches = []
        self.click_events = []

    def step(self, events: list[ContactEvent] | None, t: float) -> dict:
        sensor_missing = events is None
        n_raw = 0 if sensor_missing else len(events)
        untrusted = False  # frame contributes no motion, no FSM advance, no click
        if sensor_missing:
            # existing contract: no data — keep recognizer state, emit nothing
            touches = self.recognizer.update(None)
        elif (not math.isfinite(t)
              or (self._last_frame_t is not None
                  and t < self._last_frame_t - TIME_TOL)
              or any(not math.isfinite(ev.t)
                     or abs(ev.t - t) > TIME_TOL  # stale-in-gap is still stale
                     or not all(math.isfinite(v)
                                for v in (ev.x, ev.y, ev.normal_force))
                     for ev in events)):
            # FAIL CLOSED: one bad frame time or one bad event invalidates the
            # WHOLE frame — the trusted part must not move the cursor either.
            # Keep the recognizer state and the last valid clock; cancel the
            # pending tap and reset the cursor delta baseline.
            untrusted = True
            touches = self.recognizer.update(None)
        else:
            self._last_frame_t = t
            touches = self.recognizer.update(events)

        self.last_touches = touches
        dx = dy = 0.0
        if untrusted:
            self.tracker.reset()
        else:
            dx, dy = self.tracker.update(touches, self.dt)
        cur = self.gui.cursor
        cur.x = min(self.gui.width - 1, max(0.0, cur.x + dx))
        cur.y = min(self.gui.height - 1, max(0.0, cur.y + dy))
        if math.isfinite(t):
            cur.t = t

        primary = touches[0] if touches else None
        if primary is not None:
            if not self._was_pressed:
                self._was_pressed = True
                self._press_t = primary.t
                self._press_xy = (primary.x, primary.y)
                self._press_finger = primary.finger
                self._press_max_excursion = 0.0
                # a first frame that already carries several fingers is a
                # multitouch gesture, never a tap
                self._tap_cancelled = len(touches) > 1
            else:
                # finger identity change or multitouch cancels the pending tap
                if primary.finger != self._press_finger or len(touches) > 1:
                    self._tap_cancelled = True
            self._press_last_xy = (primary.x, primary.y)
            if self._press_xy is not None:
                self._press_max_excursion = max(
                    self._press_max_excursion,
                    math.hypot(primary.x - self._press_xy[0],
                               primary.y - self._press_xy[1]))
        elif self._was_pressed:
            if sensor_missing or untrusted:
                # missing/invalid frame is NOT a trusted release: cancel the
                # pending tap and keep the press open for a later explicit
                # empty frame (or reset) before a fresh press can click.
                self._tap_cancelled = True
            else:
                # observed empty frame => release
                self._was_pressed = False
                held = t - (self._press_t if self._press_t is not None else t)
                moved = self._press_max_excursion
                if (held >= 0.0 and not self._tap_cancelled
                        and held <= self.tap_max_duration
                        and moved <= self.tap_max_movement):
                    self.gui.register_click(t)
                    self.click_events.append({"t": t, "x": cur.x, "y": cur.y})
        self.pressed = self._was_pressed
        cur.pressed = self.pressed
        state = self.gui.step(t if math.isfinite(t) else cur.t)
        state["touch_count"] = len(touches)
        state["touch_fingers"] = [x.finger for x in touches]
        state["recognition"] = RECOGNITION_PROXY
        state["contact_count"] = n_raw
        state["sensor_missing"] = sensor_missing
        state["events_valid"] = not sensor_missing and not untrusted
        return state
