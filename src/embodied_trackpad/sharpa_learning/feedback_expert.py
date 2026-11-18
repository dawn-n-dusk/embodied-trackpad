"""Real relative-trackpad feedback expert for Sharpa BC collection.

Privileged simulator teacher: reads actual GUI cursor/target and physical
contact pose, but never writes cursor/click state, never teleports the robot,
and never steps the simulator.  Emits normalized 29-DoF joint deltas consumed
by :meth:`SharpaEnv.step`.

The trackpad is **relative**, not a touch screen.  A new contact establishes a
baseline; the cursor moves by ``finger_displacement * gain`` afterwards.
Landing the fingertip at mapped target coordinates does **not** move the
cursor.  The expert therefore servos the *cursor error* to zero via small
relative pad displacements, then performs a short tap while the cursor stays
on target.

Kinematics use the measured current arm configuration as the FK baseline and
solve differentially for the desired *actual* fingertip displacement.  This
compensates the fixed curled hand pose, which would otherwise shift the real
tip away from ``solve_home``'s zero-hand model and cause drift.
"""
from __future__ import annotations

import math
from typing import ClassVar

import numpy as np

from .env import ACTION_DIM, ARM_DELTA_SCALE, FINGERS, HAND_DELTA_SCALE, SharpaEnv, SUBSTEPS

# Hand joint name heuristics.  These are used only to pick a safe static hand
# pose; the expert never relies on a particular joint name for correctness.
_FINGER_NAMES = ("index", "middle", "ring", "pinky", "thumb")
_FLEXION_KEYWORDS = ("MCP", "PIP", "IP", "DIP", "CMC", "TM", "flex", "curl")
_ABDUCTION_KEYWORDS = ("abduc", "ABD", "spread", "_AA")


def _finger_from_joint_name(name: str) -> str | None:
    for f in _FINGER_NAMES:
        if f in name:
            return f
    return None


def _is_flexion_joint(name: str) -> bool:
    has_flex = any(k in name for k in _FLEXION_KEYWORDS)
    has_abd = any(k in name for k in _ABDUCTION_KEYWORDS)
    return has_flex and not has_abd


class FeedbackExpert:
    """Privileged feedback teacher for one relative trackpad tap episode.

    Phase machine (replanned every ``action`` call from the current student
    observation):

        clear -> approach -> slide -> tap_lift -> tap_descend -> tap_release
        -> done

    Recovery branches:

        slide -> recontact -> slide (or reanchor on timeout)
        slide -> reanchor -> approach -> slide
        tap_release -> tap_lift (bounded retry if click missing)

    ``fail`` is entered on IK failure, excessive recontact/reanchor attempts,
    or an internal phase timeout.  The expert still emits legal hold actions so
    the episode can terminate cleanly through the environment's normal success
    criteria (or truncation).
    """

    # Force thresholds (N).  Keep close to device.py defaults so the teacher
    # observes the same contact state as the device adapter.
    CONTACT_ON_N: ClassVar[float] = 0.20
    CONTACT_OFF_N: ClassVar[float] = 0.08

    # Once the cursor is within this many pixels of the target, stop sliding
    # and attempt the final tap.  Must be <= the GUI target radius.
    SLIDE_CURSOR_TOL_PX: ClassVar[float] = 8.0

    # Per-step displacement caps (m).  Small values keep contact forces low
    # and avoid cursor overshoot on the relative trackpad.
    MAX_SLIDE_DELTA_M: ClassVar[float] = 0.003
    MAX_REANCHOR_DELTA_M: ClassVar[float] = 0.005
    MAX_APPROACH_DELTA_M: ClassVar[float] = 0.004
    MAX_TAP_LIFT_DELTA_M: ClassVar[float] = 0.010
    MAX_TAP_PRESS_DELTA_M: ClassVar[float] = 0.004

    # Pad-edge guard: stay inside the physical pad by this margin.
    PAD_MARGIN_M: ClassVar[float] = 0.005

    # Phase timeouts in decision steps (15 Hz -> ~0.067 s/step).
    CLEAR_TIMEOUT_STEPS: ClassVar[int] = 30
    APPROACH_TIMEOUT_STEPS: ClassVar[int] = 120
    SLIDE_TIMEOUT_STEPS: ClassVar[int] = 360
    TAP_TIMEOUT_STEPS: ClassVar[int] = 60
    RECONTACT_TIMEOUT_STEPS: ClassVar[int] = 20
    REANCHOR_TIMEOUT_STEPS: ClassVar[int] = 40
    MAX_RECONTACT_ATTEMPTS: ClassVar[int] = 2
    MAX_REANCHOR_ATTEMPTS: ClassVar[int] = 2
    MAX_TAP_ATTEMPTS: ClassVar[int] = 2

    def __init__(self, env: SharpaEnv):
        self.env = env
        self.cfg = env.scene_config
        self.dt_dec = float(env.dt) * SUBSTEPS
        # One static legal hand pose: non-index fingers raised, index slightly
        # curled so it is the contact finger.  Differential IK compensates the
        # offset between this curled pose and solve_home's zero-hand model.
        self._hand_pose = self._make_hand_pose(index_flexion=0.06)
        self._phase = "clear"
        self._phase_enter_step = 0
        self._step = 0
        self._contact_step: int | None = None
        self._recontact_attempts = 0
        self._reanchor_attempts = 0
        self._tap_attempts = 0
        self._tap_subphase: str | None = None

    def reset(self) -> None:
        """Reset the internal phase machine.  Does not touch simulator state."""
        self._phase = "clear"
        self._phase_enter_step = 0
        self._step = 0
        self._contact_step = None
        self._recontact_attempts = 0
        self._reanchor_attempts = 0
        self._tap_attempts = 0
        self._tap_subphase = None
        self._hand_pose = self._make_hand_pose(index_flexion=0.06)

    def _make_hand_pose(self, index_flexion: float) -> np.ndarray:
        """Return a legal hand pose with non-index fingers clear.

        Lateral/abduction joints (names containing ``_AA`` or ``abduc``) are
        left near zero; only flexion joints are bent.  The index finger is
        curled slightly positive so it touches first.  All values are clipped
        to the measured joint limits with margin.
        """
        lo, hi = self.env.joint_limits
        names = self.env.action_names
        q = np.zeros(ACTION_DIM, dtype=float)
        for ci, name in enumerate(names):
            if ci < 7:
                continue
            finger = _finger_from_joint_name(name)
            if finger == "index":
                val = float(index_flexion) if _is_flexion_joint(name) else 0.0
            elif finger is not None:
                val = -0.12 if _is_flexion_joint(name) else 0.0
            else:
                val = 0.0
            margin = 0.01
            q[ci] = float(np.clip(val, lo[ci] + margin, hi[ci] - margin))
        return q

    # ------------------------------------------------------------------
    # Observation / privileged-state helpers
    # ------------------------------------------------------------------
    def _q_meas(self, obs: dict) -> np.ndarray:
        return np.asarray(obs["proprio"], dtype=float)[:ACTION_DIM]

    def _qd_meas(self, obs: dict) -> np.ndarray:
        return np.asarray(obs["proprio"], dtype=float)[ACTION_DIM:]

    def _index_contact_state(self, obs: dict) -> tuple[float, bool]:
        """Return (force_N, valid).  Invalid frames are treated as unknown,
        not as observed release."""
        idx = list(FINGERS).index("index")
        valid = bool(obs["touch_valid"][idx])
        force = float(obs["touch"][idx]) if valid else 0.0
        return force, valid

    def _cursor_error(self) -> tuple[float, float]:
        priv = self.env._privileged()
        cx, cy = priv["oracle_cursor_px"]
        tx, ty = priv["oracle_target_px"]
        return float(tx - cx), float(ty - cy)

    def _cursor_in_target(self) -> bool:
        return bool(self.env._privileged().get("oracle_cursor_over_target", False))

    def _click_on_target(self) -> bool:
        return int(self.env._privileged().get("oracle_click_on_target", 0)) > 0

    def _index_tip_world(self) -> np.ndarray:
        pos = self.env.tips["index"].get_world_poses()[0]
        return np.asarray(pos).reshape(-1)[:3].astype(float)

    def _gain(self) -> float:
        return float(getattr(self.env.adapter.tracker, "gain", 3200.0))

    # ------------------------------------------------------------------
    # Kinematics helpers (differential, measured-FK compensated)
    # ------------------------------------------------------------------
    def _world_to_base(self, p_world: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        ax, ay = cfg.base_xy
        c, s = math.cos(-cfg.base_yaw), math.sin(-cfg.base_yaw)
        dx, dy = float(p_world[0]) - ax, float(p_world[1]) - ay
        return np.array([
            c * dx - s * dy,
            s * dx + c * dy,
            float(p_world[2]) - cfg.table_top_z,
        ], dtype=float)

    def _base_to_world(self, p_base: np.ndarray) -> np.ndarray:
        """Inverse of ``_world_to_base``: arm-base coordinates -> world."""
        cfg = self.cfg
        ax, ay = cfg.base_xy
        c, s = math.cos(cfg.base_yaw), math.sin(cfg.base_yaw)
        bx, by = float(p_base[0]), float(p_base[1])
        return np.array([
            c * bx - s * by + ax,
            s * bx + c * by + ay,
            float(p_base[2]) + cfg.table_top_z,
        ], dtype=float)

    def _fk_zero_hand_tip(self, q7: np.ndarray) -> np.ndarray:
        """Index fingertip position predicted by the zero-hand IK model."""
        from embodied_trackpad.sharpa_sim import kinematics as kn
        cfg = self.cfg
        return kn.fk_index_tip(
            np.asarray(q7, dtype=float),
            tuple(cfg.mount_pos0),
            tuple(cfg.mount_rot0_wxyz),
            tuple(cfg.mount_rot1_wxyz),
        )

    def _solve_ik(self, tip_world: np.ndarray, seeds=()) -> np.ndarray | None:
        from embodied_trackpad.sharpa_sim import kinematics as kn
        target_base = self._world_to_base(np.asarray(tip_world, dtype=float))
        if not seeds:
            seeds = [tuple(self.env.home_q7)]
        home = kn.solve_home(
            target_base,
            tuple(cfg.mount_pos0) if (cfg := self.cfg) else (0.0, 0.0, 0.107),
            tuple(cfg.mount_rot0_wxyz),
            tuple(cfg.mount_rot1_wxyz),
            seeds=seeds,
            max_iter=300,
        )
        if not home.success:
            return None
        return np.asarray(home.q, dtype=float)

    def _arm_move(self, q_meas: np.ndarray, measured_tip: np.ndarray,
                  world_delta: np.ndarray, max_step_m: float) -> np.ndarray | None:
        """Differential arm IK: move actual fingertip by ``world_delta``.

        ``_fk_zero_hand_tip`` returns the tip in the ARM BASE frame, while
        ``world_delta`` is a world-frame displacement.  We therefore convert the
        zero-model tip to world, add the world delta, and let ``_solve_ik``
        convert the target back to base.  This avoids the previous double
        transform where a world delta was added to a base-frame point.
        """
        world_delta = np.asarray(world_delta, dtype=float)
        n = float(np.linalg.norm(world_delta))
        if n > max_step_m:
            world_delta = world_delta * (max_step_m / n)
        q7 = np.asarray(q_meas[:7], dtype=float)
        zero_tip_now_base = self._fk_zero_hand_tip(q7)
        zero_tip_now_world = self._base_to_world(zero_tip_now_base)
        zero_target_world = zero_tip_now_world + world_delta
        q_arm = self._solve_ik(zero_target_world, seeds=[tuple(q7)])
        return q_arm

    def _build_target(self, q_arm: np.ndarray) -> np.ndarray:
        q = np.zeros(ACTION_DIM, dtype=float)
        q[:7] = q_arm
        q[7:] = self._hand_pose[7:]
        return self._clip_limits(q)

    def _clip_limits(self, q: np.ndarray) -> np.ndarray:
        lo, hi = self.env.joint_limits
        margin = 0.005
        return np.clip(q, lo + margin, hi - margin)

    def _delta(self, q_meas: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
        """Normalized delta action from current q to reference q.

        Per-joint clamp to [-1,1] is sufficient: ``SharpaEnv.step`` applies
        the frozen delta_scale and its own velocity/joint-limit clipping.
        Whole-vector norm caps are intentionally omitted because they couple
        unrelated hand errors and starve the arm.
        """
        d = (q_ref - q_meas) / self.env.delta_scale
        return np.clip(d, -1.0, 1.0).astype(np.float32)

    def _safe_hold(self, q_meas: np.ndarray) -> np.ndarray:
        """Hold current joint configuration (zero normalized delta)."""
        return np.zeros(ACTION_DIM, dtype=np.float32)

    def _enter_phase(self, phase: str) -> None:
        self._phase = phase
        self._phase_enter_step = self._step
        if phase == "tap_lift":
            self._tap_subphase = "lift"

    # ------------------------------------------------------------------
    # Utility targets
    # ------------------------------------------------------------------
    def _desired_tip_above_pad_centre(self, hover_m: float) -> np.ndarray:
        cfg = self.cfg
        z = cfg.table_top_z + cfg.pad_size[2] + hover_m
        return np.array([cfg.pad_xy[0], cfg.pad_xy[1], z], dtype=float)

    def _hand_pose_reached(self, q_meas: np.ndarray) -> bool:
        """True when the measured hand joints are close to the clear pose."""
        hand_err = np.abs(q_meas[7:] - self._hand_pose[7:])
        scale = self.env.delta_scale[7:]
        return bool(np.all(hand_err <= 2.0 * scale))

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------
    def _phase_clear(self, q_meas: np.ndarray, force: float,
                     valid: bool) -> np.ndarray:
        q_ref = self._build_target(np.asarray(self.env.home_q7, dtype=float))
        d = self._delta(q_meas, q_ref)
        elapsed = self._step - self._phase_enter_step
        touching = valid and force >= self.CONTACT_OFF_N
        ready = self._hand_pose_reached(q_meas) and not touching
        if ready:
            self._enter_phase("approach")
        elif elapsed > self.CLEAR_TIMEOUT_STEPS:
            # Bounded timeout: proceed if we are not touching, else fail honest.
            if touching:
                self._enter_phase("fail")
            else:
                self._enter_phase("approach")
        return d

    def _phase_approach(self, q_meas: np.ndarray, force: float,
                        valid: bool) -> np.ndarray:
        if self._step - self._phase_enter_step > self.APPROACH_TIMEOUT_STEPS:
            self._enter_phase("fail")
            return self._safe_hold(q_meas)
        progress = self._step - self._phase_enter_step
        hover_m = max(0.0, 0.025 - 0.00025 * progress)
        desired_tip = self._desired_tip_above_pad_centre(hover_m)
        measured_tip = self._index_tip_world()
        world_delta = desired_tip - measured_tip
        q_arm = self._arm_move(q_meas, measured_tip, world_delta,
                               self.MAX_APPROACH_DELTA_M)

        # Actual contact is real feedback and must transition us to slide
        # regardless of whether this single-frame IK succeeded.
        if valid and force >= self.CONTACT_ON_N:
            self._contact_step = self._step
            self._enter_phase("slide")
            if q_arm is None:
                return self._safe_hold(q_meas)
            q_ref = self._build_target(q_arm)
            return self._delta(q_meas, q_ref)

        if q_arm is None:
            return self._safe_hold(q_meas)
        q_ref = self._build_target(q_arm)
        return self._delta(q_meas, q_ref)

    def _phase_slide(self, q_meas: np.ndarray, force: float,
                     valid: bool) -> np.ndarray:
        if self._step - self._phase_enter_step > self.SLIDE_TIMEOUT_STEPS:
            self._enter_phase("fail")
            return self._safe_hold(q_meas)
        if (not valid) or force < self.CONTACT_OFF_N:
            if self._recontact_attempts >= self.MAX_RECONTACT_ATTEMPTS:
                self._enter_phase("fail")
            else:
                self._recontact_attempts += 1
                self._enter_phase("recontact")
            return self._safe_hold(q_meas)

        err_x, err_y = self._cursor_error()
        if math.hypot(err_x, err_y) <= self.SLIDE_CURSOR_TOL_PX:
            self._enter_phase("tap_lift")
            return self._safe_hold(q_meas)

        gain = self._gain()
        dx_pad = float(np.clip(err_x / gain,
                               -self.MAX_SLIDE_DELTA_M, self.MAX_SLIDE_DELTA_M))
        dy_pad = float(np.clip(-err_y / gain,
                               -self.MAX_SLIDE_DELTA_M, self.MAX_SLIDE_DELTA_M))

        # Pad-edge guard: if the planned displacement would leave the physical
        # pad, lift and recentre instead of pretending to tap.
        measured_tip = self._index_tip_world()
        pad_x, pad_y = self.env._pad_local_xy(measured_tip)
        half_w, half_h = self.cfg.pad_size[0] / 2, self.cfg.pad_size[1] / 2
        if (abs(pad_x + dx_pad) > half_w - self.PAD_MARGIN_M or
                abs(pad_y + dy_pad) > half_h - self.PAD_MARGIN_M):
            if self._reanchor_attempts >= self.MAX_REANCHOR_ATTEMPTS:
                self._enter_phase("fail")
            else:
                self._reanchor_attempts += 1
                self._enter_phase("reanchor")
            return self._safe_hold(q_meas)

        yaw = float(self.cfg.pad_yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        world_delta = np.array([c * dx_pad - s * dy_pad,
                                s * dx_pad + c * dy_pad, 0.0], dtype=float)
        q_arm = self._arm_move(q_meas, measured_tip, world_delta,
                               self.MAX_SLIDE_DELTA_M)
        if q_arm is None:
            return self._safe_hold(q_meas)
        q_ref = self._build_target(q_arm)
        return self._delta(q_meas, q_ref)

    def _phase_recontact(self, q_meas: np.ndarray, force: float,
                         valid: bool) -> np.ndarray:
        """Small downward press from current measured tip to re-establish touch."""
        if self._step - self._phase_enter_step > self.RECONTACT_TIMEOUT_STEPS:
            if self._reanchor_attempts >= self.MAX_REANCHOR_ATTEMPTS:
                self._enter_phase("fail")
            else:
                self._reanchor_attempts += 1
                self._enter_phase("reanchor")
            return self._safe_hold(q_meas)
        measured_tip = self._index_tip_world()
        q_arm = self._arm_move(q_meas, measured_tip,
                               np.array([0.0, 0.0, -0.002]), 0.003)

        # Contact re-established: transition back to slide even if IK failed.
        if valid and force >= self.CONTACT_ON_N:
            self._enter_phase("slide")
            if q_arm is None:
                return self._safe_hold(q_meas)
            q_ref = self._build_target(q_arm)
            return self._delta(q_meas, q_ref)

        if q_arm is None:
            return self._safe_hold(q_meas)
        q_ref = self._build_target(q_arm)
        return self._delta(q_meas, q_ref)

    def _phase_reanchor(self, q_meas: np.ndarray, force: float,
                        valid: bool) -> np.ndarray:
        """Lift clear and move back to the pad centre before re-approaching."""
        if self._step - self._phase_enter_step > self.REANCHOR_TIMEOUT_STEPS:
            self._enter_phase("fail")
            return self._safe_hold(q_meas)
        measured_tip = self._index_tip_world()
        desired_tip = self._desired_tip_above_pad_centre(0.020)
        world_delta = desired_tip - measured_tip
        q_arm = self._arm_move(q_meas, measured_tip, world_delta,
                               self.MAX_REANCHOR_DELTA_M)
        if q_arm is None:
            return self._safe_hold(q_meas)
        q_ref = self._build_target(q_arm)
        d = self._delta(q_meas, q_ref)
        # Once we are clear and centred, start a fresh approach.
        lifted = (not valid) or force < self.CONTACT_OFF_N
        close = float(np.linalg.norm(desired_tip - measured_tip)) < 0.003
        if lifted and close:
            self._enter_phase("approach")
        return d

    def _phase_tap(self, q_meas: np.ndarray, force: float,
                   valid: bool) -> np.ndarray:
        if self._step - self._phase_enter_step > self.TAP_TIMEOUT_STEPS:
            self._enter_phase("fail")
            return self._safe_hold(q_meas)

        measured_tip = self._index_tip_world()

        if self._tap_subphase == "lift":
            q_arm = self._arm_move(q_meas, measured_tip,
                                   np.array([0.0, 0.0, self.MAX_TAP_LIFT_DELTA_M]),
                                   self.MAX_TAP_LIFT_DELTA_M)
            if q_arm is None:
                return self._safe_hold(q_meas)
            q_ref = self._build_target(q_arm)
            d = self._delta(q_meas, q_ref)
            if valid and force < self.CONTACT_OFF_N:
                self._tap_subphase = "descend"
            return d

        if self._tap_subphase == "descend":
            q_arm = self._arm_move(q_meas, measured_tip,
                                   np.array([0.0, 0.0, -self.MAX_TAP_PRESS_DELTA_M]),
                                   self.MAX_TAP_PRESS_DELTA_M)
            if q_arm is None:
                return self._safe_hold(q_meas)
            q_ref = self._build_target(q_arm)
            d = self._delta(q_meas, q_ref)
            if valid and force >= self.CONTACT_ON_N:
                self._tap_subphase = "release"
            return d

        if self._tap_subphase == "release":
            q_arm = self._arm_move(q_meas, measured_tip,
                                   np.array([0.0, 0.0, self.MAX_TAP_LIFT_DELTA_M]),
                                   self.MAX_TAP_LIFT_DELTA_M)
            if q_arm is None:
                return self._safe_hold(q_meas)
            q_ref = self._build_target(q_arm)
            d = self._delta(q_meas, q_ref)
            if valid and force < self.CONTACT_OFF_N:
                if self._click_on_target():
                    self._enter_phase("done")
                elif self._tap_attempts < self.MAX_TAP_ATTEMPTS:
                    self._tap_attempts += 1
                    self._enter_phase("tap_lift")
                else:
                    self._enter_phase("fail")
            return d

        return self._safe_hold(q_meas)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def action(self, obs: dict, t: int) -> np.ndarray:
        """Return a finite float32[29] normalized [-1,1] action.

        ``obs`` is the actor observation (screen/wrist RGB, proprio, touch).
        The teacher may additionally read the simulator's privileged state
        (cursor, target, measured tip pose) through ``self.env``.
        """
        self._step = int(t)
        q_meas = self._q_meas(obs)
        force, valid = self._index_contact_state(obs)

        if self._phase == "clear":
            return self._phase_clear(q_meas, force, valid)
        if self._phase == "approach":
            return self._phase_approach(q_meas, force, valid)
        if self._phase == "slide":
            return self._phase_slide(q_meas, force, valid)
        if self._phase == "recontact":
            return self._phase_recontact(q_meas, force, valid)
        if self._phase == "reanchor":
            return self._phase_reanchor(q_meas, force, valid)
        if self._phase == "tap_lift":
            if self._tap_subphase is None:
                self._tap_subphase = "lift"
            return self._phase_tap(q_meas, force, valid)
        if self._phase == "done":
            return self._safe_hold(q_meas)
        if self._phase == "fail":
            return self._safe_hold(q_meas)

        return self._safe_hold(q_meas)


class TeacherEnabledEnv:
    """Thin wrapper that lets a SharpaEnv expose :meth:`expert_action`.

    The wrapper delegates reset/step/close to the underlying environment and
    caches the last observation so the installed :class:`FeedbackExpert` can
    be queried via ``env.expert_action()``.  The actor observation still
    contains only screen/wrist RGB, proprioception and touch; the wrapper does
    not inject oracle GUI state into the actor's observation.

    The constructor accepts either a ready ``SharpaEnv`` instance or a config
    dict (plus optional ``app``) that is forwarded to ``SharpaEnv``, so it can
    be used both directly and as a BC factory string such as
    ``embodied_trackpad.sharpa_learning.feedback_expert:TeacherEnabledEnv``.
    """

    def __init__(self, env: SharpaEnv | None = None,
                 config: dict | None = None, app=None, **kwargs):
        if env is not None:
            self._env = env
        elif config is not None or kwargs:
            self._env = SharpaEnv(config=config, app=app, **kwargs)
        else:
            raise ValueError("TeacherEnabledEnv requires either env, config, or kwargs")
        self.teacher = FeedbackExpert(self)
        self._last_obs: dict | None = None
        self._last_t: int = 0

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def reset(self, seed: int = 0, options: dict | None = None):
        self.teacher.reset()
        self._last_t = 0
        obs, info = self._env.reset(seed=seed, options=options)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._last_obs = obs
        self._last_t += 1
        return obs, reward, terminated, truncated, info

    def expert_action(self) -> np.ndarray:
        if self._last_obs is None:
            raise RuntimeError("reset must be called before expert_action")
        return self.teacher.action(self._last_obs, self._last_t)

    def close(self):
        return self._env.close()
