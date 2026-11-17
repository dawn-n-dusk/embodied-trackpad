"""SharpaWave right-hand + 7-DoF arm trackpad simulation environment.

This package is split into two parts:

- :mod:`~embodied_trackpad.sharpa_sim.config` and
  :mod:`~embodied_trackpad.sharpa_sim.device` are pure Python and run anywhere
  (they are covered by the local ``tests/test_sharpa_sim_*.py`` suite).
- :mod:`~embodied_trackpad.sharpa_sim.isaac_scene` and
  :mod:`~embodied_trackpad.sharpa_sim.isaac_runner` import ``isaacsim``/``pxr``
  and only run inside the pinned Isaac Sim runtime on the Lim workstation.
"""

from .config import SharpaSceneConfig, ARM_PANDA_7DOF, HAND_22_CHANNELS
from .device import (
    ContactEvent,
    SimulatedCapacitiveRecognizer,
    RelativeCursorTracker,
    TrackpadDeviceAdapter,
    VirtualGUI,
)

__all__ = [
    "SharpaSceneConfig",
    "ARM_PANDA_7DOF",
    "HAND_22_CHANNELS",
    "ContactEvent",
    "SimulatedCapacitiveRecognizer",
    "RelativeCursorTracker",
    "TrackpadDeviceAdapter",
    "VirtualGUI",
]
