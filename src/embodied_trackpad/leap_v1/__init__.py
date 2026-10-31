"""LEAP Hand v1 Full baseline (16 independent channels, four fingers).

Independent of the v2 ``leap_sim`` package: separate pinned Menagerie assets,
separate manifest, separate embodiment/calibration IDs. No v2 code paths are
modified and no v2 results are relabeled.

The model/diagnostics stack requires the optional physics dependencies
(``mujoco``, ``numpy``). They are imported LAZILY via module ``__getattr__``
so that the stdlib-only asset tooling (``embodied_trackpad.leap_v1.assets``,
``tools/fetch_leap_v1_assets.py``) works on a foundation-only installation.
Requesting a model export without the physics deps raises the underlying
ModuleNotFoundError naming the missing package.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from embodied_trackpad.leap_v1.model import (
        CALIBRATION_ID,
        DOWNLOAD_ROOT,
        EMBODIMENT_ID,
        FINGERS,
        FINGER_JOINTS,
        MANIFEST_JSON,
        MOTOR_JOINTS,
        OPEN_POSE,
        TIP_GEOMS,
        TIP_SITES,
        LeapV1Hand,
        load_hand,
        model_inventory,
        open_pose_qpos,
        verify_manifest,
    )

# Public model-layer names, resolved lazily from .model on first access.
_LAZY_MODEL_EXPORTS = frozenset(
    {
        "CALIBRATION_ID",
        "DOWNLOAD_ROOT",
        "EMBODIMENT_ID",
        "FINGERS",
        "FINGER_JOINTS",
        "MANIFEST_JSON",
        "MOTOR_JOINTS",
        "OPEN_POSE",
        "TIP_GEOMS",
        "TIP_SITES",
        "LeapV1Hand",
        "load_hand",
        "model_inventory",
        "open_pose_qpos",
        "verify_manifest",
    }
)

__all__ = sorted(_LAZY_MODEL_EXPORTS)


def __getattr__(name: str):
    if name in _LAZY_MODEL_EXPORTS:
        module = importlib.import_module("embodied_trackpad.leap_v1.model")
        value = getattr(module, name)
        globals()[name] = value  # cache for subsequent attribute access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_MODEL_EXPORTS)
