"""Training-attached monitor interface (observer side + publisher contract).

This package lets a trainer/renderer publish its latest actual frame and
versioned metadata into a run directory, and lets a local viewer serve that
state truthfully. It contains no simulation, no policy, no motor output and
no training; labels are producer-reported until independent integration
acceptance.
"""
from .protocol import (
    FRAME_NAME,
    LABEL_BASIS,
    LIFECYCLES,
    METADATA_NAME,
    METRIC_KEYS,
    MonitorProtocolError,
    POLICY_KINDS,
    RECORD_NAME,
    SCHEMA_VERSION,
    SOURCE_KINDS,
    Snapshot,
    derive_display,
    is_training_active,
    read_verified_record,
    validate_record,
)
from .publisher import MonitorPublisher, png_dimensions, utc_now_iso

__all__ = [
    "FRAME_NAME",
    "LABEL_BASIS",
    "LIFECYCLES",
    "METADATA_NAME",
    "METRIC_KEYS",
    "MonitorProtocolError",
    "MonitorPublisher",
    "POLICY_KINDS",
    "RECORD_NAME",
    "SCHEMA_VERSION",
    "SOURCE_KINDS",
    "Snapshot",
    "derive_display",
    "is_training_active",
    "png_dimensions",
    "read_verified_record",
    "utc_now_iso",
    "validate_record",
]
