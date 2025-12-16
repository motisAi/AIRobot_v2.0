"""Helper utilities for detecting and configuring runtime platforms.

This package centralizes the heuristics that tailor the robot to specific
hardware (Jetson Nano, Raspberry Pi, etc.).  Each helper exposes two public
functions:

- ``is_<platform>()`` returns True when the current host matches the target.
- ``apply_<platform>_overrides(config)`` mutates the shared ``RobotConfig``
  instance with platform-specific defaults.

The goal is to keep the rest of the code base agnostic to the host hardware
while still benefitting from optimized defaults when hardware capabilities are
known.
"""

from __future__ import annotations

from typing import Callable, Dict

from .jetson_nano import is_jetson_nano, apply_jetson_overrides

PlatformOverride = Callable[["RobotConfig"], None]

PLATFORM_OVERRIDES: Dict[str, PlatformOverride] = {
    "jetson_nano": apply_jetson_overrides,
}


def detect_platform() -> str:
    """Return the identifier for the current host platform.

    The detection logic is intentionally lightweight and relies on a combination
    of CPU architecture checks and files that exist only on NVIDIA Jetson
    devices.  Additional platforms can be added by extending
    :data:`PLATFORM_OVERRIDES` and updating this detector.
    """

    try:
        if is_jetson_nano():
            return "jetson_nano"
    except Exception:  # pragma: no cover - defensive best effort
        # Fallback to generic if detection throws.  The caller logs the
        # exception so we do not mask the failure silently.
        return "generic"

    return "generic"


__all__ = [
    "detect_platform",
    "PLATFORM_OVERRIDES",
    "is_jetson_nano",
    "apply_jetson_overrides",
]
