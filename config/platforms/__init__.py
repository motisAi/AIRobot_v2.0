"""Helper utilities for detecting and configuring runtime platforms.

This package centralizes the heuristics that tailor the robot to specific
hardware (Raspberry Pi 5, etc.).  Each helper exposes two public functions:

- ``is_<platform>()`` returns True when the current host matches the target.
- ``apply_<platform>_overrides(config)`` mutates the shared ``RobotConfig``
  instance with platform-specific defaults.

The goal is to keep the rest of the code base agnostic to the host hardware
while still benefitting from optimized defaults when hardware capabilities are
known.
"""

from __future__ import annotations

from typing import Callable, Dict

from .raspberry_pi5 import is_raspberry_pi5, apply_rpi5_overrides

PlatformOverride = Callable[["RobotConfig"], None]

PLATFORM_OVERRIDES: Dict[str, PlatformOverride] = {
    "raspberry_pi5": apply_rpi5_overrides,
}


def detect_platform() -> str:
    """Return the identifier for the current host platform.

    Detection order: Raspberry Pi 5 first (primary target), then generic
    fallback.
    """

    try:
        if is_raspberry_pi5():
            return "raspberry_pi5"
    except Exception:  # pragma: no cover
        pass

    return "generic"


__all__ = [
    "detect_platform",
    "PLATFORM_OVERRIDES",
    "is_raspberry_pi5",
    "apply_rpi5_overrides",
]


__all__ = [
    "detect_platform",
    "PLATFORM_OVERRIDES",
    "is_raspberry_pi5",
    "apply_rpi5_overrides",
    "is_jetson_nano",
    "apply_jetson_overrides",
]
