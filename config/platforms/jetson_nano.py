"""Jetson Nano specific configuration helpers."""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from config.settings import RobotConfig

JETSON_SIGNATURE_FILES = (
    Path("/etc/nv_tegra_release"),
    Path("/sys/firmware/devicetree/base/model"),
)


def is_jetson_nano() -> bool:
    """Return True when running on a Jetson Nano with L4T/JetPack.

    The heuristic checks both the CPU architecture (Jetson Nano is aarch64)
    and the presence of files that NVIDIA's BSP always installs.  The helper is
    intentionally conservative—if any check fails we fall back to "generic" so
    that the rest of the application still boots.
    """

    try:
        if platform.machine().lower() not in {"aarch64", "armv8l"}:
            return False

        for signature in JETSON_SIGNATURE_FILES:
            if signature.exists():
                try:
                    marker = signature.read_text(errors="ignore").lower()
                except Exception:
                    marker = ""
                if "jetson" in marker or "tegra" in marker:
                    return True

        return Path("/etc/nv_tegra_release").exists()
    except Exception:  # pragma: no cover - defensive
        return False


def apply_jetson_overrides(robot_config: "RobotConfig") -> None:
    """Mutate ``robot_config`` with Jetson Nano friendly defaults.

    The overrides stay within pure Python where possible while still enabling
    CUDA/TensorRT optimizations if the user installed them manually.  Every
    change is wrapped in ``try/except`` so a partially configured device never
    prevents the robot from starting.
    """

    try:
        robot_config.system.platform_name = "jetson_nano"
        robot_config.system.enable_gpu = True
        robot_config.system.enable_hailo = False
        robot_config.system.use_tensor_rt = True
        robot_config.system.vision_thread_count = 1
        robot_config.system.audio_thread_count = 1
        robot_config.system.frame_skip = max(robot_config.system.frame_skip, 4)

        robot_config.hardware.camera_index = 0  # USB cam by default
        robot_config.hardware.camera_resolution = (640, 480)
        robot_config.hardware.camera_fps = 24
        robot_config.hardware.uart_device_map.setdefault("esp32", "/dev/ttyTHS1")
        robot_config.hardware.uart_device_map.setdefault("sim7600x", "/dev/ttyTHS0")

        robot_config.hardware.wake_word_microphone_name = (
            robot_config.hardware.wake_word_microphone_name
            or os.getenv("JETSON_WAKE_MIC", "USB")
        )
        robot_config.hardware.speech_microphone_name = (
            robot_config.hardware.speech_microphone_name
            or os.getenv("JETSON_DIALOG_MIC", "USB")
        )

        logging.info("Applied Jetson Nano overrides")
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("Failed to apply Jetson overrides: %s", exc)
