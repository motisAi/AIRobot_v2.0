"""Raspberry Pi 5 specific configuration helpers.

Detects the RPi5 host and applies Hailo-aware defaults when the AI
accelerator is present.  Falls back to CPU-optimized settings otherwise.
"""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from config.settings import RobotConfig


# Files that exist on Raspberry Pi OS / Ubuntu for Pi
RPI_SIGNATURE_FILES = (
    Path("/proc/device-tree/model"),
    Path("/sys/firmware/devicetree/base/model"),
)


def is_raspberry_pi5() -> bool:
    """Return True when running on a Raspberry Pi 5.

    Checks CPU architecture and the device-tree model string that the kernel
    always exposes.  Conservative: returns False on any error so the robot
    still boots on unknown hardware.
    """
    try:
        if platform.machine().lower() not in {"aarch64", "armv8l"}:
            return False

        for sig_file in RPI_SIGNATURE_FILES:
            if sig_file.exists():
                try:
                    model = sig_file.read_text(errors="ignore").lower()
                except Exception:
                    model = ""
                if "raspberry pi 5" in model:
                    return True

        return False
    except Exception:  # pragma: no cover - defensive
        return False


def apply_rpi5_overrides(robot_config: "RobotConfig") -> None:
    """Mutate ``robot_config`` with Raspberry Pi 5 + Hailo defaults.

    All changes are wrapped in try/except so a partially configured device
    never prevents the robot from starting.
    """
    try:
        robot_config.system.platform_name = "raspberry_pi5"
        robot_config.system.enable_gpu = False  # No CUDA on RPi5
        robot_config.system.use_tensor_rt = False  # No TensorRT on RPi5

        # Hailo-specific tuning (detected separately in settings.py)
        if robot_config.system.enable_hailo:
            robot_config.system.frame_skip = 2  # Hailo is fast
            robot_config.hardware.camera_fps = 30
            robot_config.hardware.camera_resolution = (640, 480)
            robot_config.model.whisper_model = "base"
            robot_config.model.object_model = "yolov8s"  # Hailo can handle small
        else:
            robot_config.system.frame_skip = 4
            robot_config.hardware.camera_fps = 15
            robot_config.hardware.camera_resolution = (320, 240)
            robot_config.model.whisper_model = "tiny"
            robot_config.model.object_model = "yolov8n"

        # RPi5 has 4 cores — keep threads reasonable
        robot_config.system.vision_thread_count = 1
        robot_config.system.audio_thread_count = 1
        robot_config.system.max_threads = 6

        # Default camera on RPi5 (USB cam on /dev/video0)
        robot_config.hardware.camera_index = 0

        # UART mappings for RPi5 — ESP32 usually on USB, SIM7600X on GPIO UART
        robot_config.hardware.uart_device_map.setdefault("esp32", "/dev/ttyUSB0")
        robot_config.hardware.uart_device_map.setdefault("sim7600x", "/dev/ttyAMA1")

        # Microphone hints — user can override via .env
        robot_config.hardware.wake_word_microphone_name = (
            robot_config.hardware.wake_word_microphone_name
            or os.getenv("RPI_WAKE_MIC", None)
        )
        robot_config.hardware.speech_microphone_name = (
            robot_config.hardware.speech_microphone_name
            or os.getenv("RPI_DIALOG_MIC", None)
        )

        logging.info("Applied Raspberry Pi 5 overrides (hailo=%s)", robot_config.system.enable_hailo)
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("Failed to apply RPi5 overrides: %s", exc)
