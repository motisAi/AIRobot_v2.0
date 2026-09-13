"""Shared Audio Device Manager
==============================
Prevents microphone and speaker conflicts by centralizing device
allocation.  Each module requests a device by role (wake_word, dialogue,
playback) and gets an exclusive index.  If two modules accidentally ask
for the same physical device the manager raises early instead of causing
a silent ALSA/PortAudio error at runtime.

Design:
 - Enumerate PortAudio devices once at init.
 - Map role -> device index via config hints or auto-detection.
 - Provide ``acquire(role)`` / ``release(role)`` for exclusive access.
 - Thread-safe.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Tuple

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:  # pragma: no cover
    pyaudio = None
    PYAUDIO_AVAILABLE = False

from config.settings import hardware_config


class AudioDevice:
    """Metadata for a single audio input or output."""

    __slots__ = ("index", "name", "max_input_channels", "max_output_channels",
                 "default_sample_rate")

    def __init__(self, index: int, info: dict):
        self.index = index
        self.name: str = info.get("name", f"device_{index}")
        self.max_input_channels: int = int(info.get("maxInputChannels", 0))
        self.max_output_channels: int = int(info.get("maxOutputChannels", 0))
        self.default_sample_rate: float = float(info.get("defaultSampleRate", 16000))

    @property
    def is_input(self) -> bool:
        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        return self.max_output_channels > 0

    def __repr__(self) -> str:
        direction = []
        if self.is_input:
            direction.append("IN")
        if self.is_output:
            direction.append("OUT")
        return f"AudioDevice({self.index}, '{self.name}', {'/'.join(direction)})"


class AudioManager:
    """Enumerates audio devices and hands out exclusive leases per role."""

    # Standard roles
    ROLE_WAKE_WORD = "wake_word"
    ROLE_DIALOGUE = "dialogue"
    ROLE_PLAYBACK = "playback"

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        self._devices: List[AudioDevice] = []
        self._input_devices: List[AudioDevice] = []
        self._output_devices: List[AudioDevice] = []

        # role -> (device_index, owner_name)
        self._leases: Dict[str, Tuple[int, str]] = {}
        self._lock = threading.Lock()

        self._enumerate_devices()
        self._log_device_map()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def acquire(self, role: str, owner: str,
                preferred_name: Optional[str] = None,
                preferred_index: Optional[int] = None) -> Optional[int]:
        """Acquire exclusive access to a microphone for *role*.

        Args:
            role: One of ROLE_WAKE_WORD, ROLE_DIALOGUE, ROLE_PLAYBACK.
            owner: Human-readable module name for logging.
            preferred_name: Substring to match against device names.
            preferred_index: Explicit device index (takes priority).

        Returns:
            Device index on success, None if no suitable device is available.
        """
        with self._lock:
            if role in self._leases:
                idx, prev_owner = self._leases[role]
                self.logger.warning("Role '%s' already acquired by '%s' (device %d)",
                                    role, prev_owner, idx)
                return idx

            # Determine pool
            if role == self.ROLE_PLAYBACK:
                pool = self._output_devices
            else:
                pool = self._input_devices

            device = self._resolve_device(pool, preferred_name, preferred_index)
            if device is None:
                self.logger.error("No audio device available for role '%s'", role)
                return None

            # Check for hardware conflict — same physical device used by another role
            for other_role, (other_idx, other_owner) in self._leases.items():
                if other_idx == device.index and other_role != role:
                    self.logger.warning(
                        "Device %d ('%s') already used by role '%s' (%s). "
                        "This WILL cause audio conflicts. Assign separate devices "
                        "in config/settings.py or .env.",
                        device.index, device.name, other_role, other_owner,
                    )

            self._leases[role] = (device.index, owner)
            self.logger.info("Audio device %d ('%s') acquired for role '%s' by '%s'",
                             device.index, device.name, role, owner)
            return device.index

    def release(self, role: str) -> None:
        """Release a previously acquired device."""
        with self._lock:
            removed = self._leases.pop(role, None)
            if removed:
                self.logger.info("Released audio device for role '%s'", role)

    def get_device_index(self, role: str) -> Optional[int]:
        """Return the device index for an already-acquired role."""
        with self._lock:
            entry = self._leases.get(role)
            return entry[0] if entry else None

    def list_input_devices(self) -> List[AudioDevice]:
        """Return all detected input devices."""
        return list(self._input_devices)

    def list_output_devices(self) -> List[AudioDevice]:
        """Return all detected output devices."""
        return list(self._output_devices)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _enumerate_devices(self) -> None:
        """Scan PortAudio for available devices."""
        if not PYAUDIO_AVAILABLE:
            self.logger.warning("PyAudio not installed — audio device enumeration skipped")
            return

        pa = pyaudio.PyAudio()
        try:
            count = pa.get_device_count()
            for idx in range(count):
                try:
                    info = pa.get_device_info_by_index(idx)
                    dev = AudioDevice(idx, info)
                    self._devices.append(dev)
                    if dev.is_input:
                        self._input_devices.append(dev)
                    if dev.is_output:
                        self._output_devices.append(dev)
                except Exception:
                    continue
        finally:
            pa.terminate()

    def _resolve_device(self, pool: List[AudioDevice],
                        name_hint: Optional[str],
                        explicit_index: Optional[int]) -> Optional[AudioDevice]:
        """Pick a device from *pool* using the provided hints."""
        if not pool:
            return None

        # Explicit index wins
        if explicit_index is not None:
            for dev in pool:
                if dev.index == explicit_index:
                    return dev

        # Name substring match
        if name_hint:
            hint_lower = name_hint.lower()
            for dev in pool:
                if hint_lower in dev.name.lower():
                    return dev

        # Fallback: first available in pool
        return pool[0]

    def _log_device_map(self) -> None:
        """Log the discovered audio topology for debugging."""
        if not self._devices:
            self.logger.warning("No audio devices detected")
            return
        self.logger.info("Detected %d audio device(s):", len(self._devices))
        for dev in self._devices:
            self.logger.info("  %s", dev)
