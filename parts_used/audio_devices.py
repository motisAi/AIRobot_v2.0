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


# ---------------------------------------------------------------------------
# The ONE mic resolver (architecture §4). Stable identity first: /dev/snd/by-id
# -> controlC<N> (ALSA card) -> the PortAudio entry whose name carries "(hw:N,".
# Falls back to a name substring; NEVER returns a 'default' — None means "not
# found", and callers must log + back off instead of opening index None.
# `by_id_root` and `pa` are injectable so tests run without hardware.
# ---------------------------------------------------------------------------
import glob as _glob
import os as _os
import re as _re


def resolve_alsa_card(by_id_glob, by_id_root: str = "/dev/snd/by-id"):
    """ALSA card number for a /dev/snd/by-id glob (e.g. 'usb-C-Media_*'), or None."""
    if not by_id_glob:
        return None
    for p in sorted(_glob.glob(_os.path.join(by_id_root, by_id_glob))):
        try:
            target = _os.readlink(p)
        except OSError:
            continue
        mm = _re.search(r"controlC(\d+)", target)
        if mm:
            return int(mm.group(1))
    return None


def _input_table(pa):
    rows = []
    try:
        count = pa.get_device_count()
    except Exception:
        return rows
    for i in range(count):
        try:
            d = pa.get_device_info_by_index(i)
        except Exception:
            continue
        if int(d.get("maxInputChannels", 0) or 0) > 0:
            rows.append((i, str(d.get("name", ""))))
    return rows


def find_input_index(by_id_glob, name_hint, *, by_id_root: str = "/dev/snd/by-id", pa=None):
    """PortAudio input index for a mic identified by by-id glob (preferred) or name.

    Returns None when the mic cannot be found (caller: ERROR + backoff, never
    open 'default'). Name fallback is kept permissive on purpose so the current
    working setup cannot regress if a PortAudio name lacks the '(hw:N,' suffix.
    """
    if pa is None:
        from parts_used.audio_portaudio import get_pa
        pa = get_pa()
    if pa is None:
        return None
    inputs = _input_table(pa)
    card = resolve_alsa_card(by_id_glob, by_id_root)
    if card is not None:
        for i, nm in inputs:
            if f"(hw:{card}," in nm:
                return i
    if name_hint:
        h = str(name_hint).lower()
        for i, nm in inputs:
            if h in nm.lower():
                return i
    return None


def hw_card_of(index, pa=None):
    """ALSA card number parsed from a PortAudio device name, or None."""
    if index is None:
        return None
    if pa is None:
        from parts_used.audio_portaudio import get_pa
        pa = get_pa()
    if pa is None:
        return None
    try:
        nm = str(pa.get_device_info_by_index(index).get("name", ""))
    except Exception:
        return None
    mm = _re.search(r"\(hw:(\d+),", nm)
    return int(mm.group(1)) if mm else None


def same_device(idx_a, idx_b, pa=None) -> bool:
    """True if two resolved indices are the same physical mic (same index or same ALSA card)."""
    if idx_a is None or idx_b is None:
        return False
    if idx_a == idx_b:
        return True
    a, b = hw_card_of(idx_a, pa), hw_card_of(idx_b, pa)
    return a is not None and a == b
