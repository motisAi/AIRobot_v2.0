"""Mic resolution by stable identity (architecture §4): by-id -> ALSA card -> PortAudio
'(hw:N,' entry; name fallback; None when missing (never a default); two-mic invariant.
Runs without hardware (temp by-id dir + fake PortAudio table).

Run:  venv/bin/python tests/test_mic_resolution.py   (pytest-compatible)
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from parts_used.audio_devices import find_input_index, resolve_alsa_card, same_device  # noqa: E402


class FakePA:
    def __init__(self, names):
        self._n = names

    def get_device_count(self):
        return len(self._n)

    def get_device_info_by_index(self, i):
        return {"name": self._n[i], "maxInputChannels": 1, "defaultSampleRate": 44100}


TABLE = ["USB PnP Sound Device: Audio (hw:0,0)", "Auto Focus Camera: USB Audio (hw:3,0)",
         "sysdefault", "pulse", "default"]


def _by_id_dir():
    d = tempfile.mkdtemp()
    os.symlink("../controlC0", os.path.join(d, "usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device-00"))
    os.symlink("../controlC3", os.path.join(d, "usb-Signo_Camera_WB-400_Auto_Focus_Camera_200901010001-02"))
    return d


def test_by_id_resolution():
    d = _by_id_dir(); pa = FakePA(TABLE)
    assert resolve_alsa_card("usb-C-Media_*", d) == 0
    assert resolve_alsa_card("usb-Signo_*", d) == 3
    assert find_input_index("usb-C-Media_*", "USB PnP Sound Device", by_id_root=d, pa=pa) == 0
    assert find_input_index("usb-Signo_*", "Auto Focus Camera", by_id_root=d, pa=pa) == 1


def test_missing_returns_none_never_default():
    d = _by_id_dir(); pa = FakePA(["sysdefault", "pulse", "default"])
    assert find_input_index("usb-Nope_*", "Nonexistent Mic", by_id_root=d, pa=pa) is None


def test_name_fallback_when_no_by_id():
    d = tempfile.mkdtemp(); pa = FakePA(TABLE)
    assert find_input_index(None, "Auto Focus Camera", by_id_root=d, pa=pa) == 1


def test_two_mic_invariant():
    pa = FakePA(TABLE)
    assert not same_device(0, 1, pa)          # different physical devices
    assert same_device(1, 1, pa)
    assert same_device(1, 1, pa) and not same_device(None, 1, pa)


if __name__ == "__main__":
    test_by_id_resolution()
    test_missing_returns_none_never_default()
    test_name_fallback_when_no_by_id()
    test_two_mic_invariant()
    print("OK: mic resolution by stable id, no-default, two-mic invariant")
