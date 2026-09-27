"""MicStream owner-thread guard (architecture §12 rule 1; bug_028/bug_054).
Runs without hardware: a fake stream object stands in for PyAudio's.

Run:  venv/bin/python tests/test_audio_threading.py   (pytest-compatible)
"""
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from parts_used.audio_portaudio import MicStream, open_input  # noqa: E402


class FakeStream:
    def __init__(self):
        self.stopped = False
        self.closed = False

    def read(self, n, exception_on_overflow=False):
        return b"\x00" * (2 * n)

    def stop_stream(self):
        self.stopped = True

    def close(self):
        self.closed = True


def test_owner_can_close_and_reader_thread_cannot():
    fs = FakeStream()
    ms = MicStream(fs, device_index=0, rate=44100, label="t")
    assert len(ms.read(10)) == 20
    err = {}

    def other():
        try:
            ms.close()
        except RuntimeError as e:
            err["e"] = e

    t = threading.Thread(target=other)
    t.start(); t.join(2)
    assert "e" in err, "close() from a non-owner thread must raise"
    assert not fs.closed, "non-owner must not have closed the stream"
    ms.close()                       # owner thread
    assert fs.closed and ms.closed
    ms.close()                       # idempotent


def test_open_input_refuses_none_index():
    try:
        open_input(None, 44100, 1024)
    except ValueError:
        return
    raise AssertionError("open_input(None) must raise (never open ALSA 'default')")


if __name__ == "__main__":
    test_owner_can_close_and_reader_thread_cannot()
    test_open_input_refuses_none_index()
    print("OK: MicStream owner guard + no-default rule")
