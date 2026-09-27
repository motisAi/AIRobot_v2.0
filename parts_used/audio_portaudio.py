"""Process-wide shared PyAudio instance + the single-owner MicStream wrapper.

Rules (docs/architecture/stella-architecture.md §4 and §12; bug_028, bug_054, bug_055):
- ONE ``pyaudio.PyAudio()`` per process (``get_pa()``) — repeated ``Pa_Initialize``
  calls were the source of the ``defaultInputDevice < deviceCount`` core dumps.
  Never call ``.terminate()`` on it; it lives for the life of the process. Its
  input-device table is logged ONCE at init so the journal shows what the app
  resolved against (PortAudio freezes that table at init).
- A PortAudio stream is opened, read, stopped and closed by exactly ONE thread.
  ``MicStream`` records its owner thread and raises if stop/close is attempted
  from any other thread (closing cross-thread corrupts the heap -> SIGABRT).
- Never open with ``input_device_index=None`` (that is ALSA 'default', i.e. the
  pulse plugin = whichever source PulseAudio favours today). ``open_input()``
  refuses it; callers resolve the device first (parts_used.audio_devices).
"""

from __future__ import annotations

import logging
import threading

try:
    import pyaudio
except ImportError:  # pragma: no cover
    pyaudio = None

_pa = None
_lock = threading.Lock()
_log = logging.getLogger("AudioPA")


def get_pa():
    """Return the shared PyAudio instance (or None if pyaudio is missing)."""
    global _pa
    if pyaudio is None:
        return None
    with _lock:
        if _pa is None:
            _pa = pyaudio.PyAudio()
            try:
                rows = []
                for i in range(_pa.get_device_count()):
                    d = _pa.get_device_info_by_index(i)
                    if int(d.get("maxInputChannels", 0) or 0) > 0:
                        rows.append(f"[{i}] {d.get('name')} @{int(d.get('defaultSampleRate', 0) or 0)}")
                _log.info("PortAudio input device table (frozen at init): %s",
                          "; ".join(rows) if rows else "none")
            except Exception as exc:  # pragma: no cover
                _log.warning("Could not list PortAudio devices: %s", exc)
        return _pa


def open_lock() -> threading.Lock:
    """A shared lock callers can use to serialise stream opening if desired."""
    return _lock


def device_name(index, pa=None) -> str:
    """PortAudio device name for an index ('' if unknown)."""
    pa = pa or get_pa()
    if pa is None or index is None:
        return ""
    try:
        return str(pa.get_device_info_by_index(index).get("name", ""))
    except Exception:
        return ""


class MicStream:
    """A PortAudio input stream owned by the thread that opened it.

    read() passes through. stop_stream()/close() raise RuntimeError from any
    other thread — that is the crash class of bug_028/bug_054. close() is
    idempotent and stops the stream first.
    """

    def __init__(self, stream, device_index, rate: int, label: str = ""):
        self._s = stream
        self.device_index = device_index
        self.rate = rate
        self.label = label
        self._owner = threading.get_ident()
        self._closed = False

    def _check_owner(self, op: str) -> None:
        if threading.get_ident() != self._owner:
            raise RuntimeError(
                f"PortAudio stream {self.label!r}: {op} from a non-owner thread — "
                "exactly one thread owns a stream (bug_028/bug_054)")

    def read(self, n: int, exception_on_overflow: bool = False):
        return self._s.read(n, exception_on_overflow=exception_on_overflow)

    def stop_stream(self) -> None:
        self._check_owner("stop_stream")
        try:
            self._s.stop_stream()
        except Exception:
            pass

    def close(self) -> None:
        self._check_owner("close")
        if self._closed:
            return
        self._closed = True
        try:
            self._s.stop_stream()
        except Exception:
            pass
        self._s.close()

    @property
    def closed(self) -> bool:
        return self._closed


def open_input(device_index, rate: int, frames_per_buffer: int,
               expect_hw=None, label: str = "") -> MicStream:
    """Open an input stream on a RESOLVED device index (never None) and wrap it.

    expect_hw: optional ALSA card number; the PortAudio name must contain
    '(hw:<N>,' or we refuse — guards against a stale/renumbered index.
    """
    if device_index is None:
        raise ValueError("refusing to open a mic with input_device_index=None "
                         "(ALSA 'default' = PulseAudio's pick) — resolve the device first")
    if pyaudio is None:
        raise RuntimeError("pyaudio is not available")
    pa = get_pa()
    if expect_hw is not None:
        nm = device_name(device_index, pa)
        if f"(hw:{expect_hw}," not in nm:
            raise RuntimeError(f"device {device_index} is {nm!r}, expected ALSA hw:{expect_hw}")
    s = pa.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True,
                frames_per_buffer=frames_per_buffer, input_device_index=device_index)
    return MicStream(s, device_index, rate, label)
