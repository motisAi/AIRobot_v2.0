"""Process-wide shared PyAudio instance.

Repeatedly constructing ``pyaudio.PyAudio()`` (which calls PortAudio's
``Pa_Initialize``) is what triggers the intermittent
``InitializeHostApis: Assertion 'defaultInputDevice < deviceCount' failed``
core dump on the Pi — especially when the wake-word and speech mics churn
concurrently. Creating ONE instance for the whole app and reusing it removes
those repeated initialisations and the crash.

Use ``get_pa()`` everywhere instead of ``pyaudio.PyAudio()``, and never call
``.terminate()`` on it — it lives for the life of the process.
"""

from __future__ import annotations

import threading

try:
    import pyaudio
except ImportError:  # pragma: no cover
    pyaudio = None

_pa = None
_lock = threading.Lock()


def get_pa():
    """Return the shared PyAudio instance (or None if pyaudio is missing)."""
    global _pa
    if pyaudio is None:
        return None
    with _lock:
        if _pa is None:
            _pa = pyaudio.PyAudio()
        return _pa


def open_lock() -> threading.Lock:
    """A shared lock callers can use to serialise stream opening if desired."""
    return _lock
