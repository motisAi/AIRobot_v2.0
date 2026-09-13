"""AudioArbiter — one owner of the single speaker at a time.

The Pi has one audio output (HDMI/USB), which only plays one stream at a time.
This arbiter centralizes "who owns the speaker": high-priority audio (Stella's
speech, alert chimes) preempts low-priority *duckable* audio (music), which
yields the device for the duration and resumes afterward.

Any number of speech/alert sources can use `with arbiter.speak():` — a lock
serializes them so two never collide on the device — and any number of duckable
sources (currently the music player) register via `add_duckable()`.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager

logger = logging.getLogger("AudioArbiter")


class AudioArbiter:
    def __init__(self):
        self._lock = threading.RLock()
        self._duckables = []   # objects with is_playing()/pause_output()/resume_output()

    def add_duckable(self, source):
        """Register a low-priority audio source that yields the speaker on demand."""
        with self._lock:
            if source is not None and source not in self._duckables:
                self._duckables.append(source)

    def _pause_duckables(self):
        paused = []
        for s in list(self._duckables):
            try:
                if s.is_playing():
                    s.pause_output()
                    paused.append(s)
            except Exception as exc:
                logger.debug("pause duckable failed: %s", exc)
        return paused

    def _resume(self, paused):
        for s in paused:
            try:
                s.resume_output()
            except Exception as exc:
                logger.debug("resume duckable failed: %s", exc)

    @contextmanager
    def speak(self):
        """Exclusive speaker access for one spoken line / alert. Ducks (frees) any
        playing music for the duration, then restores it. Serializes concurrent
        speakers so they never fight over the device."""
        self._lock.acquire()
        paused = self._pause_duckables()
        try:
            yield
        finally:
            self._resume(paused)
            self._lock.release()
