"""Wake word detection module tailored for the "Gonzo" keyword.

Engine preference (config: model.wake_word_engine = auto):
  1. **Vosk** — free, offline, small-model speech recognition that detects the
     actual spoken word "gonzo". Recommended; no signup, no cost.
  2. **Porcupine** — Picovoice keyword spotting (requires a registered key).
  3. **Energy** — a lightweight RMS/VAD loudness gate that reacts to any loud
     sound. Zero setup, but not word-specific. Used only as a last resort.

All detection stays on-device per the privacy requirements.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional
from collections import deque

import numpy as np

try:
    import pyaudio
except ImportError:  # pragma: no cover - optional dependency
    pyaudio = None

try:
    import pvporcupine
    import pvrecorder
    PORCUPINE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PORCUPINE_AVAILABLE = False

try:
    import json as _json
    from vosk import Model as VoskModel, KaldiRecognizer, SetLogLevel as _VoskSetLogLevel
    _VoskSetLogLevel(-1)  # silence vosk's verbose logging
    VOSK_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    VOSK_AVAILABLE = False

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    VAD_AVAILABLE = False

from config.settings import model_config, hardware_config
from core.robot_brain import RobotEvent


class WakeWordModule:
    """Continuously listens for the configured wake word ("gonzo")."""

    def __init__(self, brain=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.brain = brain

        self.keyword = model_config.wake_word.lower().strip() or "gonzo"
        self.sensitivity = float(model_config.wake_word_sensitivity)
        self.access_key = model_config.picovoice_access_key
        self.model_path = Path(model_config.wake_word_model_path)

        self.device_index = self._resolve_microphone_index(
            name_hint=hardware_config.wake_word_microphone_name,
            explicit_index=hardware_config.wake_word_device_index,
        )

        self.listen_event = threading.Event()
        self.listen_event.set()
        self.shutdown_event = threading.Event()
        self._stream_closed_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.running = False

        # Wake-word engine selection (auto|vosk|porcupine|energy).
        self.engine_pref = getattr(model_config, "wake_word_engine", "auto").lower()
        self.vosk_model_path = Path(getattr(model_config, "vosk_model_path", ""))
        self.wake_phrases = [p.lower() for p in
                             getattr(model_config, "wake_word_phrases", [self.keyword])]
        # Fuzzy prefix for near-miss transcriptions, DERIVED from the wake word
        # so it follows config changes (e.g. "gonzo"->"gonz", "robby"->"robb").
        kw = self.keyword.split()[-1] if self.keyword else "gonzo"
        self._wake_prefix = kw[:4] if len(kw) >= 4 else kw
        self.detector_mode = self._select_mode()
        self.porcupine = None
        self._vosk_model = None
        self.energy_threshold = 300.0
        self.energy_window = deque(maxlen=50)
        self.vad = webrtcvad.Vad(2) if VAD_AVAILABLE else None

    def _select_mode(self) -> str:
        """Pick the wake-word engine based on config and availability."""
        pref = self.engine_pref
        if pref == "energy":
            return "energy"
        if pref == "porcupine":
            return "porcupine" if self._porcupine_ready() else "energy"
        if pref == "vosk":
            return "vosk" if self._vosk_ready() else "energy"
        # auto: prefer real-word offline (vosk) -> porcupine -> energy
        if self._vosk_ready():
            return "vosk"
        if self._porcupine_ready():
            return "porcupine"
        return "energy"

    def _vosk_ready(self) -> bool:
        return VOSK_AVAILABLE and self.vosk_model_path.exists()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Launch the listener thread."""

        if self.running:
            return

        if pyaudio is None and not PORCUPINE_AVAILABLE:
            self.logger.error("Neither PyAudio nor Porcupine are available; wake-word detection disabled")
            return

        if self.detector_mode == "porcupine":
            try:
                self.porcupine = self._build_porcupine_instance()
            except Exception as exc:  # pragma: no cover - external dependency
                self.logger.warning("Falling back to energy detector: %s", exc)
                self.detector_mode = "energy"

        if self.detector_mode == "vosk":
            try:
                self.logger.info("Loading Vosk wake-word model from %s", self.vosk_model_path)
                self._vosk_model = VoskModel(str(self.vosk_model_path))
            except Exception as exc:
                self.logger.warning("Vosk load failed, falling back to energy: %s", exc)
                self.detector_mode = "energy"

        self.shutdown_event.clear()
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        self.running = True
        self.logger.info("Wake-word detector running in %s mode", self.detector_mode)

    def stop(self) -> None:
        """Gracefully stop the listener thread."""

        self.shutdown_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.running = False

        if self.porcupine:
            try:
                self.porcupine.delete()
            except Exception:
                pass
            self.porcupine = None

    def pause_listening(self, reason: str = "dialogue") -> None:
        """Temporarily pause detection and release the microphone."""

        self.logger.info("Pausing wake-word listener (%s) — releasing mic", reason)
        self.listen_event.clear()
        # Wait for the stream to actually close before returning
        if not self._stream_closed_event.wait(timeout=2.0):
            self.logger.warning("Timed out waiting for wake-word stream to close")

    def resume_listening(self) -> None:
        """Resume passive listening (stream will reopen in the loop)."""

        self._stream_closed_event.clear()
        self.listen_event.set()
        self.logger.info("Wake-word listener resumed")

    # ------------------------------------------------------------------
    # Detection loops
    # ------------------------------------------------------------------
    def _listen_loop(self) -> None:
        """Dispatch to the active detection strategy."""

        if self.detector_mode == "porcupine":
            self._porcupine_loop()
        elif self.detector_mode == "vosk":
            self._vosk_loop()
        else:
            self._energy_loop()

    def _vosk_loop(self) -> None:
        """Free, offline, word-specific wake detection using Vosk.

        Streams the wake microphone through a small Vosk model and fires when a
        configured wake phrase (or a fuzzy 'gonz*' token) is recognised.
        """
        if pyaudio is None or self._vosk_model is None:
            self.logger.error("Vosk detector not ready; aborting")
            return

        TARGET_RATE = 16000
        audio = None
        stream = None
        actual_rate = TARGET_RATE

        def _open():
            nonlocal audio, stream, actual_rate
            from modules.hardware.audio_pa import get_pa
            audio = get_pa()          # shared instance — do NOT terminate it
            if audio is None:
                return False
            # 48k first: the camera mic doesn't support 16k and would spam
            # paInvalidSampleRate. We resample to 16k for Vosk anyway.
            for rate in (48000, 44100, 16000):
                try:
                    stream = audio.open(
                        format=pyaudio.paInt16, channels=1, rate=rate, input=True,
                        frames_per_buffer=4096, input_device_index=self.device_index,
                    )
                    actual_rate = rate
                    self.logger.info("Vosk wake mic open at %d Hz (device %s)",
                                     rate, self.device_index)
                    return True
                except Exception:
                    continue
            audio = None  # shared instance — do not terminate
            self.logger.error("Could not open wake mic for Vosk")
            return False

        def _close():
            nonlocal audio, stream
            if stream:
                try:
                    stream.stop_stream(); stream.close()
                except Exception:
                    pass
                stream = None
            # Do NOT terminate the shared PyAudio instance — just drop our ref.
            audio = None
            self._stream_closed_event.set()

        if not _open():
            return

        rec = KaldiRecognizer(self._vosk_model, TARGET_RATE)
        self.logger.info("Vosk wake-word detector active for phrases: %s", self.wake_phrases)

        try:
            while not self.shutdown_event.is_set():
                if not self.listen_event.is_set():
                    if stream is not None:
                        _close()
                    time.sleep(0.05)
                    continue
                if stream is None:
                    if not _open():
                        time.sleep(1.0)
                        continue
                    rec = KaldiRecognizer(self._vosk_model, TARGET_RATE)

                try:
                    frame = stream.read(4096, exception_on_overflow=False)
                except Exception:
                    continue

                if actual_rate != TARGET_RATE:
                    frame = self._resample_to_16k(frame, actual_rate)

                text = ""
                if rec.AcceptWaveform(frame):
                    text = _json.loads(rec.Result()).get("text", "")
                else:
                    text = _json.loads(rec.PartialResult()).get("partial", "")

                if text and self._matches_wake(text):
                    self._emit_detection(confidence=0.9, method="vosk")
                    rec = KaldiRecognizer(self._vosk_model, TARGET_RATE)  # reset
                    time.sleep(1.2)
        finally:
            _close()

    def _matches_wake(self, text: str) -> bool:
        """True if recognised text contains a wake phrase or a 'gonz*' token."""
        t = text.lower().strip()
        if not t:
            return False
        if any(p in t for p in self.wake_phrases):
            return True
        return any(tok.startswith(self._wake_prefix) for tok in t.split())

    @staticmethod
    def _resample_to_16k(frame: bytes, src_rate: int) -> bytes:
        """Downsample an int16 mono frame to 16 kHz for Vosk."""
        samples = np.frombuffer(frame, dtype=np.int16)
        if samples.size == 0 or src_rate == 16000:
            return frame
        n_out = int(round(samples.size * 16000 / src_rate))
        if n_out <= 0:
            return frame
        x_old = np.linspace(0, 1, samples.size, endpoint=False)
        x_new = np.linspace(0, 1, n_out, endpoint=False)
        resampled = np.interp(x_new, x_old, samples).astype(np.int16)
        return resampled.tobytes()

    def _porcupine_loop(self) -> None:
        """Run Porcupine on the configured microphone."""

        if not PORCUPINE_AVAILABLE:
            self.logger.error("Porcupine requested but not installed")
            return

        recorder = pvrecorder.PvRecorder(
            device_index=self.device_index if self.device_index is not None else -1,
            frame_length=self.porcupine.frame_length,
        )

        try:
            recorder.start()
            while not self.shutdown_event.is_set():
                if not self.listen_event.is_set():
                    time.sleep(0.05)
                    continue

                pcm = recorder.read()
                result = self.porcupine.process(pcm)
                if result >= 0:
                    self._emit_detection(confidence=0.99, method="porcupine")
        except Exception as exc:
            self.logger.error(f"Porcupine loop crashed: {exc}")
        finally:
            recorder.stop()
            recorder.delete()

    def _energy_loop(self) -> None:
        """Fallback RMS/VAD detector used when Porcupine is unavailable."""

        if pyaudio is None:
            self.logger.error("PyAudio not installed; cannot run fallback detector")
            return

        mic_rate = hardware_config.microphone_rate
        self.logger.warning("Energy-based wake-word detector active (higher false positives)")

        debug_counter = 0
        audio = None
        stream = None

        def _open_stream():
            nonlocal audio, stream, mic_rate
            from modules.hardware.audio_pa import get_pa
            audio = get_pa()          # shared instance — do NOT terminate it
            if audio is None:
                return False
            for try_rate in [mic_rate, 48000, 44100, 22050, 16000]:
                try:
                    stream = audio.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=try_rate,
                        input=True,
                        frames_per_buffer=512,
                        input_device_index=self.device_index,
                    )
                    mic_rate = try_rate
                    self.logger.info(f"Opened wake-word mic at {try_rate} Hz (device {self.device_index})")
                    return True
                except Exception:
                    continue
            self.logger.error("Unable to open microphone for wake-word detection at any sample rate")
            audio = None
            return False

        def _close_stream():
            nonlocal audio, stream
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
                stream = None
            audio = None  # shared instance — do not terminate
            self.logger.info("Wake-word mic released")
            self._stream_closed_event.set()

        # Initial open
        if not _open_stream():
            return

        try:
            while not self.shutdown_event.is_set():
                # --- paused: close the stream and wait ---
                if not self.listen_event.is_set():
                    if stream is not None:
                        _close_stream()
                    time.sleep(0.05)
                    continue

                # --- resumed: reopen the stream if needed ---
                if stream is None:
                    if not _open_stream():
                        time.sleep(1.0)
                        continue

                try:
                    frame = stream.read(512, exception_on_overflow=False)
                except Exception:
                    continue

                rms = self._calculate_rms(frame)
                self.energy_window.append(rms)
                dynamic_threshold = max(np.mean(self.energy_window) * 2.5, self.energy_threshold)

                debug_counter += 1
                if debug_counter % 500 == 0:
                    self.logger.info(f"Energy: rms={rms:.0f}, threshold={dynamic_threshold:.0f}, window_mean={np.mean(self.energy_window):.0f}")

                if rms > dynamic_threshold:
                    if self.vad and not self._contains_voice(frame):
                        continue
                    self._emit_detection(confidence=min(rms / dynamic_threshold, 1.0), method="energy")
                    time.sleep(1.0)
        finally:
            _close_stream()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _emit_detection(self, confidence: float, method: str) -> None:
        """Notify the robot brain that the wake word was heard."""

        if not self.brain:
            return

        self.logger.info(f"Wake word detected (method={method}, confidence={confidence:.2f})")
        event = RobotEvent(
            type='wake_word_detected',
            source='wake_word',
            data={'confidence': confidence, 'method': method, 'keyword': self.keyword},
            priority=2,
        )
        try:
            self.brain.emit_event(event)
        except Exception as exc:
            self.logger.error(f"Failed to emit wake-word event: {exc}")

    def _porcupine_ready(self) -> bool:
        """Return True if Porcupine should be used."""

        if not PORCUPINE_AVAILABLE or not self.access_key:
            return False
        if self.model_path.is_file():
            return True
        # Allow Porcupine to leverage its bundled keyword list when no custom
        # .ppn file is available (it contains a "hey {keyword}" fallback).
        return True

    def _build_porcupine_instance(self):
        """Create a configured Porcupine instance."""

        if self.model_path.is_file():
            return pvporcupine.create(
                access_key=self.access_key,
                keyword_paths=[str(self.model_path)],
                sensitivities=[self.sensitivity],
            )
        return pvporcupine.create(
            access_key=self.access_key,
            keywords=[self.keyword],
            sensitivities=[self.sensitivity],
        )

    def _resolve_microphone_index(self, name_hint: Optional[str], explicit_index: Optional[int]) -> Optional[int]:
        """Return the ALSA/PortAudio device index that best matches ``name_hint``."""

        if explicit_index is not None:
            return explicit_index

        if not name_hint:
            return None

        from modules.hardware.audio_pa import get_pa
        audio = get_pa()          # shared instance — do NOT terminate it
        if audio is None:
            return None
        try:
            for idx in range(audio.get_device_count()):
                info = audio.get_device_info_by_index(idx)
                device_name = info.get('name', '').lower()
                if name_hint.lower() in device_name:
                    return idx
        except Exception as exc:
            self.logger.warning(f"Failed to enumerate audio devices: {exc}")

        return None

    @staticmethod
    def _calculate_rms(frame: bytes) -> float:
        """Compute the root-mean-square energy for an audio frame."""

        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float64)
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(samples))))

    def _contains_voice(self, frame: bytes) -> bool:
        """Use WebRTC VAD to determine if the frame contains speech."""

        if not self.vad:
            return True

        try:
            # WebRTC VAD only supports 8000, 16000, 32000, 48000 Hz
            # If our rate doesn't match, skip VAD and rely on RMS only
            rate = hardware_config.microphone_rate
            if rate not in (8000, 16000, 32000, 48000):
                return True
            return self.vad.is_speech(frame, rate)
        except Exception:
            return True
