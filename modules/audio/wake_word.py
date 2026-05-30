"""Wake word detection module tailored for the "Gonzo" keyword.

The detector prefers Picovoice Porcupine when a valid ``.ppn`` model and access
key are available. When Porcupine cannot be used the module falls back to a
lightweight RMS/VAD gate so the robot still reacts, albeit with reduced
accuracy. All detections stay on-device per the privacy requirements.
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
        self.thread: Optional[threading.Thread] = None
        self.running = False

        self.detector_mode = "porcupine" if self._porcupine_ready() else "energy"
        self.porcupine = None
        self.energy_threshold = 800.0
        self.energy_window = deque(maxlen=20)
        self.vad = webrtcvad.Vad(2) if VAD_AVAILABLE else None

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
        """Temporarily pause detection while the robot converses."""

        self.logger.debug("Pausing wake-word listener (%s)", reason)
        self.listen_event.clear()

    def resume_listening(self) -> None:
        """Resume passive listening."""

        self.listen_event.set()
        self.logger.debug("Wake-word listener resumed")

    # ------------------------------------------------------------------
    # Detection loops
    # ------------------------------------------------------------------
    def _listen_loop(self) -> None:
        """Dispatch to the active detection strategy."""

        if self.detector_mode == "porcupine":
            self._porcupine_loop()
        else:
            self._energy_loop()

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
        audio = pyaudio.PyAudio()
        stream = None
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
                break
            except Exception:
                continue
        if stream is None:
            self.logger.error("Unable to open microphone for wake-word detection at any sample rate")
            return

        self.logger.warning("Energy-based wake-word detector active (higher false positives)")

        debug_counter = 0
        try:
            while not self.shutdown_event.is_set():
                if not self.listen_event.is_set():
                    time.sleep(0.05)
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
                    self.logger.debug(f"Energy: rms={rms:.0f}, threshold={dynamic_threshold:.0f}, window_mean={np.mean(self.energy_window):.0f}")

                if rms > dynamic_threshold:
                    if self.vad and not self._contains_voice(frame):
                        continue
                    self._emit_detection(confidence=min(rms / dynamic_threshold, 1.0), method="energy")
                    time.sleep(1.0)
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

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

        if pyaudio is None or not name_hint:
            return None

        audio = pyaudio.PyAudio()
        try:
            for idx in range(audio.get_device_count()):
                info = audio.get_device_info_by_index(idx)
                device_name = info.get('name', '').lower()
                if name_hint.lower() in device_name:
                    return idx
        except Exception as exc:
            self.logger.warning(f"Failed to enumerate audio devices: {exc}")
        finally:
            audio.terminate()

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
