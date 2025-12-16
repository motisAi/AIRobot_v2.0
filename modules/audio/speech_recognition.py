"""Speech recognition module that prefers Whisper but falls back to
``speech_recognition`` when GPU support is unavailable.

The implementation records audio on demand (when the wake word fires) so it can
run continuously on resource-constrained Jetson hardware without wasting CPU
cycles.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Optional

try:
    import pyaudio
except ImportError:  # pragma: no cover - optional dependency
    pyaudio = None

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    whisper = None
    WHISPER_AVAILABLE = False

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    sr = None
    SR_AVAILABLE = False

from config.settings import model_config, hardware_config
from core.robot_brain import RobotEvent


class SpeechRecognitionModule:
    """Handles voice capture and transcription."""

    def __init__(self, brain=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.brain = brain

        self.sample_rate = hardware_config.microphone_rate
        self.chunk_size = hardware_config.microphone_chunk
        self.timeout = hardware_config.microphone_timeout
        self.phrase_limit = hardware_config.microphone_phrase_time_limit

        self.device_index = hardware_config.speech_device_index
        self.device_name = hardware_config.speech_microphone_name

        self.running = False
        self.listener_lock = threading.Lock()
        self.active_listener: Optional[threading.Thread] = None
        self._stop_recording = threading.Event()

        self.whisper_model_name = model_config.whisper_model
        self.whisper_language = model_config.whisper_language
        self.whisper_device = model_config.whisper_device
        self.whisper_instance = None

        self.recognizer = sr.Recognizer() if SR_AVAILABLE else None

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Flag the service as available. Heavy models load lazily."""

        if self.running:
            return

        if pyaudio is None and not SR_AVAILABLE:
            self.logger.error("PyAudio is required for command capture. Install portaudio bindings.")
            return

        self.running = True
        self.logger.info("Speech recognition module ready (Whisper=%s)", WHISPER_AVAILABLE)

    def stop(self) -> None:
        """Stop active listeners."""

        self.running = False
        self._stop_recording.set()
        if self.active_listener and self.active_listener.is_alive():
            self.active_listener.join(timeout=2.0)
        self.active_listener = None

    # ------------------------------------------------------------------
    def listen_for_command(self, timeout: Optional[float] = None) -> bool:
        """Capture audio for a single command.

        Args:
            timeout: Maximum seconds to wait for the utterance.
        Returns:
            bool: True when a new listener thread was started.
        """

        if not self.running:
            self.logger.warning("Speech recognition module not running")
            return False

        with self.listener_lock:
            if self.active_listener and self.active_listener.is_alive():
                self.logger.debug("Speech recognizer already listening")
                return False

            self._stop_recording.clear()
            self.active_listener = threading.Thread(
                target=self._capture_and_transcribe,
                args=(timeout or self.phrase_limit,),
                daemon=True,
            )
            self.active_listener.start()
            return True

    # ------------------------------------------------------------------
    def _capture_and_transcribe(self, duration: float) -> None:
        """Record audio and dispatch it to the selected recognizer."""

        audio_path = None
        try:
            audio_path = self._record_audio(duration)
            if not audio_path:
                self._emit_failure("audio_unavailable")
                return

            text = self._transcribe(audio_path)
            if text:
                self._emit_success(text)
            else:
                self._emit_failure("empty_transcript")
        except Exception as exc:
            self.logger.error(f"Speech capture failed: {exc}")
            self._emit_failure("exception")
        finally:
            if audio_path and Path(audio_path).exists():
                try:
                    Path(audio_path).unlink()
                except Exception:
                    pass
            with self.listener_lock:
                self.active_listener = None

    def _record_audio(self, duration: float) -> Optional[str]:
        """Record PCM audio from the configured microphone."""

        if pyaudio is None:
            self.logger.error("PyAudio missing; cannot capture audio")
            return None

        device_index = self._resolve_microphone_index()
        audio = pyaudio.PyAudio()
        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=device_index,
            )
        except Exception as exc:
            self.logger.error(f"Failed to open microphone: {exc}")
            return None

        frames = []
        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                if self._stop_recording.is_set():
                    break
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    frames.append(data)
                except Exception as exc:
                    self.logger.warning(f"Microphone read error: {exc}")
                    break
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

        if not frames:
            return None

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_name = tmp_file.name
        tmp_file.close()
        with wave.open(tmp_name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))

        return tmp_name

    def _transcribe(self, audio_path: str) -> Optional[str]:
        """Convert the recorded WAV file to text."""

        if WHISPER_AVAILABLE:
            try:
                self._ensure_whisper_model()
                result = self.whisper_instance.transcribe(
                    audio_path,
                    language=self.whisper_language,
                    fp16=False,
                )
                text = result.get('text', '').strip()
                return text or None
            except Exception as exc:
                self.logger.warning(f"Whisper transcription failed: {exc}")

        if SR_AVAILABLE and self.recognizer:
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
            try:
                text = self.recognizer.recognize_google(audio)
                return text.strip()
            except Exception as exc:
                self.logger.warning(f"SpeechRecognition fallback failed: {exc}")

        return None

    def _ensure_whisper_model(self) -> None:
        """Load the Whisper model on first use."""

        if not WHISPER_AVAILABLE or self.whisper_instance:
            return

        model_name = self.whisper_model_name or "tiny"
        try:
            self.whisper_instance = whisper.load_model(
                model_name,
                device=self.whisper_device,
                download_root=str(Path.home() / ".cache" / "robot_whisper"),
            )
        except Exception as exc:
            self.logger.warning(f"Unable to load Whisper model '{model_name}': {exc}")
            self.whisper_instance = None

    def _resolve_microphone_index(self) -> Optional[int]:
        """Resolve the microphone index using the configured hint."""

        if self.device_index is not None:
            return self.device_index

        if pyaudio is None or not self.device_name:
            return None

        audio = pyaudio.PyAudio()
        try:
            for idx in range(audio.get_device_count()):
                info = audio.get_device_info_by_index(idx)
                if self.device_name.lower() in info.get('name', '').lower():
                    return idx
        except Exception as exc:
            self.logger.warning(f"Could not enumerate microphones: {exc}")
        finally:
            audio.terminate()

        return None

    # ------------------------------------------------------------------
    def _emit_success(self, text: str) -> None:
        if not self.brain:
            return
        event = RobotEvent(
            type='speech_recognized',
            source='speech_recognition',
            data={'text': text},
            priority=3,
        )
        try:
            self.brain.emit_event(event)
        except Exception as exc:
            self.logger.error(f"Failed to emit speech_recognized event: {exc}")

    def _emit_failure(self, reason: str) -> None:
        if not self.brain:
            return
        event = RobotEvent(
            type='speech_listen_failed',
            source='speech_recognition',
            data={'reason': reason},
            priority=5,
        )
        try:
            self.brain.emit_event(event)
        except Exception as exc:
            self.logger.error(f"Failed to emit speech_listen_failed event: {exc}")
