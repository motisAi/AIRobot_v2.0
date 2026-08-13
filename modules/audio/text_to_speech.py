"""
Text-to-Speech Module
=====================
Converts text to speech using Coqui TTS (offline) or pyttsx3 as fallback.
Optimized for Raspberry Pi with caching and queue management.

"""

import time
import threading
import queue
import logging
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
import tempfile
import os

# Audio playback
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import pyaudio
    import wave
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

# TTS engines
try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False
    print("Warning: Coqui TTS not installed")

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("Warning: pyttsx3 not installed")

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import config, model_config, hardware_config, behavior_config


class TextToSpeechModule:
    """
    Text-to-Speech system using Coqui TTS or pyttsx3.
    Includes caching, queue management, and voice customization.
    """
    
    def __init__(self, brain=None):
        """
        Initialize TTS module
        
        Args:
            brain: Reference to robot brain
        """
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Text-to-Speech Module")
        
        # Brain reference
        self.brain = brain
        
        # Configuration
        self.tts_model = model_config.tts_model if hasattr(model_config, 'tts_model') else None
        # Language comes from the single behavior.language switch (en | he).
        self.language = (getattr(behavior_config, 'language', None)
                         or getattr(model_config, 'tts_language', 'en') or 'en').lower()
        self.speed = model_config.tts_speed if hasattr(model_config, 'tts_speed') else 1.0
        # Hebrew (and other non-English) can't use the English Piper voice; route
        # it to a language-aware path (gTTS -> espeak-ng fallback).
        self._nonlatin = self.language.startswith('he')

        # Audio OUTPUT routing — pin playback to the configured device (e.g. the
        # HDMI monitor now, a USB speaker later) so replies are actually heard.
        self.alsa_device = self._resolve_output_device()
        if self.alsa_device:
            self.logger.info("TTS audio output pinned to ALSA device '%s'", self.alsa_device)
        
        # State
        self.running = False
        self.speaking = False
        self.paused = False
        
        # TTS engine
        self.tts_engine = None
        self.engine_type = None
        
        # Speech queue
        self.speech_queue = queue.Queue()
        self.priority_queue = queue.PriorityQueue()
        
        # Cache directory
        self.cache_dir = Path(config.PROJECT_ROOT) / "data" / "tts_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index = {}
        self._load_cache_index()
        
        # Threading
        self.speech_thread = None
        
        # Audio player
        self.audio_player = None
        
        # Voice settings
        self.voice_settings = {
            'pitch': 1.0,
            'volume': 0.8,
            'rate': 150  # Words per minute
        }
        
        # Statistics
        self.speech_count = 0
        self.cache_hits = 0
        self.total_characters = 0
        
        # Initialize TTS engine
        self._initialize_engine()
        
        # Initialize audio player
        self._initialize_audio()
    
    def _initialize_engine(self):
        """Initialize TTS engine based on config + availability"""

        preferred = getattr(model_config, 'tts_engine', 'piper').lower()

        # Piper — neural, offline, natural voice (recommended default).
        if preferred == 'piper' and self._init_piper():
            self.engine_type = 'piper'
            self.logger.info("Using Piper TTS engine (%s)", Path(self.piper_voice).name)
            return

        # Coqui TTS (heavy; only if explicitly available/requested)
        if COQUI_AVAILABLE and preferred == 'coqui':
            try:
                self._init_coqui_tts()
                self.engine_type = 'coqui'
                self.logger.info("Using Coqui TTS engine")
                return
            except Exception as e:
                self.logger.warning(f"Failed to initialize Coqui TTS: {e}")

        # Fallback to pyttsx3
        if PYTTSX3_AVAILABLE:
            try:
                self._init_pyttsx3()
                self.engine_type = 'pyttsx3'
                self.logger.info("Using pyttsx3 TTS engine")
                return
            except Exception as e:
                self.logger.warning(f"Failed to initialize pyttsx3: {e}")
        
        # Last resort: espeak command line
        self._init_espeak()
        self.engine_type = 'espeak'
        self.logger.info("Using espeak TTS engine")
    
    def _init_piper(self) -> bool:
        """Locate the Piper binary + voice model. Returns True if usable."""
        root = Path(config.PROJECT_ROOT)

        def _resolve(p):
            p = Path(p)
            return p if p.is_absolute() else (root / p)

        self.piper_binary = str(_resolve(getattr(model_config, 'piper_binary',
                                'data/models/piper/piper/piper')))
        self.piper_voice = str(_resolve(getattr(model_config, 'piper_voice',
                               'data/models/piper/en_US-amy-medium.onnx')))
        self.piper_length_scale = float(getattr(model_config, 'piper_length_scale', 1.0))
        self.piper_pitch = float(getattr(model_config, 'piper_pitch', 0.0))
        self.piper_noise_scale = float(getattr(model_config, 'piper_noise_scale', 0.667))

        if not Path(self.piper_binary).exists():
            self.logger.warning("Piper binary not found at %s", self.piper_binary)
            return False
        if not Path(self.piper_voice).exists():
            self.logger.warning("Piper voice not found at %s", self.piper_voice)
            return False
        return True

    def _init_coqui_tts(self):
        """Initialize Coqui TTS"""
        
        # List available models
        available_models = TTS.list_models()
        
        # Select model based on configuration or auto-select
        if self.tts_model and self.tts_model in available_models:
            model_name = self.tts_model
        else:
            # Auto-select lightweight model for Raspberry Pi
            lightweight_models = [
                "tts_models/en/ljspeech/glow-tts",
                "tts_models/en/ljspeech/speedy-speech",
                "tts_models/en/jenny/jenny"
            ]
            
            model_name = None
            for model in lightweight_models:
                if model in available_models:
                    model_name = model
                    break
            
            if not model_name and available_models:
                model_name = available_models[0]
        
        if not model_name:
            raise Exception("No TTS models available")
        
        self.logger.info(f"Loading TTS model: {model_name}")
        
        # Initialize TTS
        self.tts_engine = TTS(
            model_name=model_name,
            progress_bar=False,
            gpu=False  # CPU only for Raspberry Pi
        )
        
        self.tts_model = model_name
        
        # Get speaker if multi-speaker model
        if hasattr(self.tts_engine, 'speakers') and self.tts_engine.speakers:
            self.speakers = self.tts_engine.speakers
            self.current_speaker = self.speakers[0] if self.speakers else None
        else:
            self.speakers = []
            self.current_speaker = None
    
    def _init_pyttsx3(self):
        """Initialize pyttsx3 TTS"""
        
        self.tts_engine = pyttsx3.init()
        
        # Configure voice
        voices = self.tts_engine.getProperty('voices')
        
        # Try to find English voice
        for voice in voices:
            if 'english' in voice.id.lower() or self.language in voice.id.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
        
        # Set properties
        self.tts_engine.setProperty('rate', self.voice_settings['rate'])
        self.tts_engine.setProperty('volume', self.voice_settings['volume'])
    
    def _init_espeak(self):
        """Initialize espeak as last resort"""
        
        # Check if espeak is available
        import subprocess
        try:
            subprocess.run(['espeak', '--version'], 
                         capture_output=True, check=True)
            self.tts_engine = 'espeak'
        except:
            self.logger.error("No TTS engine available!")
            self.tts_engine = None
    
    def _initialize_audio(self):
        """Initialize audio playback system"""

        # If we have an explicit ALSA output device (e.g. the HDMI monitor),
        # play via `aplay -D` and do NOT initialise pygame — pygame/SDL would
        # open and HOLD the default audio device, causing 'Device or resource
        # busy' when aplay tries to use the same hardware.
        if self.alsa_device:
            self.audio_player = 'system'
            self.logger.info("Using aplay -D %s for audio playback", self.alsa_device)
            return

        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init(
                    frequency=22050,
                    size=-16,
                    channels=2,
                    buffer=512
                )
                self.audio_player = 'pygame'
                self.logger.info("Using pygame for audio playback")
            except:
                pass

        if not self.audio_player and PYAUDIO_AVAILABLE:
            self.audio_player = 'pyaudio'
            self.logger.info("Using pyaudio for audio playback")

        if not self.audio_player:
            self.audio_player = 'system'
            self.logger.info("Using system command for audio playback")
    
    def start(self):
        """Start TTS service"""
        
        if self.running:
            return
        
        self.running = True
        
        # Start speech processing thread
        self.speech_thread = threading.Thread(target=self._speech_loop)
        self.speech_thread.daemon = True
        self.speech_thread.start()
        
        self.logger.info("TTS service started")
    
    def stop(self):
        """Stop TTS service"""
        
        self.running = False
        
        # Clear queues
        while not self.speech_queue.empty():
            self.speech_queue.get()
        
        # Wait for thread
        if self.speech_thread:
            self.speech_thread.join(timeout=2.0)
        
        # Clean up pygame
        if PYGAME_AVAILABLE:
            pygame.mixer.quit()
        
        self.logger.info("TTS service stopped")
    
    def speak(self, text: str, priority: int = 5, wait: bool = False):
        """
        Speak text
        
        Args:
            text: Text to speak
            priority: Priority level (1=highest)
            wait: Wait for speech to complete
        """
        
        if not text:
            return
        
        # Clean text
        text = text.strip()
        
        # Add to queue
        if priority < 5:
            self.priority_queue.put((priority, text))
        else:
            self.speech_queue.put(text)
        
        self.logger.debug(f"Queued speech: '{text[:50]}...' (priority: {priority})")
        
        # Wait if requested
        if wait:
            while self.speaking or not self.speech_queue.empty():
                time.sleep(0.1)

    def speak_blocking(self, text: str) -> bool:
        """Synthesize and play inline, returning only when playback is done.

        Reliable for the conversation loop — no queue/worker-thread race, and
        the caller is guaranteed audio has finished before it listens again.
        Returns True if audio actually played.
        """
        if not text:
            return False
        text = text.strip()
        self.speaking = True
        try:
            audio_file = self._get_cached_audio(text)
            if audio_file and Path(audio_file).exists():
                self._play_audio(audio_file)
                return True
            audio_file = self._generate_audio(text)
            if audio_file and Path(audio_file).exists():
                self._play_audio(audio_file)   # play FIRST, then cache (rename)
                self._cache_audio(text, audio_file)
                return True
            self.logger.warning("TTS produced no audio for: %r", text[:60])
            return False
        except Exception as e:
            self.logger.error("speak_blocking failed: %s", e)
            return False
        finally:
            self.speaking = False
    
    def _speech_loop(self):
        """Main speech processing loop"""
        
        while self.running:
            try:
                # Check priority queue first
                if not self.priority_queue.empty():
                    _, text = self.priority_queue.get(timeout=0.1)
                    self._process_speech(text)
                
                # Then normal queue
                elif not self.speech_queue.empty():
                    text = self.speech_queue.get(timeout=0.1)
                    self._process_speech(text)
                
                else:
                    time.sleep(0.1)
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Speech loop error: {e}")
                time.sleep(1.0)
    
    def _process_speech(self, text: str):
        """
        Process and speak text
        
        Args:
            text: Text to speak
        """
        
        self.speaking = True
        self.total_characters += len(text)
        
        played = False
        try:
            # Check cache first
            audio_file = self._get_cached_audio(text)

            if audio_file and Path(audio_file).exists():
                self.cache_hits += 1
                self._play_audio(audio_file)
                played = True
            else:
                # Generate new audio, PLAY it, THEN cache it. (Caching renames
                # the temp file, so caching before playing would leave nothing
                # to play — the cause of intermittent 'no voice'.)
                audio_file = self._generate_audio(text)
                if audio_file and Path(audio_file).exists():
                    self._play_audio(audio_file)
                    played = True
                    self._cache_audio(text, audio_file)
                else:
                    self.logger.warning("TTS produced no audio for: %r", text[:60])

            # Emit event
            if played and self.brain:
                from core.robot_brain import RobotEvent
                self.brain.emit_event(RobotEvent(
                    type='speech_complete',
                    source='tts',
                    data={'text': text}
                ))

            self.speech_count += 1

        except Exception as e:
            self.logger.error(f"Speech processing error: {e}")

        finally:
            self.speaking = False
    
    def _generate_audio(self, text: str) -> Optional[str]:
        """
        Generate audio from text
        
        Args:
            text: Text to convert
            
        Returns:
            Path to audio file or None
        """
        
        output_file = tempfile.NamedTemporaryFile(
            suffix='.wav', 
            delete=False,
            dir=self.cache_dir
        ).name
        
        try:
            # Hebrew / non-Latin: the English Piper voice can't pronounce it, so
            # use a language-aware synth regardless of the configured engine.
            if self._nonlatin:
                if self._generate_nonlatin(text, output_file):
                    return output_file
                return None

            if self.engine_type == 'piper':
                # Neural offline voice: pipe text -> piper -> wav
                import subprocess
                proc = subprocess.run(
                    [self.piper_binary,
                     '--model', self.piper_voice,
                     '--length_scale', str(self.piper_length_scale),
                     '--noise_scale', str(self.piper_noise_scale),
                     '--output_file', output_file],
                    input=text.encode('utf-8'),
                    capture_output=True, timeout=30,
                )
                if proc.returncode != 0 or not Path(output_file).exists():
                    self.logger.error("Piper synth failed: %s",
                                      proc.stderr.decode('utf-8', 'ignore')[:200])
                    return None
                # Optional pitch shift via sox (semitones -> cents).
                if abs(self.piper_pitch) > 0.01:
                    try:
                        shifted = output_file + '.pitch.wav'
                        r = subprocess.run(
                            ['sox', output_file, shifted, 'pitch',
                             str(int(self.piper_pitch * 100))],
                            capture_output=True, timeout=15)
                        if r.returncode == 0 and Path(shifted).exists():
                            Path(shifted).replace(output_file)
                    except Exception as exc:
                        self.logger.debug("sox pitch shift skipped: %s", exc)

            elif self.engine_type == 'coqui':
                # Use Coqui TTS
                if self.current_speaker:
                    self.tts_engine.tts_to_file(
                        text=text,
                        file_path=output_file,
                        speaker=self.current_speaker,
                        language=self.language
                    )
                else:
                    self.tts_engine.tts_to_file(
                        text=text,
                        file_path=output_file
                    )
                
            elif self.engine_type == 'pyttsx3':
                # Use pyttsx3
                self.tts_engine.save_to_file(text, output_file)
                self.tts_engine.runAndWait()
                
            elif self.engine_type == 'espeak':
                # Use espeak command
                import subprocess
                subprocess.run([
                    'espeak',
                    '-w', output_file,
                    '-s', str(int(self.voice_settings['rate'])),
                    text
                ], check=True)
            
            else:
                return None
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Audio generation failed: {e}")
            if Path(output_file).exists():
                os.unlink(output_file)
            return None
    
    def _generate_nonlatin(self, text: str, output_file: str) -> bool:
        """Synthesize non-Latin speech (e.g. Hebrew) to a WAV at output_file.

        Prefers gTTS (Google, natural voice, needs internet) and converts the
        mp3 to wav with ffmpeg; falls back to espeak-ng's offline voice so she
        can still speak with no internet. Returns True on success.
        """
        import subprocess
        lang = 'iw' if self.language.startswith('he') else self.language  # gTTS uses 'iw' for Hebrew

        # 1) gTTS -> mp3 -> ffmpeg -> wav (best quality)
        try:
            from gtts import gTTS
            mp3 = output_file + '.mp3'
            gTTS(text=text, lang=lang).save(mp3)
            r = subprocess.run(['ffmpeg', '-y', '-i', mp3, '-ar', '22050', '-ac', '1', output_file],
                               capture_output=True, timeout=30)
            try:
                os.unlink(mp3)
            except Exception:
                pass
            if r.returncode == 0 and Path(output_file).exists():
                return True
            self.logger.warning("ffmpeg mp3->wav failed: %s",
                                r.stderr.decode('utf-8', 'ignore')[:160])
        except Exception as exc:
            self.logger.info("gTTS unavailable/failed (%s) — using espeak-ng", str(exc)[:120])

        # 2) espeak-ng offline fallback (robotic but always works)
        try:
            voice = 'he' if self.language.startswith('he') else self.language
            r = subprocess.run(['espeak-ng', '-v', voice, '-s', '150', '-w', output_file, text],
                               capture_output=True, timeout=20)
            if r.returncode == 0 and Path(output_file).exists():
                return True
            self.logger.error("espeak-ng failed: %s", r.stderr.decode('utf-8', 'ignore')[:160])
        except Exception as exc:
            self.logger.error("espeak-ng synth error: %s", exc)
        return False

    def _device_opens(self, dev: str) -> bool:
        """True if an ALSA device can actually be opened for playback. Plays 50ms
        of silence (inaudible) — used to pick the live HDMI port."""
        import subprocess, wave, tempfile, os
        path = None
        try:
            path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            with wave.open(path, 'wb') as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(22050)
                wf.writeframes(b'\x00' * (int(22050 * 0.05) * 2))
            r = subprocess.run(['aplay', '-q', '-D', dev, path],
                               capture_output=True, timeout=5)
            return r.returncode == 0
        except Exception:
            return False
        finally:
            if path:
                try:
                    os.unlink(path)
                except Exception:
                    pass

    def _resolve_output_device(self):
        """Work out an ALSA device string for playback, based on
        config/config.yaml -> hardware.audio_output_*.

        HDMI outputs need the 'sysdefault:CARD=' route (they only accept
        IEC958 framing, which raw 'plughw' can't provide); other cards use
        'plughw:<idx>,0'. Returns None to use the ALSA default."""
        card = getattr(hardware_config, 'audio_output_card', None)
        name = getattr(hardware_config, 'audio_output_name', None)
        device = (getattr(hardware_config, 'audio_output_device', 'default') or 'default').lower()

        if device == 'default' and card is None and not name:
            return None

        # Parse `aplay -l` -> list of (idx, card_id, description_lower)
        cards = []
        try:
            import subprocess, re
            out = subprocess.run(['aplay', '-l'], capture_output=True, text=True).stdout
            for line in out.splitlines():
                m = re.match(r'card (\d+): (\S+) \[(.+?)\]', line)
                if m:
                    cards.append((int(m.group(1)), m.group(2),
                                  (m.group(2) + ' ' + m.group(3)).lower()))
        except Exception:
            cards = []

        def dev_string(idx, card_id, desc):
            if 'hdmi' in card_id.lower() or 'hdmi' in desc:
                return f"sysdefault:CARD={card_id}"
            return f"plughw:{idx},0"

        # 1) Explicit card index.
        if card is not None:
            for idx, cid, desc in cards:
                if idx == int(card):
                    return dev_string(idx, cid, desc)
            return f"plughw:{int(card)},0"

        # 2) HDMI: ALSA card numbers shuffle across reboots, and only the port
        # the monitor is actually plugged into accepts audio — so PROBE each HDMI
        # sink and use the first that truly opens. Reboot-proof, no fixed index.
        hdmi_cards = [(i, c, d) for i, c, d in cards if 'hdmi' in d]
        want_hdmi = device == 'hdmi' or (name and 'hdmi' in str(name).lower())
        if want_hdmi and hdmi_cards:
            if name:  # prefer the named port first, then fall back to any HDMI
                hdmi_cards.sort(key=lambda cc: 0 if str(name).lower() in cc[1].lower() else 1)
            for i, c, d in hdmi_cards:
                dev = dev_string(i, c, d)
                if self._device_opens(dev):
                    self.logger.info("HDMI audio auto-selected: %s", dev)
                    return dev
            self.logger.warning("No HDMI sink accepted audio; using %s anyway",
                                dev_string(*hdmi_cards[0]))
            return dev_string(*hdmi_cards[0])

        # 3) Match by keyword / name (non-HDMI).
        def find(keywords):
            for idx, cid, desc in cards:
                if any(k in desc for k in keywords):
                    return (idx, cid, desc)
            return None

        hit = None
        if name:
            hit = find([name.lower()])
        elif device == 'usb':
            hit = find(['usb'])
        elif device == 'analog':
            hit = find(['headphone', 'bcm2835', 'analog'])
        elif device == 'auto':
            for idx, cid, desc in cards:
                if 'hdmi' not in desc:
                    hit = (idx, cid, desc)
                    break
            if hit is None and cards:
                hit = cards[0]

        return dev_string(*hit) if hit else None

    def _play_audio(self, audio_file: str):
        """
        Play audio file

        Args:
            audio_file: Path to audio file
        """
        # HDMI/USB allows one stream at a time, so free the speaker from any
        # playing music for the moment Stella speaks, then hand it back. (Direct,
        # lock-free — a held lock across playback risked a deadlock/freeze.)
        mp = getattr(self, 'music', None)
        _freed = bool(mp is not None and mp.is_playing())
        if _freed:
            try:
                mp.pause_output()
                time.sleep(0.15)  # let ALSA release the device
            except Exception:
                _freed = False

        try:
            # If an explicit output device is configured, aplay -D is the most
            # reliable way to route sound to it (pygame/SDL uses ALSA default).
            if self.alsa_device:
                import subprocess
                r = subprocess.run(['aplay', '-q', '-D', self.alsa_device, audio_file],
                                   capture_output=True)
                if r.returncode != 0:
                    self.logger.warning("aplay failed (%s): %s", self.alsa_device,
                                        r.stderr.decode('utf-8', 'ignore')[:160])
                return

            if self.audio_player == 'pygame':
                # Use pygame
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.set_volume(self.voice_settings['volume'])
                pygame.mixer.music.play()
                
                # Wait for completion
                while pygame.mixer.music.get_busy():
                    if not self.running or self.paused:
                        pygame.mixer.music.stop()
                        break
                    time.sleep(0.1)
                    
            elif self.audio_player == 'pyaudio':
                # Use pyaudio
                self._play_with_pyaudio(audio_file)
                
            else:
                # Use system command
                import subprocess
                if sys.platform == 'darwin':
                    subprocess.run(['afplay', audio_file])
                elif sys.platform == 'win32':
                    import winsound
                    winsound.PlaySound(audio_file, winsound.SND_FILENAME)
                else:
                    subprocess.run(['aplay', audio_file])

        except Exception as e:
            self.logger.error(f"Audio playback failed: {e}")
        finally:
            if _freed and mp is not None:
                try:
                    mp.resume_output()
                except Exception:
                    pass

    def _play_with_pyaudio(self, audio_file: str):
        """Play audio using pyaudio"""
        
        wf = wave.open(audio_file, 'rb')
        
        p = pyaudio.PyAudio()
        
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True
        )
        
        chunk_size = 1024
        data = wf.readframes(chunk_size)
        
        while data and self.running and not self.paused:
            stream.write(data)
            data = wf.readframes(chunk_size)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf.close()
    
    def _cache_key(self, text: str) -> str:
        """Cache key includes engine + voice + speed/pitch/expressiveness so any
        of those changes invalidates old cached audio (no stale voice)."""
        if self.engine_type == 'piper':
            params = (f"{getattr(self,'piper_voice','')}|"
                      f"{getattr(self,'piper_length_scale',1.0)}|"
                      f"{getattr(self,'piper_pitch',0.0)}|"
                      f"{getattr(self,'piper_noise_scale',0.667)}")
        else:
            params = ''
        raw = f"{self.engine_type}|{self.language}|{params}|{text}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_cached_audio(self, text: str) -> Optional[str]:
        """
        Get cached audio file for text
        
        Args:
            text: Text to look up
            
        Returns:
            Path to cached audio file or None
        """
        
        # Generate cache key
        cache_key = self._cache_key(text)
        
        if cache_key in self.cache_index:
            cache_file = self.cache_dir / f"{cache_key}.wav"
            if cache_file.exists():
                return str(cache_file)
        
        return None
    
    def _cache_audio(self, text: str, audio_file: str):
        """
        Cache audio file
        
        Args:
            text: Original text
            audio_file: Path to audio file
        """
        
        try:
            # Generate cache key
            cache_key = self._cache_key(text)
            cache_file = self.cache_dir / f"{cache_key}.wav"
            
            # Move file to cache
            if Path(audio_file).exists():
                Path(audio_file).rename(cache_file)
                
                # Update index
                self.cache_index[cache_key] = {
                    'text': text[:100],
                    'created': time.time()
                }
                
                # Save index
                self._save_cache_index()
                
                # Limit cache size
                self._cleanup_cache()
                
        except Exception as e:
            self.logger.error(f"Cache error: {e}")
    
    def _load_cache_index(self):
        """Load cache index from disk"""
        
        index_file = self.cache_dir / "index.json"
        
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    self.cache_index = json.load(f)
            except:
                self.cache_index = {}
    
    def _save_cache_index(self):
        """Save cache index to disk"""
        
        index_file = self.cache_dir / "index.json"
        
        try:
            with open(index_file, 'w') as f:
                json.dump(self.cache_index, f)
        except Exception as e:
            self.logger.error(f"Failed to save cache index: {e}")
    
    def _cleanup_cache(self, max_files: int = 100):
        """
        Clean up old cache files
        
        Args:
            max_files: Maximum number of cache files
        """
        
        if len(self.cache_index) > max_files:
            # Sort by creation time
            sorted_items = sorted(
                self.cache_index.items(),
                key=lambda x: x[1].get('created', 0)
            )
            
            # Remove oldest
            to_remove = len(self.cache_index) - max_files
            
            for cache_key, _ in sorted_items[:to_remove]:
                cache_file = self.cache_dir / f"{cache_key}.wav"
                if cache_file.exists():
                    cache_file.unlink()
                del self.cache_index[cache_key]
            
            self._save_cache_index()
    
    def pause(self):
        """Pause speech"""
        self.paused = True
        
        if self.audio_player == 'pygame':
            pygame.mixer.music.pause()
    
    def resume(self):
        """Resume speech"""
        self.paused = False
        
        if self.audio_player == 'pygame':
            pygame.mixer.music.unpause()
    
    def clear_queue(self):
        """Clear speech queue"""
        
        while not self.speech_queue.empty():
            self.speech_queue.get()
        
        while not self.priority_queue.empty():
            self.priority_queue.get()
    
    def set_voice_settings(self, **kwargs):
        """
        Update voice settings
        
        Args:
            pitch: Voice pitch
            volume: Voice volume (0-1)
            rate: Speech rate (words per minute)
        """
        
        for key, value in kwargs.items():
            if key in self.voice_settings:
                self.voice_settings[key] = value
        
        # Apply to engine
        if self.engine_type == 'pyttsx3' and self.tts_engine:
            if 'rate' in kwargs:
                self.tts_engine.setProperty('rate', kwargs['rate'])
            if 'volume' in kwargs:
                self.tts_engine.setProperty('volume', kwargs['volume'])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get module statistics"""
        
        cache_size = len(list(self.cache_dir.glob('*.wav')))
        
        return {
            'running': self.running,
            'speaking': self.speaking,
            'engine': self.engine_type,
            'model': self.tts_model if self.engine_type == 'coqui' else 'N/A',
            'speech_count': self.speech_count,
            'total_characters': self.total_characters,
            'queue_size': self.speech_queue.qsize(),
            'cache_hits': self.cache_hits,
            'cache_size': cache_size,
            'cache_hit_rate': f"{(self.cache_hits/self.speech_count*100) if self.speech_count > 0 else 0:.1f}%"
        }
    
    def shutdown(self):
        """Clean shutdown"""
        self.stop()
        
        # Clean up TTS engine
        if self.engine_type == 'pyttsx3' and self.tts_engine:
            self.tts_engine.stop()


if __name__ == "__main__":
    """Test TTS module"""
    
    import sys
    
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create module
    tts_module = TextToSpeechModule()
    
    print("\nText-to-Speech Test")
    print("=" * 50)
    print("Commands:")
    print("  s <text> - Speak text")
    print("  p        - Pause")
    print("  r        - Resume")  
    print("  c        - Clear queue")
    print("  t        - Show statistics")
    print("  q        - Quit")
    print()
    
    # Start module
    tts_module.start()
    
    # Test messages
    test_messages = [
        "Hello! I am your AI robot assistant.",
        "My text to speech system is working properly.",
        "I can speak in a natural voice."
    ]
    
    try:
        # Initial greeting
        tts_module.speak(test_messages[0], priority=1)
        
        while True:
            cmd = input("Command: ").strip()
            
            if cmd.startswith('s '):
                text = cmd[2:]
                tts_module.speak(text)
                print(f"Speaking: '{text}'")
            
            elif cmd == 'p':
                tts_module.pause()
                print("Paused")
            
            elif cmd == 'r':
                tts_module.resume()
                print("Resumed")
            
            elif cmd == 'c':
                tts_module.clear_queue()
                print("Queue cleared")
            
            elif cmd == 't':
                stats = tts_module.get_statistics()
                print("\nStatistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                print()
            
            elif cmd == 'q':
                break
    
    except KeyboardInterrupt:
        pass
    
    finally:
        tts_module.stop()
        print("\nModule stopped")