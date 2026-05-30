#!/usr/bin/env python3
"""Audit all dependencies for the AI Robot."""
import subprocess, os, sys

print("=== Python Packages ===")
pkgs = [
    'openai', 'speech_recognition', 'pyttsx3', 'pygame', 'face_recognition',
    'dlib', 'cv2', 'numpy', 'flask', 'whisper', 'pvporcupine', 'pvrecorder',
    'webrtcvad', 'pyaudio', 'TTS', 'requests', 'dotenv', 'transitions',
    'scipy', 'soundfile', 'sounddevice', 'torch', 'torchaudio'
]
for p in pkgs:
    try:
        m = __import__(p)
        v = getattr(m, '__version__', getattr(m, 'VERSION', 'ok'))
        print(f"  {p}: {v}")
    except Exception:
        print(f"  {p}: NOT INSTALLED")

print("\n=== System Tools ===")
for tool in ['flac', 'ffmpeg', 'espeak-ng', 'arecord', 'aplay', 'curl', 'sox']:
    r = subprocess.run(['which', tool], capture_output=True)
    status = "FOUND" if r.returncode == 0 else "MISSING"
    print(f"  {tool}: {status}")

print("\n=== Env Vars ===")
from pathlib import Path
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            key = line.split('=', 1)[0].strip()
            print(f"  {key}: SET")
else:
    print("  .env file: NOT FOUND")
print(f"  OPENAI_API_KEY (env): {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
print(f"  PICOVOICE_ACCESS_KEY (env): {'SET' if os.getenv('PICOVOICE_ACCESS_KEY') else 'NOT SET'}")

print("\n=== Microphone Test ===")
try:
    import pyaudio
    pa = pyaudio.PyAudio()
    # Try reading from device 0 (camera mic - wake word)
    try:
        s = pa.open(format=pyaudio.paInt16, channels=1, rate=48000, input=True,
                    frames_per_buffer=512, input_device_index=0)
        data = s.read(512, exception_on_overflow=False)
        import struct, math
        samples = struct.unpack(f'{len(data)//2}h', data)
        rms = math.sqrt(sum(x*x for x in samples) / len(samples))
        print(f"  Device 0 (camera mic): OK, RMS={rms:.0f}")
        s.close()
    except Exception as e:
        print(f"  Device 0 (camera mic): FAILED - {e}")
    # Try reading from device 2 (USB PnP - speech)
    try:
        s = pa.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True,
                    frames_per_buffer=512, input_device_index=2)
        data = s.read(512, exception_on_overflow=False)
        samples = struct.unpack(f'{len(data)//2}h', data)
        rms = math.sqrt(sum(x*x for x in samples) / len(samples))
        print(f"  Device 2 (USB PnP): OK, RMS={rms:.0f}")
        s.close()
    except Exception as e:
        print(f"  Device 2 (USB PnP): FAILED - {e}")
    pa.terminate()
except Exception as e:
    print(f"  PyAudio: FAILED - {e}")

print("\n=== OpenAI API Test ===")
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI()
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello in one word"}],
        max_tokens=5
    )
    print(f"  GPT-4o-mini: OK - '{r.choices[0].message.content}'")
except Exception as e:
    print(f"  GPT-4o-mini: FAILED - {e}")

print("\n=== Google Speech API Test ===")
try:
    import speech_recognition as sr
    print(f"  speech_recognition: {sr.__version__}")
    # Check if FLAC works
    r2 = subprocess.run(['flac', '--version'], capture_output=True)
    print(f"  FLAC: {r2.stdout.decode().strip() if r2.returncode == 0 else 'MISSING'}")
except Exception as e:
    print(f"  Speech Recognition: FAILED - {e}")

print("\n=== TTS Test ===")
try:
    import pyttsx3
    engine = pyttsx3.init()
    print(f"  pyttsx3: OK")
    engine.stop()
except Exception as e:
    print(f"  pyttsx3: FAILED - {e}")
