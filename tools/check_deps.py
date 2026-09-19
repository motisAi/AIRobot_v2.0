#!/usr/bin/env python3
"""Audit dependencies & hardware for Gonzo (RPi5 + Hailo-10H)."""
import subprocess, os, json, urllib.request
from pathlib import Path

def ok(b): return "OK" if b else "-- MISSING"

print("=== Python Packages ===")
pkgs = [
    'yaml', 'numpy', 'cv2', 'face_recognition', 'dlib', 'pyaudio', 'webrtcvad',
    'pygame', 'pyttsx3', 'speech_recognition', 'vosk', 'ddgs', 'openai',
    'anthropic', 'serial', 'gpiozero', 'flask', 'transitions', 'dotenv', 'psutil',
]
for p in pkgs:
    try:
        m = __import__(p)
        v = getattr(m, '__version__', getattr(m, 'VERSION', 'ok'))
        print(f"  {p}: {v}")
    except Exception:
        print(f"  {p}: -- NOT INSTALLED (optional for some features)")

print("\n=== System Tools ===")
for tool in ['espeak-ng', 'arecord', 'aplay', 'curl', 'ffmpeg', 'hailortcli']:
    r = subprocess.run(['which', tool], capture_output=True)
    print(f"  {tool}: {ok(r.returncode == 0)}")

print("\n=== Hailo-10H NPU ===")
try:
    r = subprocess.run(['hailortcli', 'fw-control', 'identify'],
                       capture_output=True, text=True, timeout=10)
    line = next((l for l in r.stdout.splitlines() if 'Architecture' in l), '')
    print(f"  device: {line.strip() or 'detected' if r.returncode==0 else 'NOT FOUND'}")
except Exception as e:
    print(f"  hailortcli: -- {e}")

print("\n=== On-device LLM (hailo-ollama) ===")
try:
    with urllib.request.urlopen("http://localhost:8000/api/tags", timeout=4) as resp:
        tags = json.loads(resp.read().decode())
    models = [m.get('name') for m in tags.get('models', [])]
    print(f"  installed models: {', '.join(models) or 'NONE (pull one via /hailo/v1/pull)'}")
except Exception as e:
    print(f"  hailo-ollama: -- not reachable ({e})")

print("\n=== Wake-word model (Vosk) ===")
vosk_dir = Path(__file__).parent / 'data' / 'models' / 'vosk-small-en'
print(f"  {vosk_dir}: {ok(vosk_dir.exists())}")

print("\n=== Config ===")
try:
    from config.settings import config, ai_config, microcontroller_config
    print(f"  config loaded: robot={config.behavior.robot_name}, "
          f"ai.mode={ai_config.mode}, npu_model={ai_config.hailo_ollama_model}, "
          f"microcontroller={microcontroller_config.connected}")
except Exception as e:
    print(f"  config: -- FAILED {e}")

print("\n=== Microphones ===")
try:
    import pyaudio
    pa = pyaudio.PyAudio()
    for idx in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(idx)
        if info.get('maxInputChannels', 0) > 0:
            print(f"  [{idx}] {info.get('name')}")
    pa.terminate()
except Exception as e:
    print(f"  PyAudio: -- FAILED {e}")

print("\n=== Playback devices ===")
r = subprocess.run(['aplay', '-l'], capture_output=True, text=True)
for l in r.stdout.splitlines():
    if l.startswith('card'):
        print(f"  {l}")

print("\n=== Web search (free) ===")
try:
    from modules.ai.web_search import search_web, available
    res = search_web("hello world", max_results=1)
    print(f"  backend={available()} results={len(res)} {ok(bool(res))}")
except Exception as e:
    print(f"  web search: -- {e}")

print("\n=== Env / secrets (.env) ===")
print(f"  OPENAI_API_KEY: {ok(bool(os.getenv('OPENAI_API_KEY')))}")
print(f"  ANTHROPIC_API_KEY: {ok(bool(os.getenv('ANTHROPIC_API_KEY')))}")
print("\nDone.")
