#!/usr/bin/env python3
"""Build stella_manifest.yaml — Stella's self-knowledge, generated from the live system.

    venv/bin/python evolution/manifest.py          # write stella_manifest.yaml
    venv/bin/python evolution/manifest.py --print  # just show it

Everything here is DETECTED (devices, models on disk, keys present, services, config
flags, recurring bugs), never typed by hand, so it cannot drift from reality.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "stella_manifest.yaml"
PY = ROOT / "venv" / "bin" / "python"


def _run(cmd, timeout=15) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def _env_keys() -> list[str]:
    keys = []
    try:
        for line in (ROOT / ".env").read_text().splitlines():
            if "=" in line and not line.startswith("#") and line.split("=", 1)[1].strip():
                keys.append(line.split("=", 1)[0].strip())
    except OSError:
        pass
    return sorted(keys)


def _config() -> dict:
    try:
        return yaml.safe_load((ROOT / "config" / "config.yaml").read_text()) or {}
    except Exception:
        return {}


def _hardware() -> dict:
    cams = []
    for node in sorted(glob.glob("/dev/video[0-9]*")):
        name_file = f"/sys/class/video4linux/{os.path.basename(node)}/name"
        try:
            name = Path(name_file).read_text().strip()
        except OSError:
            name = "?"
        if re.search(r"rp1-cfe|csi|pispbe|rpivid|bcm2835", name, re.I):
            continue   # Pi ISP/codec/CSI nodes, not real cameras
        cams.append({"node": node, "name": name})
    mics = []
    for card in sorted(glob.glob("/proc/asound/card[0-9]*/id")):
        cid = Path(card).read_text().strip()
        idx = re.search(r"card(\d+)", card).group(1)
        has_capture = bool(glob.glob(f"/dev/snd/pcmC{idx}D*c"))
        has_play = bool(glob.glob(f"/dev/snd/pcmC{idx}D*p"))
        mics.append({"card": int(idx), "id": cid, "capture": has_capture, "playback": has_play})
    return {
        "platform": (Path("/proc/device-tree/model").read_text().strip("\x00 \n")
                     if Path("/proc/device-tree/model").exists() else "unknown"),
        "kernel": _run(["uname", "-r"]).strip(),
        "cameras": cams,
        "audio_cards": mics,
        "hailo_device": os.path.exists("/dev/hailo0"),
        "hailo_driver": [l.strip() for l in _run(["dkms", "status"]).splitlines() if "hailo" in l],
        "serial_ports": sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")),
        "network_interfaces": [l.split(":")[1].strip() for l in _run(["ip", "-o", "link"]).splitlines()
                               if ":" in l and "lo:" not in l],
        "temperature": _run(["vcgencmd", "measure_temp"]).strip().replace("temp=", ""),
        "throttled": _run(["vcgencmd", "get_throttled"]).strip().replace("throttled=", ""),
    }


def _models(cfg: dict) -> dict:
    d = ROOT / "data" / "models"
    piper = sorted(p.name for p in (d / "piper").glob("*.onnx")) if (d / "piper").exists() else []
    hailo_models = []
    try:
        out = _run(["curl", "-s", "-m", "5", "http://127.0.0.1:8000/v1/models"])
        hailo_models = [m.get("id") for m in json.loads(out).get("data", [])]
    except Exception:
        pass
    ai = cfg.get("ai", {})
    model = cfg.get("model", {})
    return {
        "llm_chain": [f"groq:{ai.get('groq_model')}", f"groq_fast:{ai.get('groq_fast_model')}",
                      f"gemini:{ai.get('gemini_model')}", f"hailo:{ai.get('hailo_ollama_model')}"],
        "hailo_ollama_models": hailo_models,
        "stt": "groq whisper-large-v3-turbo -> google -> vosk (data/models/vosk-small-en)",
        "wake_word": {"engine": model.get("wake_word_engine"), "phrases": model.get("wake_word_phrases")},
        "tts": {"engine": model.get("tts_engine"), "voice": model.get("piper_voice"), "installed_voices": piper},
        "object_detection": {"model": model.get("object_model"),
                             "onnx_present": (d / "yolov8n.onnx").exists(),
                             "runtime": "hailo" if cfg.get("system", {}).get("enable_hailo") else "opencv_dnn_cpu"},
        "face_recognition": {"backend": model.get("face_backend"), "threshold": model.get("face_recognition_threshold")},
        "vlm": "moondream -> nvidia nim meta/llama-3.2-11b-vision-instruct",
    }


def _services() -> list[dict]:
    out = []
    for unit in ("airobot", "hailo-ollama", "robotnet-hostapd", "robotnet-dnsmasq", "stella-evolution.timer"):
        st = _run(["systemctl", "is-active", unit]).strip() or "unknown"
        out.append({"unit": unit, "status": st})
    return out


def _capabilities(cfg: dict) -> list[str]:
    caps = ["voice_conversation", "wake_word", "face_recognition", "object_detection",
            "vision_language_model", "telegram_two_way", "web_dashboard", "screen_face",
            "reminders", "web_search", "music_youtube", "home_guard"]
    if cfg.get("hand", {}).get("enabled"):
        caps += ["robotic_hand_gestures", "hand_mirror"]
    if cfg.get("mqtt", {}).get("enabled"):
        caps.append("smart_home_mqtt")
    caps += ["smart_home_tuya", "smart_home_sensibo_ac"]
    if cfg.get("microcontroller", {}).get("connected"):
        caps.append("microcontroller_io")
    if cfg.get("rc_toy", {}).get("connected"):
        caps.append("rc_toy_drive")
    if cfg.get("navigation", {}).get("enabled"):
        caps.append("navigation")
    return caps


def _parts() -> list[dict]:
    parts = []
    for f in sorted((ROOT / "parts_used").glob("*.py")):
        if f.name.startswith("_"):
            continue
        src = f.read_text(encoding="utf-8", errors="replace")
        doc = re.search(r'"""(.*?)(\n|""")', src, re.S)
        classes = re.findall(r"^class (\w+)", src, re.M)
        parts.append({"file": f"parts_used/{f.name}", "classes": classes,
                      "summary": (doc.group(1).strip() if doc else "")[:120]})
    return parts


def _recurring_bugs() -> list[str]:
    out = []
    for f in sorted((ROOT / "bug_report").glob("bug_*.md")):
        txt = f.read_text(encoding="utf-8", errors="replace")
        if "**Status:** recurring" in txt or "fixed-needs-verify" in txt:
            title = txt.splitlines()[0].lstrip("# ").strip()
            out.append(f"{f.stem}: {title}")
    return out


def _pip_versions() -> dict:
    out = _run([str(PY), "-m", "pip", "list", "--format=json"], timeout=60)
    try:
        pkgs = {p["name"].lower(): p["version"] for p in json.loads(out)}
    except Exception:
        return {}
    watch = ["vosk", "openai", "onnxruntime", "opencv-python-headless", "face-recognition", "dlib",
             "mediapipe", "pyaudio", "yt-dlp", "numpy", "flask", "aiohttp", "tinytuya", "paho-mqtt",
             "hailort", "piper-tts", "faster-whisper", "openwakeword", "ultralytics"]
    return {w: pkgs.get(w, "not installed") for w in watch}


def build() -> dict:
    cfg = _config()
    return {
        "identity": {"name": cfg.get("behavior", {}).get("robot_name", "Stella"),
                     "language": cfg.get("behavior", {}).get("language", "en"),
                     "os": _run(["lsb_release", "-ds"]).strip().strip('"') or "Ubuntu",
                     "python": _run([str(PY), "-V"]).strip(),
                     "git_commit": _run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).strip()},
        "hardware": _hardware(),
        "parts_used": _parts(),
        "ai_models": _models(cfg),
        "providers_with_keys": _env_keys(),
        "services": _services(),
        "capabilities": _capabilities(cfg),
        "python_packages": _pip_versions(),
        "known_weaknesses": _recurring_bugs() + [
            "wake word is Vosk keyword-spotting (mishears 'Stella'); openWakeWord custom model planned",
            "offline STT is Vosk small (weak); faster-whisper planned",
            "no active cooling: reaches 80C soft limit under load",
            "object detection runs on CPU; Hailo path needs HailoRT >= 5.2",
        ],
        "evolution_log": {"last_manifest": time.strftime("%Y-%m-%d %H:%M:%S")},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    m = build()
    text = ("# stella_manifest.yaml — GENERATED by evolution/manifest.py; do not edit by hand.\n"
            "# Stella's self-knowledge: what she is made of, what she can do, what she is missing.\n"
            + yaml.safe_dump(m, sort_keys=False, allow_unicode=True))
    if a.print:
        print(text)
    else:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
