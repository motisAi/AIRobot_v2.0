# parts_used — one file per physical part

Every sensor, actuator, radio or accelerator that Stella is built from has exactly one
file here with one main class. These files talk to hardware and nothing else: no
"greet when a face appears" logic, no LLM calls. The capability code in `modules/`
receives these objects from `main.py` and uses them.

| File | Class | Part | Interface | Config |
|---|---|---|---|---|
| `camera_usb.py` | `CameraManager`, `Frame` | USB webcam (auto‑detected by name, CSI nodes skipped) | V4L2 `/dev/video*` | `hardware.camera_*` |
| `audio_devices.py` | `AudioManager`, `AudioDevice` | Microphones + speaker roles (wake mic, dialogue mic, output) | ALSA / PortAudio | `hardware.*microphone*`, `hardware.audio_output_*` |
| `audio_portaudio.py` | `get_pa()`, `open_lock()` | Single process‑wide PortAudio instance (opening many crashes the Pi) | PortAudio | — |
| `audio_arbiter.py` | `AudioArbiter` | Who owns the one speaker right now (speech ducks music) | — | `music.duck_volume` |
| `hailo_10h.py` | `HailoDetector`, `Detection` | Hailo‑10H NPU object detector, with OpenCV‑DNN CPU fallback (current path) | HailoRT / ONNX | `system.enable_hailo`, `model.object_*` |
| `esp32_hand.py` | `Hand` | 5‑finger hand (ESP32‑S3 + PCA9685), persistent serial link | USB serial `/dev/ttyACM0` | `hand.*` |
| `esp32_controller.py` | `ESP32Controller` | Generic ESP32 I/O board protocol (relays, sensors) | USB serial | `microcontroller.*` |
| `microcontroller_bridge.py` | `MicrocontrollerController` + transports | Hardware‑agnostic sink for real‑world commands (serial / network / null) | serial or HTTP | `microcontroller.*` |
| `sim7600x_modem.py` | `SIM7600XController` | Waveshare SIM7600X 4G/LTE modem (not fitted today) | USB serial | — |
| `wifi_adapter.py` | helpers `is_online()`, `scan()`, `connect()`, QR read | Onboard WiFi + sudoers helper (`deploy/wifi-helper.sh`) | nmcli/wpa_cli | — |
| `rc_toy.py` | `RCToy` | RC toy chassis (not chosen yet) — safe no‑op until `rc_toy.connected: true` | null / ESP32 serial / BLE / WiFi | `rc_toy.*` |

Not a file here but a part: the **Hailo‑10H as LLM** is served by the system service
`hailo-ollama` (port 8000) and used by `modules/ai/ai_engine.py`; the **screen face** is
`face_bridge/`; **smart‑home devices** (Sensibo AC, Tuya plug, MQTT relays) are external
devices, so they live in `modules/smart_home/`.

## Contract for a part file
```python
class MyPart:
    def __init__(self, config): ...
    def is_available(self) -> bool: ...   # is the hardware actually present right now?
    def start(self) -> bool: ...          # never raise; return False and log once
    def stop(self) -> None: ...
```
- Resolve devices **by name**, never by index (indices change every reboot).
- Missing hardware is normal (bench, travel): log once, back off, keep the rest of Stella alive.
- Add a new part = one new file here + a config block + a line in this table. The
  self‑evolution manifest (`evolution/manifest.py`) picks it up from this folder.

Wiring and power details: [docs/hardware/wiring.md](../docs/hardware/wiring.md),
[docs/hardware/hailo.md](../docs/hardware/hailo.md).
