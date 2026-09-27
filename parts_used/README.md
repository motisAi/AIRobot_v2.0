# parts_used — one file per physical part

Every sensor, actuator, radio or accelerator that Stella is built from has exactly one
file here with one main class. These files talk to hardware and nothing else: no
"greet when a face appears" logic, no LLM calls. The capability code in `modules/`
receives these objects from `main.py` and uses them.

| File | Class | Part | Interface | Config |
|---|---|---|---|---|
| `camera_usb.py` | `CameraManager`, `Frame` | USB webcam (auto‑detected by name, CSI nodes skipped) — the only `cv2.VideoCapture` on the live path | V4L2 `/dev/video*` | `hardware.camera_*` |
| `audio_devices.py` | `AudioManager`, `AudioDevice` (→ `find_input_index()` by‑id resolver, deep‑clean fix) | The two microphones and their roles: WAKE = camera built‑in mic (card `Camera`), COMMAND = USB PnP Sound Device (card `Device`); speaker role | ALSA `/dev/snd/by-id` / PortAudio | `hardware.*microphone*`, `hardware.audio_output_*` |
| `audio_portaudio.py` | `get_pa()`, `open_lock()` (→ `MicStream`, `open_input()`, deep‑clean fix) | The single process‑wide PortAudio instance (a second `Pa_Initialize` crashes the Pi) and the owner‑thread guard for every stream | PortAudio | — |
| `audio_arbiter.py` | `AudioArbiter` | Who owns the one speaker right now (speech ducks music) | — | `music.duck_volume` |
| `hailo_10h.py` | `HailoDetector`, `Detection` | Object detector: OpenCV‑DNN CPU path on `data/models/yolov8n.onnx` is the live path; the Hailo path waits for the runtime repair (pip `hailort` is ABI‑broken) | ONNX / HailoRT | `system.enable_hailo`, `model.object_*` |
| `esp32_hand.py` | `Hand` | 5‑finger hand (ESP32‑S3 + PCA9685), persistent serial link | USB serial `/dev/ttyACM0` | `hand.*` |
| `microcontroller_bridge.py` | `MicrocontrollerController` + transports | **The only MCU command path**: hardware‑agnostic sink for real‑world commands (serial / network / null) | serial or HTTP | `microcontroller.*` |
| `wifi_adapter.py` | helpers `is_online()`, `scan()`, `connect()`, QR read | Onboard WiFi + sudoers helper (`deploy/wifi-helper.sh`) | nmcli/wpa_cli | — |
| `rc_toy.py` | `RCToy` | RC toy chassis (not chosen yet) — safe no‑op until `rc_toy.connected: true` | null / ESP32 serial / BLE / WiFi | `rc_toy.*` |

Removed 2026-09-27 (no importer anywhere): `esp32_controller.py` (superseded by `microcontroller_bridge.py`) and
`sim7600x_modem.py` (modem not fitted; its import‑time serial probe blocked every `config.settings` import).

Not a file here but a part: the **Hailo‑10H as LLM** is served by the system service
`hailo-ollama` (port 8000) and used by `modules/ai/ai_engine.py`; the **screen face** is
`face_bridge/` (`:8080`); **smart‑home devices** (Sensibo AC, Tuya plug, MQTT relays) are external
devices, so they live in `modules/smart_home/`.

## Contract for a part file
```python
class MyPart:
    def __init__(self, config): ...
    def is_available(self) -> bool: ...   # is the hardware actually present right now?
    def start(self) -> bool: ...          # never raise; return False and log once
    def stop(self) -> None: ...
```
- **Resolve devices by stable identity, never by index**: `/dev/snd/by-id` symlink → `controlC<N>` → the PortAudio
  entry whose name contains `(hw:N,` for microphones; ALSA card id string (`hw:CARD=Camera`) for mixer controls; USB
  VID:PID or by‑id for serial. ALSA card numbers and PortAudio indices both change across reboots and replugs.
  An unresolved device is an ERROR + backoff, never a silent fall‑through to `default`.
- **One thread owns a PortAudio stream**: the thread that opens a stream is the only one that reads, stops and closes
  it (`MicStream` raises otherwise). Never abandon a thread holding a stream; a close/join timeout is terminal for that
  device in this process (flag it, let the watchdog recover). Never construct a second `pyaudio.PyAudio()`. Wake and
  command must be two different devices (bug_028, bug_054, bug_055; architecture doc §4).
- Missing hardware is normal (bench, travel): log once, back off, keep the rest of Stella alive.
- Add a new part = one new file here + a config block + a line in this table. The
  self‑evolution manifest (`evolution/manifest.py`) picks it up from this folder.

Wiring and power details: [docs/hardware/wiring.md](../docs/hardware/wiring.md),
[docs/hardware/hailo.md](../docs/hardware/hailo.md). Canonical audio design:
[docs/architecture/stella-architecture.md §4](../docs/architecture/stella-architecture.md#4-audio--the-canonical-two-mic-design).
