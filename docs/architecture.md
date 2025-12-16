# AIRobot Architecture

## High-Level Flow
```
Wake Word (USB Mic) --> Speech Recognition --> Robot Brain --> Behaviors
          ^                                              |
          |                                              v
  Text-to-Speech <----------------------------------- Event Bus
```

1. **WakeWordModule** listens for "Gonzo" on a dedicated USB microphone.  It
   emits a `wake_word_detected` event and pauses itself until dialogue ends.
2. **SpeechRecognitionModule** records the command on demand, transcribes it via
   Whisper, and emits `speech_recognized` or `speech_listen_failed` events.
3. **RobotBrain** (state machine) transitions through `LISTENING -> PROCESSING ->
   RESPONDING`.  It stores memories, manages authentication, and emits
   `dialogue_idle` once the state returns to `IDLE`.
4. **FaceRecognitionModule** continuously feeds `face_detected` events.  When the
   master user is identified the brain unlocks privileged commands.
5. **TextToSpeechModule** speaks the response, caches audio files, and emits
   `speech_complete` events for visibility.
6. **ESP32Controller** handles low-level hardware when enabled.  All commands go
   through a queue so high-level behaviors remain asynchronous.

## Packages and Responsibilities
| Package | Responsibility |
| ------- | -------------- |
| `config` | Loads `.env`, applies platform overrides, exports typed dataclasses. |
| `core` | Robot brain (state machine + memory + decision engine). |
| `modules/audio` | Wake word, STT, and TTS subsystems with hardware-aware configuration. |
| `modules/vision` | Face detection and recognition using DeepFace/OpenCV. |
| `modules/hardware` | ESP32 + SIM7600X controllers with UART status tracking. |
| `modules/connectivity` | Cloud/server bridge (disabled by default). |

## Event Types
- `wake_word_detected` – emitted by `WakeWordModule`.
- `speech_recognized` – emitted by `SpeechRecognitionModule` with `{text}`.
- `speech_listen_failed` – emitted when audio capture/transcription fails.
- `dialogue_idle` – emitted by `RobotBrain` after returning to `IDLE`; used to
  resume the wake-word listener.
- `speech_complete` – emitted by `TextToSpeechModule` after playback.
- `face_detected` – emitted by `FaceRecognitionModule` with face metadata.
- `battery_low`, `object_detected`, `sensor_reading`, etc. – available for
  hardware integrations.

## Configuration Flags Worth Knowing
- `hardware.is_esp_connected` – master switch for the ESP32 controller.
- `hardware.wake_word_microphone_name` / `speech_microphone_name` – lock each
  audio module to its own USB mic.
- `system.platform_name` – set automatically via `config/platforms`.  Additional
  overrides can be added for other boards.
- `behavior.auto_charge` – placeholder for autonomous docking logic.

## Data Storage
- Faces/embeddings: `data/faces/` and `data/faces/face_db.pkl`.
- Logs: `data/logs/` (rotating per run).
- Whisper/TTS caches: `~/.cache/robot_whisper` and `data/tts_cache/`.
- Emergency snapshots: `data/emergency_*.json`.

Extend the system by adding new modules inside `modules/` and registering event
handlers in `main.AIRobot._register_event_handlers`.  The brain only needs the
`emit_event` contract, so you can plug in additional sensors or services without
changing the control loop.
