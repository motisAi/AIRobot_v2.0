# AIRobot Architecture (v2.0 — RPi5 + Hailo)

## High-Level Flow
```
                  ┌──────────────────────────────────────────────────────────┐
                  │                    CameraManager                        │
                  │  (single camera, distributes frames to subscribers)     │
                  └────────┬────────────────────────────┬───────────────────┘
                           │                            │
                  ┌────────▼────────┐          ┌────────▼────────┐
                  │ FaceRecognition │          │ ObjectDetection  │
                  │   (DeepFace)    │          │ (HailoDetector)  │
                  └────────┬────────┘          └────────┬────────┘
                           │ face_detected              │ object_detected
                  ┌────────▼────────────────────────────▼────────┐
                  │                  Event Bus                    │
                  │               (RobotBrain)                    │
                  └──┬─────┬─────┬─────┬─────┬─────┬─────┬──────┘
   wake_word_detected│     │     │     │     │     │     │dialogue_idle
          ┌──────────▼─┐ ┌─▼─────▼─┐ ┌─▼─────▼─┐ ┌─▼─────▼──┐
          │ WakeWord   │ │ Speech  │ │ TTS     │ │ AIEngine │
          │ (Porcupine)│ │ Recog.  │ │         │ │ online/  │
          └────────────┘ │(Whisper)│ │(Coqui/  │ │ offline  │
                         └─────────┘ │ pyttsx3)│ └──────────┘
                                     └─────────┘
            AudioManager — exclusive mic/speaker leases per role
```

1. **CameraManager** opens the USB camera once and distributes frames to all
   vision subscribers (face recognition, object detection, future modules).
2. **HailoDetector** runs YOLO on the Hailo NPU (HEF model). Falls back to
   OpenCV DNN ONNX → MobileNet SSD Caffe → disabled if nothing is available.
3. **ObjectDetectionModule** subscribes to CameraManager, runs HailoDetector,
   tracks objects across frames via IoU, emits `object_detected` events.
4. **FaceRecognitionModule** subscribes to CameraManager, runs DeepFace, emits
   `face_detected` events. Master-user authentication unlocks privileged commands.
5. **AudioManager** provides exclusive per-role leases (wake_word, dialogue,
   playback) so modules never fight over mic/speaker devices.
6. **WakeWordModule** listens for "Gonzo" on a dedicated USB mic, emits
   `wake_word_detected`, pauses until dialogue completes.
7. **SpeechRecognitionModule** records on demand, transcribes via Whisper, emits
   `speech_recognized` or `speech_listen_failed`.
8. **AIEngine** processes user input: OpenAI/Anthropic API (online) →
   llama-cpp-python GGUF (offline) → keyword rules (fallback).
9. **LearningDB** (SQLite) persists memories, faces, objects, conversations,
   and preferences across reboots.
10. **RobotBrain** state machine (`transitions`): IDLE → LISTENING → PROCESSING
    → RESPONDING → IDLE. Manages working/long-term memory, event routing, and
    behaviour triggers (patrol, learning, observing).
11. **TextToSpeechModule** speaks the response, caches audio, emits
    `speech_complete`. Uses Coqui TTS / pyttsx3 / espeak.
12. **ESP32Controller** handles low-level motor/servo commands over UART (queued).
13. **SIM7600XController** provides 4G LTE connectivity status and AT commands.

## Packages and Responsibilities
| Package | Responsibility |
| ------- | -------------- |
| `config` | Loads `.env`, detects platform (RPi5/Jetson), applies overrides, exports typed dataclasses. |
| `core` | Robot brain (state machine + memory + decision engine). |
| `modules/ai` | AI engine (online/offline LLM) + SQLite learning database. |
| `modules/audio` | Wake word, STT, and TTS with hardware-aware config. |
| `modules/vision` | Face recognition (DeepFace) + Hailo/OpenCV object detection. |
| `modules/hardware` | CameraManager, AudioManager, ESP32, SIM7600X controllers. |
| `modules/connectivity` | Cloud/server bridge (disabled by default). |

## Event Types
| Event | Source | Data |
| ----- | ------ | ---- |
| `wake_word_detected` | WakeWordModule | `{}` |
| `speech_recognized` | SpeechRecognitionModule | `{text}` |
| `speech_listen_failed` | SpeechRecognitionModule | `{reason}` |
| `speech_complete` | TextToSpeechModule | `{text}` |
| `dialogue_idle` | RobotBrain | `{}` |
| `face_detected` | FaceRecognitionModule | `{face_id, name, confidence, is_master}` |
| `object_detected` | ObjectDetectionModule | `{label, confidence, bbox, is_new}` |
| `user_authenticated` | main | `{user_id, method}` |
| `battery_low` | ESP32Controller | `{level}` |
| `emergency_stop` | any | `{}` |

## Configuration Flags
- `hardware.is_esp_connected` — gates ESP32 controller.
- `hardware.wake_word_microphone_name` / `speech_microphone_name` — per-role
  mic routing via AudioManager.
- `system.platform_name` — auto-detected (raspberry_pi5 / jetson_nano / generic).
- `AI_MODE` env var — `auto` (default), `online`, `offline`.
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` — cloud LLM credentials.
- `OFFLINE_MODEL_PATH` — path to local `.gguf` model.

## Data Storage
| Path | Contents |
| ---- | -------- |
| `data/robot_memory.db` | SQLite learning database (memories, faces, objects, conversations, prefs) |
| `data/faces/` | Face images and `face_db.pkl` embeddings |
| `data/models/` | Hailo HEF / ONNX / Caffe model files |
| `data/logs/` | Rotating per-run log files |
| `data/tts_cache/` | Cached TTS audio files |

## Adding New Modules
1. Create your module in `modules/<category>/`.
2. Accept `camera_manager` and/or `audio_manager` if it uses shared hardware.
3. Register event handlers in `main.AIRobot._register_event_handlers()`.
4. Instantiate and wire in `main.AIRobot.initialize_modules()`.
5. The brain only needs the `emit_event(RobotEvent)` contract.
