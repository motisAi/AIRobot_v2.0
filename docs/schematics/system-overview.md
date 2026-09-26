# Schematic — System Overview

Current data-flow and hardware topology of Stella. Mermaid renders on GitHub; an ASCII
version follows for plain-text viewers.

## Block diagram (Mermaid)

```mermaid
graph TD
    subgraph PI["Raspberry Pi 5 + Hailo-10H  (orchestrator, Python, threaded)"]
        BRAIN["Robot brain / event bus"]
        FACE["Face recognition (dlib)"]
        WAKE["Wake word (Vosk)"]
        STT["STT (Google/Vosk)"]
        TTS["TTS (Piper) -> HDMI audio"]
        AI["LLM chain: Groq 70B -> GPT-OSS-20B -> Hailo(offline)"]
        VLM["VLM look (Moondream cloud)"]
        MIRROR["Hand mirror (MediaPipe)"]
        GUARD["Motion / home guard"]
        WATCH["Hardware watchdog"]
        TG["Telegram bridge"]
        CAM["CameraManager (shared)"]
    end

    USBCAM["USB camera + onboard mic"] --> CAM
    MICA["Mic A"] --> WAKE
    MICB["Mic B"] --> STT
    CAM --> FACE
    CAM --> VLM
    CAM --> MIRROR
    CAM --> GUARD

    BRAIN --> TTS
    AI --> BRAIN
    VLM --> BRAIN
    TG <--> BRAIN
    WATCH -. checks .-> CAM
    WATCH -. checks .-> TTS

    BRAIN -->|USB serial /dev/ttyACM0 115200| ESP["ESP32-S3-WROOM-1"]
    ESP -->|I2C 0x40  SDA=GPIO8 SCL=GPIO9| PCA["PCA9685 16ch PWM"]
    PCA -->|PWM ~50Hz| SERVOS["5x finger servos (ch0..4 = pinky..thumb)"]
    PSU["External 5-6V servo supply"] -->|V+ / common GND| PCA

    CLOUD["Cloud: Groq, Moondream, Telegram"] <-.WiFi.-> PI

    TABLET["(planned) Tablet face - Teclast P30T"] <-.WiFi ws://8765 + http://8080.-> PI
```

## ASCII fallback

```
        Mic A ─────────────► [Wake word]
        Mic B ─────────────► [STT] ──► [LLM chain] ──► [brain] ──► [TTS] ──► HDMI audio
   USB camera ──► [CameraManager] ──► face / VLM / hand-mirror / motion-guard
                                                     │
   Telegram  <──WiFi──►  [brain]  ◄── VLM (Moondream, cloud)
                              │
                              ▼  USB serial /dev/ttyACM0 @115200
                       ESP32-S3-WROOM-1
                              │  I2C 0x40 (SDA=GPIO8, SCL=GPIO9)
                              ▼
                        PCA9685 (16ch PWM, ~50Hz)  ◄── external 5-6V (V+, common GND)
                              │  PWM
                              ▼
        5x finger servos:  ch0=pinky  ch1=ring  ch2=middle  ch3=index  ch4=thumb

   (planned)  Tablet face (P30T) ◄─WiFi ws://<pi>:8765 + http://<pi>:8080─► Pi
```

_Update this when connections change. For a new subsystem, add its nodes/edges here and a
matching entry in `../hardware/wiring.md`._
