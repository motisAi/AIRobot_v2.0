# Hardware Setup Guide for USB Camera & Microphones

## When you connect your hardware:

### 1. USB Camera with Built-in Microphone
- Connect USB camera
- The system will automatically detect it as `/dev/video0`
- Built-in microphone will be available as an audio input device

### 2. Additional USB Microphone
- Connect standalone USB microphone
- This will give you redundant audio input
- Better wake word detection with multiple mics

### 3. After connecting hardware, run:

```bash
# Activate virtual environment
source /home/gonzo/robot_ai_env/bin/activate

# Check connected devices
lsusb  # See USB devices
ls /dev/video*  # Check camera devices
arecord -l  # List audio input devices

# Test the system
python test_minimal.py
```

### 4. Update your Groq API Key
Edit the `.env` file and replace `your_groq_api_key_here` with your actual API key:
```bash
nano .env
```

### 5. Test with full hardware
```bash
# Run with normal config (will detect camera/mics)
python main.py

# Or run specific tests
python main.py --test
```

## Expected Hardware Detection:
- ✓ SIM7600X (already working)
- ✓ Camera (/dev/video0)
- ✓ Microphone 1 (built-in to camera)
- ✓ Microphone 2 (standalone USB)
- ✗ ESP32 (not connected)
- ✗ Hailo (not connected)

## Features that will work:
- 🎥 Face recognition and detection
- 🎤 Speech recognition and wake word
- 🗣️ Text-to-speech responses
- 🤖 AI conversation with Groq
- 🌐 Internet connectivity via SIM7600X
- 🔍 Web search integration

The system is now ready for full operation once you connect the camera and microphones!