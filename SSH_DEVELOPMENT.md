# SSH Development Configuration
# ===========================

## Running AI Robot over SSH

Since you're controlling via SSH, we need to modify the system to work headlessly:

### 1. Disable GUI Components
```python
# In settings, set these for headless operation:
HEADLESS_MODE = True
SHOW_CAMERA_WINDOW = False
SHOW_DEBUG_WINDOWS = False
```

### 2. Camera Operation
- Captures saved to `/home/gonzo/robot_ai_env/data/captures/`
- Face recognition results logged to console
- No preview windows

### 3. Audio Configuration
```bash
# Check audio devices available on Pi
aplay -l    # List audio output devices
arecord -l  # List audio input devices

# Set default audio device for Pi (not SSH session)
sudo raspi-config  # Audio options
```

### 4. Web Interface for Monitoring
The system includes a web server at `http://pi-ip:5000` where you can:
- View camera captures
- See system status
- Monitor AI responses
- Check logs

### 5. Testing Commands for SSH
```bash
# Test without GUI
python main.py --headless

# Test with file output only
python main.py --test --no-gui

# Check system status
python test_minimal.py
```

### 6. Remote Desktop Alternative
If you need GUI access:
```bash
# Install VNC server on Pi
sudo apt install realvnc-vnc-server
sudo systemctl enable vncserver-x11-serviced
sudo systemctl start vncserver-x11-serviced

# Then connect via VNC viewer from your computer
```

The AI Robot system will work great over SSH - just without live camera windows!