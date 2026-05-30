"""Write ALSA config file for Pi audio output via HDMI."""
import os

asoundrc = os.path.expanduser("~/.asoundrc")

config = """pcm.!default {
  type asym
  playback.pcm {
    type plug
    slave.pcm "hw:1,0"
  }
  capture.pcm {
    type plug
    slave.pcm "hw:0,0"
  }
}
ctl.!default {
  type hw
  card 1
}
"""

with open(asoundrc, "w") as f:
    f.write(config)

print(f"Written {asoundrc}")
print(open(asoundrc).read())
