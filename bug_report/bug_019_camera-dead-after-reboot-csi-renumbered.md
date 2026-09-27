# Bug #019 — Camera "off" after reboot: CSI camera renumbered the USB webcam

- **Date found:** 2026-09-12
- **Status:** fixed
- **Area:** vision
- **Files touched:** `parts_used/camera_usb.py`
- **Commit(s):** e2cc602

## Symptom
After a reboot: "Camera opened but test read failed", no face detection, Stella did not react to anyone in front of her.

## Root cause
A Pi CSI camera plugged into the CSI slot claimed `/dev/video0-7` (rp1-cfe-csi2 / pispbe nodes, not readable by OpenCV), pushing the working USB "Auto Focus Camera" from video0 to video8. The hard-coded config index 0 then opened a CSI node that never yields a cv2 frame.

## Fix
`camera_manager.py` now auto-detects the USB webcam: probes indices in order [configured, USB-UVC-by-/sys-name, 0..15], skips CSI/ISP nodes (rp1-cfe/pispbe/rpivid), opens with MJPG, and warm-reads up to ~2.5 s (USB cams fail the first reads). `_reopen` re-probes too. Verified: auto-detected index 8, "Face detected: Moti".

## How to verify
```bash
journalctl -u airobot -n 200 | grep -i camera      # shows the detected index and a successful test read
```
Reboot and repeat — the index may change but the camera must still open.

## Will it come back?
No for node renumbering. The CSI camera itself is unusable via cv2; it needs libcamera/picamera2 if ever wired up (currently unplugged).
