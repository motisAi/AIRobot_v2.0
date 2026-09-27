# Bug #020 — Console filled with "rp1-cfe ... csi2_chN node link is not enabled" kernel lines

- **Date found:** 2026-09-12
- **Status:** fixed
- **Area:** vision
- **Files touched:** `parts_used/camera_usb.py` (`_candidate_indices`)
- **Commit(s):** e2cc602

## Symptom
The Pi's HDMI console showed repeating alarming "rp1-cfe 1f00128000.csi: csi2_chN node link is not enabled" messages.

## Root cause
The CSI camera was plugged in but had no libcamera pipeline; opening its `/dev/video0-7` nodes makes the kernel print that line. The camera probe tried the configured index (0 = a CSI node) first on every camera start, so each service start emitted a line. The apparent flood was mostly that day's repeated debugging restarts (`NRestarts=0`, not a crash loop).

## Fix
`_candidate_indices` excludes CSI/ISP nodes entirely by `/sys` name; candidate order is USB webcams first, CSI never opened. Verified: candidates `[8,9,10,...]`, zero new kernel messages across a restart. Existing lines on the console are static — `clear` or a reboot wipes them.

## How to verify
```bash
sudo -n systemctl restart airobot
dmesg | tail -20 | grep -c rp1-cfe      # expect 0 new lines
```

## Will it come back?
No while CSI nodes are excluded. Irrelevant now that the CSI camera is unplugged.
