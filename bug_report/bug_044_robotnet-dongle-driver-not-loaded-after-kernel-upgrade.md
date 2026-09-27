# Bug #044 — RobotNet AP dead: RTL8821AU DKMS module missing after kernel upgrade

- **Date found:** 2026-09-19 (failing since the 2026-09-14 boot)
- **Status:** recurring
- **Area:** network
- **Files touched:** none in repo (system: `/etc/modules-load.d/8821au.conf`, `robotnet-*.service`)
- **Commit(s):** n/a

## Symptom
No "RobotNet" WiFi visible. `robotnet-iface.service` failed at boot ("no dongle"); `lsusb` shows the TP-Link Archer T2U Plus but no `wlx…` interface; `systemd-modules-load` logged "Failed to find module '8821au'".

## Root cause
Same mechanism as Bug #011: the kernel was upgraded to 6.8.0-1064 and the out-of-tree `8821au` DKMS module was not rebuilt, so the dongle had no driver. The 2026-09-19 `dkms autoinstall` (run for Hailo) rebuilt it, but a module built after boot is not loaded until `modprobe` or a reboot.

## Fix
```bash
sudo modprobe 8821au && sudo systemctl restart robotnet-iface robotnet-hostapd robotnet-dnsmasq
```
or simply reboot. With `linux-headers-raspi` installed (2026-09-19) future kernel upgrades rebuild both DKMS modules automatically.

## How to verify
`ip -br link | grep wlx` shows the dongle up with 10.0.0.1; the phone sees "RobotNet".

## Will it come back?
Only if a kernel upgrade lands without headers again. Guardian now warns when the dongle is present but has no interface.
