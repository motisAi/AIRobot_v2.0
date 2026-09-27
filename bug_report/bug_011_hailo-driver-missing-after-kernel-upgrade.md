# Bug #011 — Hailo PCIe driver gone after kernel upgrade: no /dev/hailo0, offline brain dead

- **Date found:** 2026-09-05 (recurred 2026-09-16)
- **Status:** recurring
- **Area:** hardware
- **Files touched:** none in the repo (system: DKMS `hailo1x_pci`, kernel headers); documented in `docs/hardware/hailo.md`
- **Commit(s):** n/a (docs in e2cc602)

## Symptom
Offline LLM answers fail (hailo-ollama returns 500 / `HAILO_OUT_OF_PHYSICAL_DEVICES`, "Failed to create VDevice"); when the cloud is also down Stella is slow and dumb. `ls /dev/hailo*` finds nothing; `lsmod | grep hailo` is empty.

## Root cause
`hailo1x_pci` is a DKMS module. Ubuntu auto-upgraded the kernel (1057 -> 1060 in September, then 1060 -> 1064) and DKMS did not rebuild the module for the new kernel. On 2026-09-16 the deeper cause was found: the `linux-headers-raspi` metapackage was never installed, so DKMS never had headers for the new kernel and could not auto-rebuild.

## Fix
Run on the Pi (needs sudo + internet):
```bash
sudo apt-get update
sudo apt-get install -y linux-headers-raspi linux-headers-$(uname -r)
sudo dkms autoinstall
sudo modprobe hailo1x_pci
sudo systemctl restart hailo-ollama
sudo -n systemctl restart airobot
```
Installing the `linux-headers-raspi` metapackage is what makes future upgrades self-heal. Optional: `sudo apt-mark hold linux-image-raspi linux-headers-raspi`.

## How to verify
```bash
ls -la /dev/hailo0
hailortcli fw-control identify          # Device Architecture: HAILO10H
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-instruct:1.5b","messages":[{"role":"user","content":"hi"}],"stream":false}'
```

## Will it come back?
Yes — on every kernel upgrade until `linux-headers-raspi` is installed (the 2026-09-16 command above was still pending Moti's sudo at last log). The RobotNet dongle driver (morrownr `8821au`) is also DKMS and breaks the same way.
