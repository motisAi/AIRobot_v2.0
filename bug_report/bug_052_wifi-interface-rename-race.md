# Bug #052 — Stella off home WiFi: onboard radio renamed wlan0 -> wlan1 (boot race)

- **Date found:** 2026-09-20 (she appeared "dead"; only RobotNet reachable)
- **Status:** fixed
- **Area:** network
- **Files touched:** `/etc/netplan/50-cloud-init.yaml` (system file, not in repo)
- **Commit(s):** n/a (system config; documented here)

## Symptom
Stella unreachable on the home network and not on any home IP; her RobotNet AP (10.0.0.1, USB dongle) still worked. `iw dev` showed the onboard radio as `wlan1`, no `wlan0`; `sudo iw dev wlan0 scan` -> "No such device (-19)".

## Root cause
Boot-order race between the two WiFi radios. Kernel log: `rtl8821au ... wlx984827df22bb: renamed from wlan0` — the USB dongle enumerated first and took `wlan0`, then udev renamed it to its predictable `wlx...` name, leaving the onboard brcmfmac as `wlan1`. The netplan only configured `wlan0`, so the onboard never joined home WiFi. Nondeterministic: worked for weeks, then this boot the order flipped. Not a power/undervoltage event (throttled=0x0 this boot).

## Fix
Configure BOTH `wlan0` and `wlan1` in `/etc/netplan/50-cloud-init.yaml` (same home AP, `optional: true`, DHCP), so whichever name the onboard radio gets, it connects. netplan's networkd backend does NOT support `match: macaddress` for wifis (only by interface name), so MAC-matching is not an option; dual-name is the working approach. Applied with `sudo netplan apply` (no reboot). Result: onboard came up as wlan1 at 192.168.11.204, RobotNet unchanged.

## How to verify
`ip -br addr | grep 192.168.11` shows the onboard WiFi with a home IP; `iw dev <name> link` shows SSID "Arik -2.4G-ext". Survives a reboot regardless of which name the onboard gets.

## Will it come back?
No — both names are covered. A permanent alternative is a systemd .link file pinning the onboard MAC (88:a2:9e:ab:1a:d9) to a fixed name; not needed now. Related, still-open: the Ethernet-cable brownout power-offs are a separate issue.
