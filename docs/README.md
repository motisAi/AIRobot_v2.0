# Stella — Project Documentation

This folder is the long-term memory of the **Stella** robot project (formerly "Gonzo").
It lives inside the git repo (`AIRobot_v2.0/docs/`) so it is versioned and pushed to
GitHub alongside the code — design decisions, wiring, and briefs stay understandable
in the future even if a PC is wiped.

## How this is organized

| Folder | What goes here |
|--------|----------------|
| `briefs/` | Incoming design/product briefs and ideas (e.g. new subsystems). One file per idea. |
| `hardware/` | Physical build: component list (BOM), wiring/connection maps, power, mechanical notes. |
| `firmware/` | Microcontroller firmware design: the ESP32 hand, serial/I2C protocols, pin maps. |
| `schematics/` | Diagrams — system overview, wiring schematics, data-flow. Mermaid (renders on GitHub) + ASCII. |
| `decisions/` | Dated design-decision log: what we chose, why, and what we rejected. |

| `architecture/` | Dated architecture documents. **Current:** [stella-architecture-2026-09-19.md](architecture/stella-architecture-2026-09-19.md) (layout, Guardian, self-evolution) on top of [stella-architecture-2026-09-13.md](architecture/stella-architecture-2026-09-13.md) (capability stack, Hailo/NVIDIA plan, roadmap). |

Pre-existing docs kept in place: `architecture.md` (older overview), `GONZO_GUIDE.md`, `rpi5_setup.md`,
`service_accounts.txt`.

Related folders outside `docs/`: the code map is the root [README.md](../README.md); every bug ever
fixed is in [`bug_report/`](../bug_report/README.md); the safety net and nightly self-check are in
[`evolution/`](../evolution/README.md).

## Workflow for adding files

1. Drop a file anywhere on the PC (Downloads is fine) and tell Claude the filename.
2. Claude files it into the right subfolder here, adds a summary, and links it from this index.
3. It goes to GitHub on the next push.

When new hardware is wired up, tell Claude the connections — the wiring map and a schematic
under `hardware/` + `schematics/` get updated so nothing is lost.

## How devices find the Pi: `motiAi.local` (mDNS)

Every device on a network has an IP address (like `192.168.11.204`), but those **change**
when you switch networks or the router hands out a new one. Hardcoding an IP means editing
configs every time you move — annoying and fragile.

**mDNS** (multicast DNS) fixes this. It lets a device answer to a **name** instead of an IP,
with no central server to set up. In plain terms:

- The Pi runs a tiny service (**avahi**) that announces on the local network:
  *"Hi, I'm `motiAi` — reach me at whatever my current IP is."*
- When a device wants the Pi, it asks the whole network at once (a *multicast* question):
  *"Who is `motiAi.local`?"* The Pi hears it and replies with its current IP.
- The `.local` ending is the signal to use mDNS instead of normal internet DNS.

So **`motiAi.local` just means "the Pi, wherever it is right now."**
- Home → it resolves to `192.168.11.x`; work → `192.168.1.x`. **Same name, you change nothing.**
- Windows, macOS, iPhones and most Androids speak mDNS out of the box.

**The one catch:** mDNS relies on multicast, which some locked-down networks (corporate/guest
WiFi) block. That's why the face controller has a **fallback**: if `motiAi.local` doesn't
answer, it scans the local network for the Pi and remembers where it found it. Either way you
never touch an IP.

*(The name comes from the Pi's hostname, `motiAi`. Rename the host and the `.local` name
follows.)*

## Index

- **Briefs**
  - [Robot face tablet subsystem](briefs/robot-face-tablet-brief.md) — add a tablet "face"
    (animated eyes/mouth, gaze, lip-sync) over a WebSocket bridge. *Not built yet — proposal.*
- **Hardware**
  - [Wiring & connection map](hardware/wiring.md)
  - [Bill of materials](hardware/bom.md)
- **Firmware**
  - [ESP32-S3 robotic hand](firmware/hand-esp32.md) — serial protocol, I2C, servo/finger map.
- **Schematics**
  - [System overview](schematics/system-overview.md)
- **Architecture**
  - [2026-09-19 — layout, Guardian, self-evolution](architecture/stella-architecture-2026-09-19.md) (current)
  - [2026-09-13 — capability stack, Hailo + NVIDIA plan, roadmap](architecture/stella-architecture-2026-09-13.md)
- **Decisions**
  - [Decision log](decisions/log.md)
- **Bugs** — [bug_report/](../bug_report/README.md) (one file per fixed bug)
