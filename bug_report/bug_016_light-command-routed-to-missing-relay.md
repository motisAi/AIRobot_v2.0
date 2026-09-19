# Bug #016 — "Turn on the light" routed to a non-existent microcontroller relay

- **Date found:** 2026-09-12
- **Status:** fixed
- **Area:** smart-home
- **Files touched:** `modules/smart_home/tuya_devices.py`, `modules/conversation/manager.py`, `modules/ai/ai_engine.py`
- **Commit(s):** e2cc602

## Symptom
Saying "turn on the light" did nothing to the Tuya plug; Stella tried to drive a wired-MCU relay that does not exist.

## Root cause
The device lookup only matched a device literally named "light". The only real device was the Tuya plug named "Smart Plug", so the request fell through the Tuya -> MQTT -> wired-MCU order to the MCU path.

## Fix
Tuya `_match` gives a **single** configured device generic on/off names (light, lamp, socket, switch, relay1, power, outlet, device). `control_device` tool description made intent-based and a flexible `set_ac` tool added so meaning ("it is hot", "kill the light") is acted on, not exact keywords.

## How to verify
Say "turn on the light" / "kill the light" — the LSPA8 plug toggles; the journal shows the Tuya path, not the MCU.

## Will it come back?
Yes if a second Tuya device is added — the generic-name shortcut only applies while there is exactly one device; then name the devices explicitly. Also breaks if the plug's DHCP IP changes (re-run the tinytuya scan or reserve the IP).
