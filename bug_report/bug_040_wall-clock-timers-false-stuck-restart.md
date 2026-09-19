# Bug #040 — Wall-clock timers caused a false "stuck conversation" restart after an NTP time jump

- **Date found:** 2026-09-16
- **Status:** fixed
- **Area:** hardware
- **Files touched:** `core/watchdog.py`, `modules/conversation/manager.py`
- **Commit(s):** 231740d

## Symptom
The watchdog restarted the service claiming a conversation was stuck, although nothing was hung — typically shortly after the Pi got internet back.

## Root cause
Conversation-activity and watchdog timers used `time.time()` (wall clock). A Pi booted offline has a wrong clock; when NTP later syncs, the clock steps by hours, so "silent for 75 s" was instantly exceeded. The watchdog also did not know when the brain was legitimately busy thinking.

## Fix
Watchdog and conversation stamps use the monotonic clock; the watchdog skips checks while the `_thinking` flag is set.

## How to verify
Boot offline, then plug in the uplink so NTP syncs: no watchdog restart in `journalctl -u airobot | grep -i watchdog`; `systemctl show airobot -p NRestarts` unchanged.

## Will it come back?
No, as long as new timers use `time.monotonic()`.
