#!/usr/bin/env bash
# nightly.sh — Stella's 03:00 self-check: manifest -> Guardian -> Scout -> morning report.
# Pauses (exit 0, logged) when the operator is working or the Pi is busy.
set -u
cd "$(dirname "$0")/.." || exit 2
PY=venv/bin/python
LOG=evolution/reports/nightly.log
mkdir -p evolution/reports
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# --- pause conditions -------------------------------------------------------
if [ -n "$(who 2>/dev/null)" ]; then log "skip: SSH session open (operator active)"; exit 0; fi
if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]; then log "skip: git tree dirty (work in progress)"; exit 0; fi
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge 2 ] || { log "skip: disk ${FREE_GB}GB free"; exit 0; }
LOAD=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
[ "${LOAD:-0}" -lt 3 ] || { log "skip: load ${LOAD}"; exit 0; }

log "start"
nice -n 15 $PY evolution/manifest.py >>"$LOG" 2>&1 || log "manifest failed"
nice -n 15 $PY evolution/guardian.py  >>"$LOG" 2>&1 || log "guardian UNHEALTHY"
nice -n 15 timeout 900 $PY evolution/scout.py >>"$LOG" 2>&1 || log "scout failed"
nice -n 15 $PY evolution/report.py    >>"$LOG" 2>&1 || log "report failed"
log "done"
