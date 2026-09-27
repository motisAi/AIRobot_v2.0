#!/bin/bash
# Root helper for Stella's WiFi assistant.
# Installed to /usr/local/bin and allowed (only this script) via sudoers for the
# robot user. Actions:  scan  |  add <ssid> <password>
#
# 'add' is intentionally ADDITIVE: it enables the new network as a candidate but
# does NOT tear down the current connection, so a wrong password can't lock you
# out of a WiFi-only Pi.
set -uo pipefail
IFACE="${WIFI_IFACE:-wlan0}"
cmd="${1:-}"

case "$cmd" in
  scan)
    wpa_cli -i "$IFACE" scan >/dev/null 2>&1
    sleep 2
    # Emit unique, non-hidden SSIDs (SSID is column 5+ of scan_results).
    wpa_cli -i "$IFACE" scan_results 2>/dev/null \
      | awk 'NR>1 { s=""; for (i=5;i<=NF;i++) s=s (i>5?" ":"") $i; if (s!="") print s }' \
      | sort -u
    ;;
  add)
    ssid="${2:-}"; psk="${3:-}"
    [ -z "$ssid" ] && { echo "ERROR: no ssid"; exit 1; }
    id="$(wpa_cli -i "$IFACE" add_network | tail -1)"
    wpa_cli -i "$IFACE" set_network "$id" ssid "\"$ssid\"" >/dev/null
    if [ -n "$psk" ]; then
      wpa_cli -i "$IFACE" set_network "$id" psk "\"$psk\"" >/dev/null
    else
      wpa_cli -i "$IFACE" set_network "$id" key_mgmt NONE >/dev/null
    fi
    wpa_cli -i "$IFACE" enable_network "$id" >/dev/null
    wpa_cli -i "$IFACE" save_config >/dev/null 2>&1
    echo "OK: added network '$ssid'"
    ;;
  *)
    echo "usage: $(basename "$0") scan | add <ssid> <password>"; exit 2 ;;
esac
