#!/bin/bash
# One-time setup so Stella can scan/add WiFi networks.
# Run:  sudo bash deploy/install-wifi-helper.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
HELPER=/usr/local/bin/stella-wifi-helper.sh
USER_NAME="${SUDO_USER:-moti_ai}"

install -D -o root -g root -m 0755 "$HERE/wifi-helper.sh" "$HELPER"

# Allow ONLY this script to run as root without a password (safe, scoped).
echo "$USER_NAME ALL=(root) NOPASSWD: $HELPER" > /etc/sudoers.d/stella-wifi
chmod 440 /etc/sudoers.d/stella-wifi
visudo -cf /etc/sudoers.d/stella-wifi >/dev/null

echo "Installed $HELPER and sudoers rule for '$USER_NAME'."
echo "Test:  sudo -n $HELPER scan"
