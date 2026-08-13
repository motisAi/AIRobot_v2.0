#!/bin/bash
# Install Gonzo as a systemd service so it auto-starts on boot and restarts
# if it crashes. Run this once:  bash deploy/install_service.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "Stopping any manually-running robot first..."
pkill -f "python main.py" 2>/dev/null || true
sleep 1

echo "Installing service (needs sudo)..."
sudo cp "$HERE/airobot.service" /etc/systemd/system/airobot.service
sudo systemctl daemon-reload
sudo systemctl enable airobot
sudo systemctl restart airobot

echo
echo "Done. Gonzo will now start on boot."
echo "  Status:  systemctl status airobot"
echo "  Live log: journalctl -u airobot -f"
echo "  Stop:    sudo systemctl stop airobot"
echo "  Dashboard: http://$(hostname -I | awk '{print $1}'):5000"
