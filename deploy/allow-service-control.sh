#!/bin/bash
# Run ONCE:  sudo bash deploy/allow-service-control.sh
# Lets the robot user restart/stop/start the 'airobot' service without a
# password, so code updates can be applied without a sudo prompt each time.
set -e
U="${SUDO_USER:-moti_ai}"
SC="$(command -v systemctl)"
echo "$U ALL=(root) NOPASSWD: $SC restart airobot, $SC start airobot, $SC stop airobot, $SC status airobot" \
  > /etc/sudoers.d/stella-service
chmod 440 /etc/sudoers.d/stella-service
visudo -cf /etc/sudoers.d/stella-service >/dev/null
echo "OK. Robot user can now run: sudo -n $SC restart airobot"
