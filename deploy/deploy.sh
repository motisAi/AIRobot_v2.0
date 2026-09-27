#!/usr/bin/env bash
# deploy.sh — restart Stella, run Guardian, roll back automatically on failure.
#
#   deploy/deploy.sh                # deploy HEAD; roll back to HEAD~1 if unhealthy
#   deploy/deploy.sh <good-commit>  # deploy HEAD; roll back to <good-commit>
#   deploy/deploy.sh --no-rollback  # just restart + Guardian
#
# Rules: the working tree must be clean (commit first — that IS the rollback
# point).  Only `sudo -n systemctl restart/stop airobot` is passwordless on
# the Pi, so nothing here needs a password.
set -u
cd "$(dirname "$0")/.." || exit 2
PY=venv/bin/python
ROLLBACK_TO=""
NO_ROLLBACK=0
for arg in "$@"; do
  case "$arg" in
    --no-rollback) NO_ROLLBACK=1 ;;
    *) ROLLBACK_TO="$arg" ;;
  esac
done

notify() {  # Telegram one-liner; silent if not configured
  local token chat
  token=$(grep -E '^TELEGRAM_TOKEN=' .env 2>/dev/null | cut -d= -f2-)
  chat=$(grep -E '^TELEGRAM_CHAT_ID=' .env 2>/dev/null | cut -d= -f2-)
  [ -n "$token" ] && [ -n "$chat" ] || return 0
  curl -s -m 10 "https://api.telegram.org/bot${token}/sendMessage" \
       --data-urlencode "chat_id=${chat}" --data-urlencode "text=$1" >/dev/null 2>&1 || true
}

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "deploy: working tree has uncommitted changes — commit first (that is the rollback point)."
  git status --short --untracked-files=no
  exit 2
fi

HEAD_SHA=$(git rev-parse --short HEAD)
[ -z "$ROLLBACK_TO" ] && ROLLBACK_TO=$(git rev-parse --short HEAD~1)

echo "deploy: restarting airobot at $HEAD_SHA (rollback target: $ROLLBACK_TO)"
sudo -n systemctl restart airobot || { echo "deploy: restart failed"; exit 1; }

if $PY evolution/guardian.py --wait; then
  echo "deploy: $HEAD_SHA HEALTHY"
  notify "Stella deploy $HEAD_SHA: healthy ✅"
  exit 0
fi

echo "deploy: $HEAD_SHA UNHEALTHY"
if [ "$NO_ROLLBACK" = 1 ]; then
  notify "Stella deploy $HEAD_SHA: UNHEALTHY ❌ (no rollback requested)"
  exit 1
fi

echo "deploy: rolling back to $ROLLBACK_TO"
git reset --hard "$ROLLBACK_TO" || exit 1
sudo -n systemctl restart airobot
if $PY evolution/guardian.py --wait; then
  notify "Stella deploy $HEAD_SHA FAILED Guardian — rolled back to $ROLLBACK_TO, healthy again."
  echo "deploy: rolled back to $ROLLBACK_TO, healthy"
else
  notify "Stella deploy $HEAD_SHA FAILED and rollback to $ROLLBACK_TO is ALSO unhealthy — check her!"
  echo "deploy: rollback ALSO unhealthy — manual attention needed"
fi
exit 1
