#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${REMOTE_VISION_SSH_HOST:-7.216.187.6}"
SSH_PORT="${REMOTE_VISION_SSH_PORT:-30180}"
SSH_USER="${REMOTE_VISION_SSH_USER:-root}"
LOCAL_PORT="${REMOTE_VISION_LOCAL_PORT:-50220}"
REMOTE_HOST="${REMOTE_VISION_REMOTE_HOST:-127.0.0.1}"
REMOTE_PORT="${REMOTE_VISION_REMOTE_PORT:-8010}"
SSH_DEBUG="${REMOTE_VISION_SSH_DEBUG:-0}"

if ! command -v sshpass >/dev/null 2>&1; then
  echo "sshpass not found. Install it on the workstation first: sudo apt install sshpass" >&2
  exit 1
fi

if [ -z "${SSHPASS:-}" ]; then
  echo "SSHPASS is empty. Example: export SSHPASS='<server-password>'" >&2
  exit 1
fi

echo "Starting SSH tunnel: 127.0.0.1:${LOCAL_PORT} -> ${SSH_USER}@${SSH_HOST}:${SSH_PORT} -> ${REMOTE_HOST}:${REMOTE_PORT}" >&2
echo "Keep this terminal open while using the tunnel. Test from another terminal with:" >&2
echo "  curl http://127.0.0.1:${LOCAL_PORT}/health" >&2

SSH_ARGS=(
  -N \
  -o ExitOnForwardFailure=yes \
  -o StrictHostKeyChecking=no \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -p "${SSH_PORT}" \
  -L "127.0.0.1:${LOCAL_PORT}:${REMOTE_HOST}:${REMOTE_PORT}" \
  "${SSH_USER}@${SSH_HOST}"
)

if [[ "${SSH_DEBUG}" == "1" ]]; then
  SSH_ARGS=(-v "${SSH_ARGS[@]}")
fi

exec sshpass -e ssh "${SSH_ARGS[@]}"
