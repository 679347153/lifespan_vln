#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${REMOTE_VISION_SSH_HOST:-7.216.187.6}"
SSH_PORT="${REMOTE_VISION_SSH_PORT:-30180}"
SSH_USER="${REMOTE_VISION_SSH_USER:-root}"
LOCAL_PORT="${REMOTE_VISION_LOCAL_PORT:-50220}"
REMOTE_PORT="${REMOTE_VISION_REMOTE_PORT:-8010}"

if ! command -v sshpass >/dev/null 2>&1; then
  echo "sshpass not found. Install it on the workstation first: sudo apt install sshpass" >&2
  exit 1
fi

if [ -z "${SSHPASS:-}" ]; then
  echo "SSHPASS is empty. Example: export SSHPASS='<server-password>'" >&2
  exit 1
fi

exec sshpass -e ssh \
  -N \
  -o ExitOnForwardFailure=yes \
  -o StrictHostKeyChecking=no \
  -p "${SSH_PORT}" \
  -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
  "${SSH_USER}@${SSH_HOST}"
