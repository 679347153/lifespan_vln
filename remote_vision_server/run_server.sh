#!/usr/bin/env bash
set -euo pipefail

export REMOTE_VISION_REPO_ROOT="${REMOTE_VISION_REPO_ROOT:-$(pwd)}"
export REMOTE_VISION_DEVICE="${REMOTE_VISION_DEVICE:-cuda}"
export REMOTE_VISION_HOST="${REMOTE_VISION_HOST:-127.0.0.1}"
export REMOTE_VISION_PORT="${REMOTE_VISION_PORT:-8010}"

exec uvicorn remote_vision_server.server:app \
  --host "${REMOTE_VISION_HOST}" \
  --port "${REMOTE_VISION_PORT}"
