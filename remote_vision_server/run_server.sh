#!/usr/bin/env bash
set -euo pipefail

export REMOTE_VISION_REPO_ROOT="${REMOTE_VISION_REPO_ROOT:-$(pwd)}"
export REMOTE_VISION_DEVICE="${REMOTE_VISION_DEVICE:-cuda}"
export REMOTE_VISION_HOST="${REMOTE_VISION_HOST:-127.0.0.1}"
export REMOTE_VISION_PORT="${REMOTE_VISION_PORT:-8010}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ -z "${BERT_BASE_UNCASED_PATH:-}" ]]; then
  if [[ -d "${REMOTE_VISION_REPO_ROOT}/../bert-base-uncased" ]]; then
    export BERT_BASE_UNCASED_PATH="${REMOTE_VISION_REPO_ROOT}/../bert-base-uncased"
  elif [[ -n "${HF_HOME:-}" && -d "${HF_HOME}/bert-base-uncased" ]]; then
    export BERT_BASE_UNCASED_PATH="${HF_HOME}/bert-base-uncased"
  elif [[ -n "${TRANSFORMERS_CACHE:-}" && -d "${TRANSFORMERS_CACHE}/bert-base-uncased" ]]; then
    export BERT_BASE_UNCASED_PATH="${TRANSFORMERS_CACHE}/bert-base-uncased"
  fi
fi

exec uvicorn remote_vision_server.server:app \
  --host "${REMOTE_VISION_HOST}" \
  --port "${REMOTE_VISION_PORT}"
