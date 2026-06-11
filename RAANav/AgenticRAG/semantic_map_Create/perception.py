"""Compatibility entry for GroundingDINO + MobileSAM perception.

The RAANav main line uses the DAAAM-style object-node frontend, but legacy
Habitat exploration/navigation scripts still call `get_detector()` here. This
module now supports two backends:

- remote: workstation-side proxy to `remote_vision_server` through HTTP/SSH.
- local: old in-process GroundingDINO + MobileSAM implementation.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional


_THIS = Path(__file__).resolve()
_AGENTIC_RAG_ROOT = _THIS.parents[1]
_REPO_ROOT = _THIS.parents[3]
for _path in [str(_REPO_ROOT), str(_AGENTIC_RAG_ROOT)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


_singleton: Optional[Any] = None


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _select_backend(explicit: Optional[str] = None) -> str:
    backend = (
        explicit
        or os.environ.get("RAANAV_PERCEPTION_BACKEND")
        or os.environ.get("REMOTE_VISION_BACKEND")
        or ""
    ).strip().lower()
    if backend in {"remote", "server", "ssh"}:
        return "remote"
    if backend in {"local", "legacy"}:
        return "local"
    if os.environ.get("REMOTE_VISION_BASE_URL") or _truthy_env("REMOTE_VISION_USE_SSH_TUNNEL"):
        return "remote"
    return "local"


def get_detector(**kwargs: Any) -> Any:
    """Return a detector compatible with the legacy OpenVocabDetector API.

    Remote backend can be enabled with either:

    ```bash
    export RAANAV_PERCEPTION_BACKEND=remote
    export REMOTE_VISION_BASE_URL=http://127.0.0.1:50220
    ```

    or automatic SSH tunnel:

    ```bash
    export RAANAV_PERCEPTION_BACKEND=remote
    export REMOTE_VISION_USE_SSH_TUNNEL=1
    export REMOTE_VISION_SSH_PASSWORD='<server-password>'
    ```
    """
    global _singleton
    backend = _select_backend(kwargs.pop("backend", None) or kwargs.pop("mode", None))
    if _singleton is not None:
        return _singleton

    if backend == "remote":
        remote_kwargs = {
            key: kwargs.pop(key)
            for key in list(kwargs.keys())
            if key
            in {
                "base_url",
                "request_timeout",
                "use_ssh_tunnel",
                "ssh_host",
                "ssh_port",
                "ssh_user",
                "ssh_password",
                "local_port",
                "remote_host",
                "remote_port",
                "health_timeout",
            }
        }
        from remote_vision_server.client import RemoteOpenVocabDetector

        _singleton = RemoteOpenVocabDetector(**remote_kwargs)
        return _singleton

    from scripts.z_legacy.legacy_gdino_frontend.perception_gdino_mobilesam import (
        get_detector as get_local_detector,
    )

    _singleton = get_local_detector(**kwargs)
    return _singleton


def build_text_prompt(labels: list[str]) -> str:
    return " . ".join(labels)
