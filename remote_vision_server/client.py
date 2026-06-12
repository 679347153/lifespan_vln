from __future__ import annotations

import atexit
import base64
import io
import json
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - handled at runtime with a clear message.
    requests = None  # type: ignore[assignment]

try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover - handled at runtime with a clear message.
    Image = None  # type: ignore[assignment]


def _require_workstation_deps() -> None:
    missing = []
    if requests is None:
        missing.append("requests")
    if Image is None:
        missing.append("pillow")
    if missing:
        raise RuntimeError(f"Missing workstation dependency: {', '.join(missing)}. Run: pip install requests pillow")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _image_to_data_url(rgb: np.ndarray) -> str:
    _require_workstation_deps()
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb[:, :, :3], mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _decode_mask_png(data_url: str) -> np.ndarray:
    _require_workstation_deps()
    raw = data_url
    if "," in raw and raw.strip().lower().startswith("data:"):
        raw = raw.split(",", 1)[1]
    data = base64.b64decode(raw)
    mask = Image.open(io.BytesIO(data)).convert("L")
    return np.asarray(mask) > 0


def _raise_for_remote_error(response: Any) -> None:
    if response.status_code < 400:
        return
    body = response.text
    try:
        parsed = response.json()
        body = json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception:
        pass
    if len(body) > 4000:
        body = body[:4000] + "\n... <truncated>"
    raise RuntimeError(
        f"Remote vision request failed: HTTP {response.status_code} {response.url}\n{body}"
    )


@dataclass
class RemoteVisionConfig:
    base_url: Optional[str] = None
    request_timeout: float = 120.0
    use_ssh_tunnel: bool = False
    ssh_host: str = "7.216.187.6"
    ssh_port: int = 30180
    ssh_user: str = "root"
    ssh_password: Optional[str] = None
    local_port: int = 0
    remote_host: str = "127.0.0.1"
    remote_port: int = 8010
    health_timeout: float = 30.0
    return_clip_embedding: bool = False
    clip_mode: str = "mask_only"
    clip_masked_weight: float = 0.75
    clip_bbox_padding: int = 20
    clip_model: Optional[str] = None
    clip_local_files_only: Optional[bool] = None

    @classmethod
    def from_env(cls, **overrides: Any) -> "RemoteVisionConfig":
        clip_local_files_only: Optional[bool]
        if "REMOTE_VISION_CLIP_LOCAL_FILES_ONLY" in os.environ:
            clip_local_files_only = _env_bool("REMOTE_VISION_CLIP_LOCAL_FILES_ONLY", True)
        else:
            clip_local_files_only = None
        config = cls(
            base_url=os.environ.get("REMOTE_VISION_BASE_URL"),
            request_timeout=float(os.environ.get("REMOTE_VISION_TIMEOUT", "120")),
            use_ssh_tunnel=_env_bool("REMOTE_VISION_USE_SSH_TUNNEL", False),
            ssh_host=os.environ.get("REMOTE_VISION_SSH_HOST", "7.216.187.6"),
            ssh_port=int(os.environ.get("REMOTE_VISION_SSH_PORT", "30180")),
            ssh_user=os.environ.get("REMOTE_VISION_SSH_USER", "root"),
            ssh_password=os.environ.get("REMOTE_VISION_SSH_PASSWORD") or os.environ.get("SSHPASS"),
            local_port=int(os.environ.get("REMOTE_VISION_LOCAL_PORT", "0")),
            remote_host=os.environ.get("REMOTE_VISION_REMOTE_HOST", "127.0.0.1"),
            remote_port=int(os.environ.get("REMOTE_VISION_REMOTE_PORT", "8010")),
            health_timeout=float(os.environ.get("REMOTE_VISION_HEALTH_TIMEOUT", "30")),
            return_clip_embedding=_env_bool("REMOTE_VISION_RETURN_CLIP", _env_bool("RAANAV_REMOTE_CLIP_EMBEDDING", False)),
            clip_mode=os.environ.get("REMOTE_VISION_CLIP_MODE", "mask_only"),
            clip_masked_weight=float(os.environ.get("REMOTE_VISION_CLIP_MASKED_WEIGHT", "0.75")),
            clip_bbox_padding=int(os.environ.get("REMOTE_VISION_CLIP_BBOX_PADDING", "20")),
            clip_model=os.environ.get("REMOTE_VISION_CLIP_MODEL_PATH") or os.environ.get("REMOTE_VISION_CLIP_MODEL"),
            clip_local_files_only=clip_local_files_only,
        )
        for key, value in overrides.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)
        return config


class SSHTunnel:
    def __init__(self, config: RemoteVisionConfig) -> None:
        self.config = config
        self.process: Optional[subprocess.Popen[str]] = None
        self.local_port = config.local_port or _find_free_port()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.local_port}"

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        if not self.config.ssh_password:
            raise RuntimeError("REMOTE_VISION_SSH_PASSWORD or SSHPASS is required for automatic SSH tunnel.")
        if shutil.which("sshpass") is None:
            raise RuntimeError("sshpass not found. Install it on the workstation first: sudo apt install sshpass")
        if shutil.which("ssh") is None:
            raise RuntimeError("ssh client not found on this workstation.")

        env = os.environ.copy()
        env["SSHPASS"] = self.config.ssh_password
        command = [
            "sshpass",
            "-e",
            "ssh",
            "-N",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-p",
            str(self.config.ssh_port),
            "-L",
            f"127.0.0.1:{self.local_port}:{self.config.remote_host}:{self.config.remote_port}",
            f"{self.config.ssh_user}@{self.config.ssh_host}",
        ]
        self.process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        atexit.register(self.close)

        deadline = time.time() + self.config.health_timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                stderr = ""
                if self.process.stderr is not None:
                    stderr = self.process.stderr.read()
                raise RuntimeError(f"SSH tunnel exited early: {stderr.strip()}")
            if _can_connect("127.0.0.1", self.local_port):
                return
            time.sleep(0.25)
        raise RuntimeError(
            f"SSH tunnel not ready: local 127.0.0.1:{self.local_port} -> "
            f"{self.config.remote_host}:{self.config.remote_port}"
        )

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


def _can_connect(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


class RemoteOpenVocabDetector:
    """Workstation-side proxy for the remote GroundingDINO + MobileSAM server."""

    def __init__(self, **kwargs: Any) -> None:
        self.config = RemoteVisionConfig.from_env(**kwargs)
        self._tunnel: Optional[SSHTunnel] = None
        if self.config.use_ssh_tunnel:
            self._tunnel = SSHTunnel(self.config)
            self._tunnel.start()
            self.base_url = self._tunnel.base_url
        else:
            self.base_url = self.config.base_url or "http://127.0.0.1:50220"
        self.base_url = self.base_url.rstrip("/")
        self.health_check()

    def close(self) -> None:
        if self._tunnel is not None:
            self._tunnel.close()

    def health_check(self) -> Dict[str, Any]:
        _require_workstation_deps()
        response = requests.get(f"{self.base_url}/health", timeout=min(10.0, self.config.request_timeout))
        response.raise_for_status()
        return response.json()

    def detect(
        self,
        rgb: np.ndarray,
        text_prompt: str,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        *,
        return_masks: bool = True,
        return_clip_embedding: Optional[bool] = None,
        clip_mode: Optional[str] = None,
        clip_masked_weight: Optional[float] = None,
        clip_bbox_padding: Optional[int] = None,
        clip_model: Optional[str] = None,
        clip_local_files_only: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        want_clip = self.config.return_clip_embedding if return_clip_embedding is None else bool(return_clip_embedding)
        payload: Dict[str, Any] = {
            "image_base64": _image_to_data_url(rgb),
            "text_prompt": text_prompt,
            "box_threshold": 0.35 if box_threshold is None else float(box_threshold),
            "text_threshold": 0.35 if text_threshold is None else float(text_threshold),
            "return_mask_png": bool(return_masks),
            "return_clip_embedding": want_clip,
        }
        if want_clip:
            payload.update(
                {
                    "clip_mode": clip_mode or self.config.clip_mode,
                    "clip_masked_weight": (
                        self.config.clip_masked_weight
                        if clip_masked_weight is None
                        else float(clip_masked_weight)
                    ),
                    "clip_bbox_padding": (
                        self.config.clip_bbox_padding
                        if clip_bbox_padding is None
                        else int(clip_bbox_padding)
                    ),
                }
            )
            selected_clip_model = clip_model or self.config.clip_model
            if selected_clip_model:
                payload["clip_model"] = selected_clip_model
            selected_local_only = (
                self.config.clip_local_files_only
                if clip_local_files_only is None
                else bool(clip_local_files_only)
            )
            if selected_local_only is not None:
                payload["clip_local_files_only"] = selected_local_only
        response = requests.post(
            f"{self.base_url}/v1/detect_segment",
            json=payload,
            timeout=self.config.request_timeout,
        )
        _raise_for_remote_error(response)
        data = response.json()
        detections = data.get("detections", [])
        for det in detections:
            mask_data = det.pop("mask_png_base64", None)
            if mask_data:
                det["mask"] = _decode_mask_png(mask_data)
        return detections

    def detect_with_depth(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        text_prompt: str,
        intrinsics: Any,
        agent_pos: np.ndarray,
        agent_heading_deg: float,
        max_depth: float = 5.0,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        detections = self.detect(
            rgb,
            text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            return_masks=True,
        )
        if not detections:
            return []

        from semantic_map_Create.occupancy_grid import camera_to_world

        for det in detections:
            mask = det.get("mask")
            bbox = det.get("bbox_xyxy", [0, 0, 0, 0])
            if mask is not None:
                mask_depths = depth[mask]
                valid_depths = mask_depths[(mask_depths > 0) & (mask_depths < max_depth)]
            else:
                valid_depths = np.asarray([], dtype=np.float32)

            if len(valid_depths) == 0:
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                cx = min(max(cx, 0), depth.shape[1] - 1)
                cy = min(max(cy, 0), depth.shape[0] - 1)
                d = float(depth[cy, cx])
                if d <= 0 or d >= max_depth:
                    det["pos_3d"] = None
                    det["pos_2d"] = None
                    det["depth_median"] = None
                    continue
                valid_depths = np.asarray([d], dtype=np.float32)

            median_d = float(np.median(valid_depths))
            if mask is not None and np.any(mask):
                ys, xs = np.where(mask)
                cu = float(np.mean(xs))
                cv = float(np.mean(ys))
            else:
                cu = float((bbox[0] + bbox[2]) / 2)
                cv = float((bbox[1] + bbox[3]) / 2)

            cam_x = (cu - intrinsics.cx) * median_d / intrinsics.fx
            cam_y = (cv - intrinsics.cy) * median_d / intrinsics.fy
            cam_z = median_d
            cam_point = np.asarray([[cam_x, cam_y, cam_z]], dtype=np.float32)
            world_point = camera_to_world(cam_point, agent_pos, agent_heading_deg)
            wx, wy, wz = float(world_point[0, 0]), float(world_point[0, 1]), float(world_point[0, 2])

            det["pos_3d"] = [wx, wy, wz]
            det["pos_2d"] = {"x": wx, "y": wz}
            det["depth_median"] = median_d

        return detections

    @staticmethod
    def build_text_prompt(labels: List[str]) -> str:
        return " . ".join(labels)
