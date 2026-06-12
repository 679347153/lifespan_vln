from __future__ import annotations

import base64
import io
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _repo_root() -> Path:
    return Path(os.environ.get("REMOTE_VISION_REPO_ROOT", Path.cwd())).resolve()


def _default_path(relative: str) -> str:
    return str((_repo_root() / relative).resolve())


def _looks_like_hf_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists() and (
        (path / "tokenizer_config.json").exists() or (path / "vocab.txt").exists()
    )


def _resolve_text_encoder_path(model_name: str = "bert-base-uncased") -> Optional[str]:
    explicit = (
        os.environ.get("BERT_BASE_UNCASED_PATH")
        or os.environ.get("REMOTE_VISION_BERT_PATH")
        or os.environ.get("GROUNDINGDINO_TEXT_ENCODER_PATH")
    )
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))

    repo = _repo_root()
    candidates.extend(
        [
            repo / model_name,
            repo.parent / model_name,
        ]
    )

    for env_name in ["HF_HOME", "TRANSFORMERS_CACHE"]:
        base = os.environ.get(env_name)
        if base:
            candidates.append(Path(base) / model_name)

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if _looks_like_hf_model_dir(candidate):
            return str(candidate)
    return None


class DetectionRequest(BaseModel):
    image_base64: Optional[str] = Field(default=None, description="Base64 image or data URL.")
    image_path: Optional[str] = Field(default=None, description="Server-local image path, useful for server tests.")
    text_prompt: Optional[str] = Field(default=None, description="GroundingDINO prompt, for example: chair . table .")
    labels: Optional[List[str]] = Field(default=None, description="Labels converted to a GroundingDINO prompt.")
    box_threshold: float = 0.35
    text_threshold: float = 0.35
    return_mask_png: bool = False


class DetectionResponse(BaseModel):
    model: str
    device: str
    elapsed_seconds: float
    width: int
    height: int
    detections: List[Dict[str, Any]]


class VisionDetector:
    def __init__(
        self,
        device: str,
        gdino_config: str,
        gdino_checkpoint: str,
        mobilesam_checkpoint: str,
    ) -> None:
        self.device = device
        self.gdino_config = gdino_config
        self.gdino_checkpoint = gdino_checkpoint
        self.mobilesam_checkpoint = mobilesam_checkpoint

        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict

        args = SLConfig.fromfile(gdino_config)
        args.device = device
        if getattr(args, "text_encoder_type", None) == "bert-base-uncased":
            local_text_encoder = _resolve_text_encoder_path("bert-base-uncased")
            if local_text_encoder:
                args.text_encoder_type = local_text_encoder
                os.environ["BERT_BASE_UNCASED_PATH"] = local_text_encoder
        self.gdino_model = build_model(args)
        checkpoint = self._torch_load(gdino_checkpoint)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        self.gdino_model.load_state_dict(clean_state_dict(state), strict=False)
        self.gdino_model.eval()
        self.gdino_model.to(device)

        from mobile_sam import SamPredictor, sam_model_registry

        sam = sam_model_registry["vit_t"](checkpoint=mobilesam_checkpoint)
        sam.to(device)
        sam.eval()
        self.sam_predictor = SamPredictor(sam)

    @staticmethod
    def _torch_load(path: str) -> Any:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def detect_segment(
        self,
        image_rgb: np.ndarray,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
        return_mask_png: bool,
    ) -> List[Dict[str, Any]]:
        from groundingdino.util.inference import predict
        import groundingdino.datasets.transforms as T

        pil_image = Image.fromarray(image_rgb)
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_tensor, _ = transform(pil_image, None)

        with torch.no_grad():
            boxes, logits, phrases = predict(
                self.gdino_model,
                image_tensor,
                prompt,
                box_threshold,
                text_threshold,
                device=self.device,
            )

        if len(boxes) == 0:
            return []

        height, width = image_rgb.shape[:2]
        boxes_xyxy = self._cxcywh_to_xyxy(boxes, width, height)
        self.sam_predictor.set_image(image_rgb)

        boxes_torch = torch.tensor(boxes_xyxy, dtype=torch.float32, device=self.device)
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_torch, image_rgb.shape[:2])

        with torch.no_grad():
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

        masks_np = masks.detach().cpu().numpy()[:, 0, :, :].astype(bool)
        detections: List[Dict[str, Any]] = []
        for idx in range(len(boxes_xyxy)):
            mask = masks_np[idx]
            record: Dict[str, Any] = {
                "label": str(phrases[idx]).strip().lower(),
                "bbox_xyxy": [float(v) for v in boxes_xyxy[idx].tolist()],
                "confidence": float(logits[idx]),
                "mask_area": int(mask.sum()),
            }
            if return_mask_png:
                record["mask_png_base64"] = _mask_to_png_base64(mask)
            detections.append(record)
        return detections

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor, width: int, height: int) -> np.ndarray:
        boxes_np = boxes.detach().cpu().numpy()
        cx, cy, w, h = boxes_np[:, 0], boxes_np[:, 1], boxes_np[:, 2], boxes_np[:, 3]
        x1 = (cx - w / 2) * width
        y1 = (cy - h / 2) * height
        x2 = (cx + w / 2) * width
        y2 = (cy + h / 2) * height
        x1 = np.clip(x1, 0, width)
        y1 = np.clip(y1, 0, height)
        x2 = np.clip(x2, 0, width)
        y2 = np.clip(y2, 0, height)
        return np.stack([x1, y1, x2, y2], axis=-1)


def _mask_to_png_base64(mask: np.ndarray) -> str:
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _load_image(request: DetectionRequest) -> np.ndarray:
    if request.image_base64:
        raw = request.image_base64
        if "," in raw and raw.strip().lower().startswith("data:"):
            raw = raw.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(raw)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}") from exc
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.asarray(image)

    if request.image_path:
        path = Path(request.image_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"image_path not found: {path}")
        return np.asarray(Image.open(path).convert("RGB"))

    raise HTTPException(status_code=400, detail="Either image_base64 or image_path is required.")


def _build_prompt(request: DetectionRequest) -> str:
    if request.text_prompt:
        prompt = request.text_prompt.strip()
    elif request.labels:
        prompt = " . ".join(label.strip().lower() for label in request.labels if label.strip())
    else:
        raise HTTPException(status_code=400, detail="Either text_prompt or labels is required.")
    if not prompt.endswith("."):
        prompt = f"{prompt} ."
    return prompt


@lru_cache(maxsize=1)
def get_detector() -> VisionDetector:
    device = os.environ.get("REMOTE_VISION_DEVICE", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("REMOTE_VISION_DEVICE=cuda but torch.cuda.is_available() is false.")

    gdino_config = os.environ.get(
        "GDINO_CONFIG",
        _default_path("third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
    )
    gdino_checkpoint = os.environ.get(
        "GDINO_CHECKPOINT",
        _default_path("RAANav/AgenticRAG/checkpoints/groundingdino_swint_ogc.pth"),
    )
    mobilesam_checkpoint = os.environ.get(
        "MOBILESAM_CHECKPOINT",
        _default_path("RAANav/AgenticRAG/checkpoints/mobile_sam.pt"),
    )
    for path in [gdino_config, gdino_checkpoint, mobilesam_checkpoint]:
        if not Path(path).exists():
            raise RuntimeError(f"Required model file not found: {path}")

    return VisionDetector(
        device=device,
        gdino_config=gdino_config,
        gdino_checkpoint=gdino_checkpoint,
        mobilesam_checkpoint=mobilesam_checkpoint,
    )


app = FastAPI(title="RAANav Remote Vision Server", version="1.0")


@app.get("/health")
def health() -> Dict[str, Any]:
    loaded = get_detector.cache_info().currsize > 0
    return {
        "ok": True,
        "loaded": loaded,
        "device": os.environ.get("REMOTE_VISION_DEVICE", "cuda"),
        "repo_root": str(_repo_root()),
        "bert_base_uncased_path": _resolve_text_encoder_path("bert-base-uncased"),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
    }


@app.get("/v1/models")
def models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": "GroundingDINO-SwinT-OGC+MobileSAM-vit_t",
                "object": "model",
                "owned_by": "remote_vision_server",
            }
        ],
    }


@app.post("/v1/detect_segment", response_model=DetectionResponse)
def detect_segment(request: DetectionRequest) -> DetectionResponse:
    start = time.time()
    image_rgb = _load_image(request)
    prompt = _build_prompt(request)
    try:
        detector = get_detector()
        detections = detector.detect_segment(
            image_rgb=image_rgb,
            prompt=prompt,
            box_threshold=request.box_threshold,
            text_threshold=request.text_threshold,
            return_mask_png=request.return_mask_png,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    height, width = image_rgb.shape[:2]
    return DetectionResponse(
        model="GroundingDINO-SwinT-OGC+MobileSAM-vit_t",
        device=detector.device,
        elapsed_seconds=round(time.time() - start, 6),
        width=width,
        height=height,
        detections=detections,
    )


if os.environ.get("REMOTE_VISION_EAGER_LOAD", "0") == "1":
    get_detector()
