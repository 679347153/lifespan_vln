from __future__ import annotations

import base64
import io
import logging
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

logger = logging.getLogger("remote_vision_server")


def _repo_root() -> Path:
    return Path(os.environ.get("REMOTE_VISION_REPO_ROOT", Path.cwd())).resolve()


def _default_path(relative: str) -> str:
    return str((_repo_root() / relative).resolve())


def _looks_like_hf_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists() and (
        (path / "tokenizer_config.json").exists() or (path / "vocab.txt").exists()
    )


def _looks_like_clip_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists() and (
        (path / "preprocessor_config.json").exists()
        or (path / "tokenizer_config.json").exists()
        or (path / "vocab.json").exists()
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


def _resolve_clip_model_path(model_name: str = "clip-vit-large-patch14") -> Optional[str]:
    explicit = (
        os.environ.get("REMOTE_VISION_CLIP_MODEL_PATH")
        or os.environ.get("RAANAV_CLIP_MODEL_PATH")
    )
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))

    repo = _repo_root()
    names = [
        model_name,
        os.environ.get("REMOTE_VISION_CLIP_MODEL", ""),
        os.environ.get("RAANAV_CLIP_MODEL", ""),
        "clip-vit-large-patch14",
        "clip-vit-base-patch32",
    ]
    for name in names:
        if not name or "/" in name:
            continue
        candidates.extend([repo / name, repo.parent / name])

    for env_name in ["HF_HOME", "TRANSFORMERS_CACHE"]:
        base = os.environ.get(env_name)
        if base:
            for name in ["clip-vit-large-patch14", "clip-vit-base-patch32"]:
                candidates.append(Path(base) / name)

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if _looks_like_clip_model_dir(candidate):
            return str(candidate)
    return None


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


class DetectionRequest(BaseModel):
    image_base64: Optional[str] = Field(default=None, description="Base64 image or data URL.")
    image_path: Optional[str] = Field(default=None, description="Server-local image path, useful for server tests.")
    text_prompt: Optional[str] = Field(default=None, description="GroundingDINO prompt, for example: chair . table .")
    labels: Optional[List[str]] = Field(default=None, description="Labels converted to a GroundingDINO prompt.")
    box_threshold: float = 0.35
    text_threshold: float = 0.35
    return_mask_png: bool = False
    return_clip_embedding: bool = False
    clip_mode: str = "mask_only"
    clip_masked_weight: float = 0.75
    clip_bbox_padding: int = 20
    clip_model: Optional[str] = None
    clip_local_files_only: Optional[bool] = None


class DetectionResponse(BaseModel):
    model: str
    device: str
    elapsed_seconds: float
    width: int
    height: int
    detections: List[Dict[str, Any]]


class ClipTextImageSimilarityRequest(BaseModel):
    text: str
    image_embeddings: List[List[float]]
    ids: Optional[List[str]] = None
    labels: Optional[List[str]] = None
    top_k: int = 20
    min_score: float = 0.0
    clip_model: Optional[str] = None
    clip_local_files_only: Optional[bool] = None


class ClipTextImageSimilarityResponse(BaseModel):
    model: str
    device: str
    elapsed_seconds: float
    query: str
    embedding_dim: int
    results: List[Dict[str, Any]]


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
        self._clip_model: Any = None
        self._clip_processor: Any = None
        self._clip_model_name: Optional[str] = None

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

    def _default_clip_model_name(self) -> str:
        return (
            os.environ.get("REMOTE_VISION_CLIP_MODEL_PATH")
            or os.environ.get("RAANAV_CLIP_MODEL_PATH")
            or _resolve_clip_model_path()
            or os.environ.get("REMOTE_VISION_CLIP_MODEL")
            or os.environ.get("RAANAV_CLIP_MODEL")
            or "openai/clip-vit-base-patch32"
        )

    def _get_clip_model(
        self,
        model_name_or_path: Optional[str] = None,
        local_files_only: Optional[bool] = None,
    ) -> Any:
        model_name = model_name_or_path or self._default_clip_model_name()
        if local_files_only is None:
            local_files_only = _env_bool(
                "REMOTE_VISION_CLIP_LOCAL_FILES_ONLY",
                default=os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            )
        if self._clip_model is not None and self._clip_model_name == model_name:
            return self._clip_model, self._clip_processor

        from transformers import CLIPModel, CLIPProcessor

        logger.info("Loading CLIP model: %s -> %s", model_name, self.device)
        model = CLIPModel.from_pretrained(model_name, local_files_only=bool(local_files_only))
        processor = CLIPProcessor.from_pretrained(model_name, local_files_only=bool(local_files_only))
        model.to(self.device)
        model.eval()
        self._clip_model = model
        self._clip_processor = processor
        self._clip_model_name = model_name
        return model, processor

    def encode_clip_text(
        self,
        text: str,
        model_name_or_path: Optional[str] = None,
        local_files_only: Optional[bool] = None,
    ) -> torch.Tensor:
        model, processor = self._get_clip_model(model_name_or_path, local_files_only)
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            feature = model.get_text_features(**inputs)
            feature = feature / feature.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return feature[0].detach().cpu()

    def text_image_similarity(
        self,
        text: str,
        image_embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        top_k: int = 20,
        min_score: float = 0.0,
        model_name_or_path: Optional[str] = None,
        local_files_only: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if not image_embeddings:
            return {"embedding_dim": 0, "results": []}
        matrix = torch.tensor(image_embeddings, dtype=torch.float32)
        if matrix.ndim != 2:
            raise ValueError("image_embeddings must be a 2D list.")
        query = self.encode_clip_text(text, model_name_or_path, local_files_only)
        if int(matrix.shape[1]) != int(query.shape[0]):
            raise ValueError(
                f"CLIP dimension mismatch: image dim={int(matrix.shape[1])}, text dim={int(query.shape[0])}. "
                "Use the same CLIP model for detection embeddings and text search."
            )
        matrix = matrix / matrix.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scores = torch.mv(matrix, query).numpy().tolist()
        rows: List[Dict[str, Any]] = []
        for idx, score in enumerate(scores):
            if float(score) < min_score:
                continue
            rows.append(
                {
                    "index": idx,
                    "id": ids[idx] if ids and idx < len(ids) else str(idx),
                    "label": labels[idx] if labels and idx < len(labels) else "",
                    "score": float(score),
                }
            )
        rows.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
        if top_k > 0:
            rows = rows[:top_k]
        return {"embedding_dim": int(matrix.shape[1]), "results": rows}

    def detect_segment(
        self,
        image_rgb: np.ndarray,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
        return_mask_png: bool,
        return_clip_embedding: bool = False,
        clip_mode: str = "mask_only",
        clip_masked_weight: float = 0.75,
        clip_bbox_padding: int = 20,
        clip_model: Optional[str] = None,
        clip_local_files_only: Optional[bool] = None,
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

        if return_clip_embedding:
            self._add_clip_embeddings(
                image_rgb=image_rgb,
                detections=detections,
                masks_np=masks_np,
                mode=clip_mode,
                masked_weight=clip_masked_weight,
                bbox_padding=clip_bbox_padding,
                model_name_or_path=clip_model,
                local_files_only=clip_local_files_only,
            )
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

    def _add_clip_embeddings(
        self,
        image_rgb: np.ndarray,
        detections: List[Dict[str, Any]],
        masks_np: np.ndarray,
        mode: str,
        masked_weight: float,
        bbox_padding: int,
        model_name_or_path: Optional[str],
        local_files_only: Optional[bool],
    ) -> None:
        if not detections:
            return
        from PIL import Image as PILImage

        model, processor = self._get_clip_model(model_name_or_path, local_files_only)
        encode_mode = "mask_only" if mode not in {"mask_only", "full"} else mode
        images_to_encode = []
        det_indices: List[int] = []
        image_roles: List[str] = []

        for idx, det in enumerate(detections):
            bbox = det.get("bbox_xyxy")
            if bbox is None:
                det["clip_embedding"] = []
                continue
            local_crop, masked_crop = _crop_and_mask_for_clip(
                image_rgb=image_rgb,
                bbox_xyxy=bbox,
                mask=masks_np[idx] if idx < len(masks_np) else None,
                padding=bbox_padding,
            )
            if local_crop.size == 0 or masked_crop.size == 0:
                det["clip_embedding"] = []
                continue
            images_to_encode.append(PILImage.fromarray(masked_crop))
            det_indices.append(idx)
            image_roles.append("masked")
            if encode_mode == "full":
                images_to_encode.append(PILImage.fromarray(local_crop))
                det_indices.append(idx)
                image_roles.append("local")

        if not images_to_encode:
            for det in detections:
                det.setdefault("clip_embedding", [])
            return

        all_feats = []
        batch_size = int(os.environ.get("REMOTE_VISION_CLIP_BATCH_SIZE", "32"))
        with torch.no_grad():
            for start in range(0, len(images_to_encode), batch_size):
                batch_imgs = images_to_encode[start : start + batch_size]
                inputs = processor(images=batch_imgs, return_tensors="pt", padding=True)
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                feats = model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                all_feats.append(feats.cpu())
        features = torch.cat(all_feats, dim=0)

        if encode_mode == "mask_only":
            for feature_idx, det_idx in enumerate(det_indices):
                detections[det_idx]["clip_embedding"] = features[feature_idx].numpy().tolist()
        else:
            accum: Dict[int, Dict[str, torch.Tensor]] = {}
            for feature_idx, (det_idx, role) in enumerate(zip(det_indices, image_roles)):
                accum.setdefault(det_idx, {})[role] = features[feature_idx]
            weight = float(np.clip(masked_weight, 0.0, 1.0))
            for det_idx, parts in accum.items():
                masked = parts.get("masked")
                local = parts.get("local")
                if masked is not None and local is not None:
                    fused = weight * masked + (1.0 - weight) * local
                    fused = fused / fused.norm().clamp(min=1e-8)
                    detections[det_idx]["clip_embedding"] = fused.numpy().tolist()
                elif masked is not None:
                    detections[det_idx]["clip_embedding"] = masked.numpy().tolist()
                else:
                    detections[det_idx]["clip_embedding"] = []

        for det in detections:
            det.setdefault("clip_embedding", [])


def _crop_and_mask_for_clip(
    image_rgb: np.ndarray,
    bbox_xyxy: List[float],
    mask: Optional[np.ndarray],
    padding: int = 20,
) -> Any:
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    x1p = max(0, x1 - padding)
    y1p = max(0, y1 - padding)
    x2p = min(width, x2 + padding)
    y2p = min(height, y2 + padding)
    local_crop = image_rgb[y1p:y2p, x1p:x2p].copy()

    if mask is not None:
        masked_rgb = image_rgb.copy()
        masked_rgb[~mask] = 0
        masked_crop = masked_rgb[y1p:y2p, x1p:x2p].copy()
    else:
        masked_crop = local_crop.copy()

    return local_crop, masked_crop


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
        "clip_model_path": _resolve_clip_model_path(),
        "clip_loaded": bool(loaded and getattr(get_detector(), "_clip_model", None) is not None) if loaded else False,
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
            },
            {
                "id": "CLIP-image-embedding",
                "object": "model",
                "owned_by": "remote_vision_server",
            },
            {
                "id": "CLIP-text-image-similarity",
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
            return_clip_embedding=request.return_clip_embedding,
            clip_mode=request.clip_mode,
            clip_masked_weight=request.clip_masked_weight,
            clip_bbox_padding=request.clip_bbox_padding,
            clip_model=request.clip_model,
            clip_local_files_only=request.clip_local_files_only,
        )
    except Exception as exc:
        logger.exception("Remote vision detect_segment failed")
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


@app.post("/v1/clip/text_image_similarity", response_model=ClipTextImageSimilarityResponse)
def clip_text_image_similarity(request: ClipTextImageSimilarityRequest) -> ClipTextImageSimilarityResponse:
    start = time.time()
    try:
        detector = get_detector()
        result = detector.text_image_similarity(
            text=request.text,
            image_embeddings=request.image_embeddings,
            ids=request.ids,
            labels=request.labels,
            top_k=request.top_k,
            min_score=request.min_score,
            model_name_or_path=request.clip_model,
            local_files_only=request.clip_local_files_only,
        )
    except Exception as exc:
        logger.exception("Remote CLIP text/image similarity failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ClipTextImageSimilarityResponse(
        model=detector._clip_model_name or detector._default_clip_model_name(),
        device=detector.device,
        elapsed_seconds=round(time.time() - start, 6),
        query=request.text,
        embedding_dim=int(result["embedding_dim"]),
        results=result["results"],
    )


if os.environ.get("REMOTE_VISION_EAGER_LOAD", "0") == "1":
    get_detector()
