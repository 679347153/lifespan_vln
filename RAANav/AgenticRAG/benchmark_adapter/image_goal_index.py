from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from RAANav.AgenticRAG.semantic_map import Object

from .common import as_path, normalize_label, token_overlap
from .episode_to_queries import QuerySpec


@dataclass
class ImageMatch:
    score: float
    backend: str
    reason: str


class ImageGoalIndex:
    """Image-goal similarity provider.

    The class is intentionally dependency-light. It first uses deterministic
    object image stems so the benchmark can run everywhere. A future CLIP
    backend can be slotted in behind the same interface without changing the
    closed-loop runner.
    """

    def __init__(self, images_dir: str = "objects_images", backend: str = "auto") -> None:
        self.images_dir = as_path(images_dir)
        self.backend = backend
        self.available_images: Dict[str, Path] = {}
        if self.images_dir.is_dir():
            for path in self.images_dir.glob("*"):
                if path.suffix.lower() in {".webp", ".png", ".jpg", ".jpeg"}:
                    self.available_images[normalize_label(path.stem)] = path

    def query_modality(self, query: QuerySpec) -> str:
        return "image" if query.task_type == "image_goal" and query.image_path else "text"

    def reference_stem(self, query: QuerySpec) -> Optional[str]:
        if not query.image_path:
            return None
        return normalize_label(Path(query.image_path).stem)

    def object_image_stem(self, obj: Object) -> Optional[str]:
        imgs = obj.imgs or {}
        for values in imgs.values():
            if isinstance(values, list) and values:
                return normalize_label(Path(str(values[0])).stem)
            if isinstance(values, str):
                return normalize_label(Path(values).stem)
        label = normalize_label(obj.label)
        if label in self.available_images:
            return label
        return None

    def similarity(self, query: QuerySpec, obj: Object) -> ImageMatch:
        if self.query_modality(query) != "image":
            score = 1.0 if normalize_label(obj.label) == query.query_label else token_overlap(obj.label, query.query_label)
            return ImageMatch(score=max(0.0, min(1.0, score)), backend="text_fallback", reason="non_image_query")

        ref = self.reference_stem(query)
        obj_stem = self.object_image_stem(obj)
        if ref and obj_stem:
            if ref == obj_stem:
                return ImageMatch(score=1.0, backend="image_stem", reason="exact_image_stem")
            overlap = token_overlap(ref, obj_stem)
            if overlap > 0:
                return ImageMatch(score=overlap, backend="image_stem", reason=f"image_stem_overlap={overlap:.2f}")

        # Last-resort fallback keeps image_goal executable when the reference
        # image or object image is absent.
        fallback = token_overlap(query.query_label, obj.label)
        return ImageMatch(score=max(0.0, min(1.0, fallback)), backend="label_fallback", reason="missing_image_or_no_match")

