from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def agentic_root() -> Path:
    return Path(__file__).resolve().parents[1]


def as_path(path: Union[str, Path], base: Optional[Path] = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    primary = (base or repo_root()) / p
    if primary.exists():
        return primary
    if base is None:
        secondary = agentic_root() / p
        if secondary.exists():
            return secondary
    return primary


def read_json(path: Union[str, Path]) -> Dict[str, Any]:
    import json

    p = as_path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {p}")
    return data


def write_json(path: Union[str, Path], data: Any) -> None:
    import json

    p = as_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _collapse_repeated_tokens(tokens: List[str]) -> List[str]:
    collapsed: List[str] = []
    for tok in tokens:
        if not collapsed or collapsed[-1] != tok:
            collapsed.append(tok)
    return collapsed


def _collapse_repeated_phrase_blocks(tokens: List[str]) -> List[str]:
    if len(tokens) < 2:
        return tokens
    out: List[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        collapsed = False
        max_block = (n - i) // 2
        for block_len in range(max_block, 0, -1):
            block = tokens[i : i + block_len]
            if block and tokens[i + block_len : i + 2 * block_len] == block:
                out.extend(block)
                i += 2 * block_len
                collapsed = True
                while i + block_len <= n and tokens[i : i + block_len] == block:
                    i += block_len
                break
        if not collapsed:
            out.append(tokens[i])
            i += 1
    return out


def normalize_label(value: Any) -> str:
    text = str(value or "").strip()
    text = text.replace("\\", "/").split("/")[-1]
    text = text.lower().replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    text = re.sub(r"_?4k$", "", text)
    text = re.sub(r"_?object_config$", "", text)
    tokens = [tok for tok in text.split("_") if tok]
    collapsed = _collapse_repeated_phrase_blocks(_collapse_repeated_tokens(tokens))
    text = "_".join(collapsed)
    return text


MATERIAL_ONLY_LABELS = {
    "acrylic",
    "aluminum",
    "brass",
    "bronze",
    "cardboard",
    "ceramic",
    "chrome",
    "concrete",
    "cotton",
    "fabric",
    "glass",
    "iron",
    "leather",
    "metal",
    "paper",
    "plastic",
    "polyester",
    "rubber",
    "silver",
    "steel",
    "stone",
    "velvet",
    "wood",
    "wooden",
}

DESCRIPTIVE_LABEL_TOKENS = {
    "ancient",
    "antique",
    "big",
    "classic",
    "decorative",
    "large",
    "little",
    "modern",
    "new",
    "old",
    "ornamental",
    "round",
    "small",
    "square",
    "tall",
    "vintage",
}

CANONICAL_PHRASE_LABELS = {
    "alarm_clock": "clock",
    "antique_ceramic_vase": "vase",
    "bottles_wine_bottles": "wine_bottles",
    "brass_pot": "pot",
    "brass_vase": "vase",
    "ceramic_vase": "vase",
    "chess_set": "chess",
    "coffee_table": "table",
    "desk_table": "table",
    "dining_table": "table",
    "floor_lamp": "lamp",
    "horse_statue": "horse_statue",
    "marble_bust": "marble_bust",
    "metal_stool": "metal_stool",
    "metal_stool_metal": "metal_stool",
    "shelf_cabinet": "cabinet",
    "statue_horse": "horse_statue",
    "statue_horse_statue": "horse_statue",
    "stool_metal": "metal_stool",
    "stool_metal_stool": "metal_stool",
    "table_lamp": "lamp",
    "tea_set": "tea_set",
    "wine_bottles": "wine_bottles",
}

CANONICAL_OBJECT_TOKENS = {
    "apple",
    "alarm",
    "bed",
    "book",
    "bottle",
    "bottles",
    "bowl",
    "bust",
    "cabinet",
    "cake",
    "camera",
    "chair",
    "chess",
    "clock",
    "console",
    "cup",
    "desk",
    "dessert",
    "fruit",
    "horse",
    "horse_statue",
    "lamp",
    "marble_bust",
    "megaphone",
    "metal_stool",
    "mirror",
    "pear",
    "pears",
    "picture",
    "pillow",
    "pillows",
    "plant",
    "pot",
    "shelf",
    "sofa",
    "statue",
    "stool",
    "table",
    "tea",
    "tea_set",
    "teapot",
    "vase",
    "wine_bottles",
}

GENERIC_NON_OBJECT_LABELS = {
    "area",
    "background",
    "image",
    "item",
    "object",
    "objects",
    "part",
    "parts",
    "scene",
    "stuff",
    "surface",
    "thing",
    "things",
}


def sanitize_detection_label(value: Any) -> str:
    """Normalize open-vocabulary detector phrases into stable object labels."""

    label = normalize_label(value)
    if not label:
        return ""
    tokens = [tok for tok in label.split("_") if tok]
    if len(tokens) >= 2:
        whole = "_".join(tokens)
        if whole in CANONICAL_PHRASE_LABELS:
            return CANONICAL_PHRASE_LABELS[whole]
    while tokens and tokens[-1] in DESCRIPTIVE_LABEL_TOKENS:
        tokens.pop()
    tokens = _collapse_repeated_phrase_blocks(_collapse_repeated_tokens(tokens))
    label = "_".join(tokens)
    if label in CANONICAL_PHRASE_LABELS:
        return CANONICAL_PHRASE_LABELS[label]
    if "tea" in tokens and any(tok in {"pear", "pears", "apple", "cake"} for tok in tokens):
        return ""
    for phrase in sorted(CANONICAL_PHRASE_LABELS, key=lambda item: -len(item.split("_"))):
        phrase_tokens = phrase.split("_")
        if len(phrase_tokens) <= 1 or len(phrase_tokens) > len(tokens):
            continue
        for i in range(0, len(tokens) - len(phrase_tokens) + 1):
            if tokens[i : i + len(phrase_tokens)] == phrase_tokens:
                return CANONICAL_PHRASE_LABELS[phrase]
    object_tokens = [tok for tok in tokens if tok in CANONICAL_OBJECT_TOKENS]
    if len(tokens) > 1 and object_tokens:
        return object_tokens[-1]
    return label


def is_noise_detection_label(value: Any) -> bool:
    label = sanitize_detection_label(value)
    if not label:
        return True
    tokens = [tok for tok in label.split("_") if tok]
    if tokens and all(tok in MATERIAL_ONLY_LABELS or tok in DESCRIPTIVE_LABEL_TOKENS for tok in tokens):
        return True
    if label in GENERIC_NON_OBJECT_LABELS:
        return True
    return False


def object_stem(obj: Dict[str, Any]) -> str:
    model_id = obj.get("model_id")
    if model_id:
        return normalize_label(model_id)
    return normalize_label(obj.get("name", obj.get("id", "object")))


def position_2d(obj: Dict[str, Any]) -> Optional[List[float]]:
    pos = obj.get("position")
    if not isinstance(pos, list) or len(pos) < 3:
        return None
    try:
        return [float(pos[0]), float(pos[2])]
    except Exception:
        return None


def euclidean_2d(a: Iterable[float], b: Iterable[float]) -> float:
    av = list(a)
    bv = list(b)
    return math.sqrt((float(av[0]) - float(bv[0])) ** 2 + (float(av[1]) - float(bv[1])) ** 2)


def euclidean_pose(a: Any, b: Any) -> float:
    return math.sqrt((float(a.x) - float(b.x)) ** 2 + (float(a.y) - float(b.y)) ** 2 + (float(a.z) - float(b.z)) ** 2)


def token_overlap(a: str, b: str) -> float:
    at = {x for x in normalize_label(a).split("_") if x and x not in {"01", "02", "03", "4k"}}
    bt = {x for x in normalize_label(b).split("_") if x and x not in {"01", "02", "03", "4k"}}
    if not at or not bt:
        return 0.0
    return len(at & bt) / max(1, len(at | bt))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def bbox_from_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    pos = obj.get("position") if isinstance(obj.get("position"), list) else [0.0, 0.0, 0.0]
    x, y, z = (safe_float(pos[0]), safe_float(pos[1]), safe_float(pos[2]))
    profile = obj.get("object_profile") or {}
    fx = safe_float(profile.get("footprint_x"), 0.0)
    fz = safe_float(profile.get("footprint_z"), 0.0)
    radius = safe_float(profile.get("radius"), safe_float(obj.get("placement_radius"), 0.25))
    if fx <= 0:
        fx = max(0.05, 2.0 * radius)
    if fz <= 0:
        fz = max(0.05, 2.0 * radius)
    height = safe_float(profile.get("height"), 0.0)
    if height <= 0:
        height = max(0.05, safe_float(profile.get("y_offset"), safe_float(obj.get("placement_y_offset"), 0.1)) * 2.0)
    mn = [x - fx / 2.0, y, z - fz / 2.0]
    mx = [x + fx / 2.0, y + height, z + fz / 2.0]
    return {
        "min": [round(v, 4) for v in mn],
        "max": [round(v, 4) for v in mx],
        "center": [round(x, 4), round(y + height / 2.0, 4), round(z, 4)],
        "size": [round(fx, 4), round(height, 4), round(fz, 4)],
    }


def footprint_region(obj: Dict[str, Any]) -> List[Dict[str, float]]:
    bbox = bbox_from_object(obj)
    mn = bbox["min"]
    mx = bbox["max"]
    return [
        {"x": mn[0], "y": mn[2]},
        {"x": mx[0], "y": mn[2]},
        {"x": mx[0], "y": mx[2]},
        {"x": mn[0], "y": mx[2]},
    ]
