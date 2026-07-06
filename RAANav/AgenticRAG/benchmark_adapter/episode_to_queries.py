from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.schemas import Episode, Subtask

from .common import as_path, normalize_label, sanitize_detection_label, token_overlap


AliasEntry = Dict[str, List[str]]
TargetNameMap = Dict[str, AliasEntry]
WEAK_ALIAS_SIMILARITY_CAP = 0.35


DEFAULT_TARGET_NAME_MAP: TargetNameMap = {
    "alarm_clock_01": {"strong": ["alarm_clock_01", "alarm_clock", "clock"], "weak": []},
    "antique_ceramic_vase_01": {
        "strong": ["antique_ceramic_vase_01", "antique_ceramic_vase", "ceramic_vase", "vase"],
        "weak": [],
    },
    "brass_pot_01": {"strong": ["brass_pot_01", "brass_pot", "pot"], "weak": ["vessel"]},
    "brass_vase_03": {"strong": ["brass_vase_03", "brass_vase", "vase"], "weak": ["decorative_vase"]},
    "camera_01": {"strong": ["camera_01", "camera"], "weak": []},
    "carrot_cake": {"strong": ["carrot_cake", "cake"], "weak": ["dessert"]},
    "chess_set": {"strong": ["chess_set", "chess"], "weak": ["board_game"]},
    "classicconsole_01": {"strong": ["classicconsole_01", "classic_console", "console"], "weak": []},
    "food_apple_01": {"strong": ["food_apple_01", "food_apple", "apple"], "weak": ["fruit"]},
    "food_pears_asian_01": {
        "strong": ["food_pears_asian_01", "food_pears_asian", "asian_pears", "pears", "pear"],
        "weak": ["fruit"],
    },
    "horse_statue_01": {"strong": ["horse_statue_01", "horse_statue"], "weak": ["statue"]},
    "marble_bust_01": {"strong": ["marble_bust_01", "marble_bust"], "weak": ["bust"]},
    "megaphone_01": {"strong": ["megaphone_01", "megaphone"], "weak": []},
    "metal_stool_02": {"strong": ["metal_stool_02", "metal_stool"], "weak": ["stool"]},
    "potted_plant_01": {"strong": ["potted_plant_01", "potted_plant", "plant"], "weak": []},
    "potted_plant_02": {"strong": ["potted_plant_02", "potted_plant", "plant"], "weak": []},
    "round_wooden_table_01": {
        "strong": ["round_wooden_table_01", "round_wooden_table", "wooden_table", "table"],
        "weak": [],
    },
    "side_table_tall_01": {"strong": ["side_table_tall_01", "side_table_tall", "side_table", "table"], "weak": []},
    "tea_set_01": {"strong": ["tea_set_01", "tea_set", "teapot"], "weak": ["cup", "tea"]},
    "throw_pillows_01": {"strong": ["throw_pillows_01", "throw_pillows", "pillows"], "weak": ["pillow"]},
    "wine_bottles_01": {"strong": ["wine_bottles_01", "wine_bottles"], "weak": ["bottles", "bottle"]},
    "wooden_table_02": {"strong": ["wooden_table_02", "wooden_table", "table"], "weak": []},
}


@dataclass
class QuerySpec:
    episode_id: str
    subtask_id: str
    task_type: str
    target_object: str
    target_object_id: str
    query_label: str
    image_path: Optional[str]
    language_prompt: Optional[str]
    alias_labels: List[str] = None
    strong_alias_labels: List[str] = None
    weak_alias_labels: List[str] = None
    sampled_region_id: Any = None
    target_instance_id: Any = None


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        norm = normalize_label(value)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _collapse_repeated_tokens(label: str) -> str:
    tokens = [tok for tok in normalize_label(label).split("_") if tok]
    if not tokens:
        return ""
    out: List[str] = []
    for tok in tokens:
        if not out or out[-1] != tok:
            out.append(tok)
    return "_".join(out)


def _strip_instance_suffix(label: str) -> str:
    tokens = [tok for tok in normalize_label(label).split("_") if tok]
    while tokens and tokens[-1].isdigit():
        tokens.pop()
    return "_".join(tokens)


def _fallback_aliases(label: str) -> List[str]:
    norm = normalize_label(label)
    aliases = [norm]
    collapsed = _collapse_repeated_tokens(norm)
    stripped = _strip_instance_suffix(collapsed)
    aliases.extend([collapsed, stripped])
    if stripped.endswith("_set"):
        aliases.append(stripped[: -len("_set")])
    if stripped.endswith("_01") or stripped.endswith("_02") or stripped.endswith("_03"):
        aliases.append(_strip_instance_suffix(stripped))
    return _dedupe_preserve_order(aliases)


def _normalize_alias_entry(value: Any, norm_key: str) -> AliasEntry:
    if isinstance(value, str):
        return {
            "strong": _dedupe_preserve_order([value, norm_key]),
            "weak": [],
        }
    if isinstance(value, list):
        return {
            "strong": _dedupe_preserve_order([str(v) for v in value] + [norm_key]),
            "weak": [],
        }
    if isinstance(value, dict):
        strong_raw = value.get("strong", [])
        weak_raw = value.get("weak", [])
        if isinstance(strong_raw, str):
            strong_values = [strong_raw]
        elif isinstance(strong_raw, list):
            strong_values = [str(v) for v in strong_raw]
        else:
            raise ValueError(f"Target name map 'strong' must be string or list: key={norm_key!r}")
        if isinstance(weak_raw, str):
            weak_values = [weak_raw]
        elif isinstance(weak_raw, list):
            weak_values = [str(v) for v in weak_raw]
        else:
            raise ValueError(f"Target name map 'weak' must be string or list: key={norm_key!r}")
        strong = _dedupe_preserve_order(strong_values + [norm_key])
        weak = [alias for alias in _dedupe_preserve_order(weak_values) if alias not in set(strong)]
        return {"strong": strong, "weak": weak}
    raise ValueError(f"Target name map values must be string, list, or object: key={norm_key!r}")


def load_target_name_map(path: Optional[str]) -> TargetNameMap:
    if not path:
        return {}
    p = as_path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Target name map must be a JSON object: {p}")
    out: TargetNameMap = {}
    for key, value in data.items():
        norm_key = normalize_label(key)
        if not norm_key:
            continue
        out[norm_key] = _normalize_alias_entry(value, norm_key)
    return out


def alias_groups_for_label(
    label: Any,
    *,
    target_name_map: Optional[TargetNameMap] = None,
    normalize_target_names: bool = True,
) -> AliasEntry:
    norm = normalize_label(label)
    strong: List[str] = []
    weak: List[str] = []
    maps = [target_name_map or {}]
    if normalize_target_names:
        maps.append(DEFAULT_TARGET_NAME_MAP)
    for mapping in maps:
        entry = mapping.get(norm)
        if not entry:
            continue
        strong.extend(entry.get("strong", []))
        weak.extend(entry.get("weak", []))
    if normalize_target_names:
        strong.extend(_fallback_aliases(norm))
    else:
        strong.append(norm)
    weak = _dedupe_preserve_order(weak)
    weak_set = set(weak)
    strong = [alias for alias in _dedupe_preserve_order(strong) if alias not in weak_set]
    strong_set = set(strong)
    weak = [alias for alias in weak if alias not in strong_set]
    return {"strong": strong, "weak": weak}


def aliases_for_label(
    label: Any,
    *,
    target_name_map: Optional[TargetNameMap] = None,
    normalize_target_names: bool = True,
) -> List[str]:
    groups = alias_groups_for_label(
        label,
        target_name_map=target_name_map,
        normalize_target_names=normalize_target_names,
    )
    return _dedupe_preserve_order([*(groups.get("strong") or []), *(groups.get("weak") or [])])


def _alias_matches(norm: str, aliases: List[str], threshold: float) -> bool:
    for alias in aliases:
        if not alias:
            continue
        if norm == alias:
            return True
        if token_overlap(norm, alias) >= threshold:
            return True
    return False


def query_label_match_strength(label: Any, query: QuerySpec, *, threshold: float = 0.6) -> str:
    norm = _collapse_repeated_tokens(sanitize_detection_label(label) or normalize_label(label))
    strong = query.strong_alias_labels or [query.query_label]
    if _alias_matches(norm, strong, threshold):
        return "strong"
    weak = query.weak_alias_labels or []
    if _alias_matches(norm, weak, threshold):
        return "weak"
    return "none"


def query_label_matches(label: Any, query: QuerySpec, *, threshold: float = 0.6, include_weak: bool = False) -> bool:
    strength = query_label_match_strength(label, query, threshold=threshold)
    if strength == "strong":
        return True
    if include_weak and strength == "weak":
        return True
    return False


def query_label_similarity(label: Any, query: QuerySpec, *, include_weak: bool = True) -> float:
    norm = _collapse_repeated_tokens(sanitize_detection_label(label) or normalize_label(label))
    strong = query.strong_alias_labels or [query.query_label]
    if norm in strong:
        return 1.0
    strong_score = max((token_overlap(norm, alias) for alias in strong if alias), default=0.0)
    if not include_weak:
        return strong_score
    weak = query.weak_alias_labels or []
    weak_match = _alias_matches(norm, weak, 0.6)
    weak_exact = WEAK_ALIAS_SIMILARITY_CAP if norm in weak else 0.0
    weak_overlap = max((token_overlap(norm, alias) for alias in weak if alias), default=0.0) * WEAK_ALIAS_SIMILARITY_CAP
    if weak_match:
        return max(min(strong_score, WEAK_ALIAS_SIMILARITY_CAP), weak_exact, weak_overlap)
    return max(strong_score, weak_exact, weak_overlap)


def query_from_subtask(
    episode: Episode,
    subtask: Subtask,
    *,
    target_name_map: Optional[TargetNameMap] = None,
    normalize_target_names: bool = True,
) -> QuerySpec:
    metadata = subtask.metadata or {}
    model_id = metadata.get("model_id")
    raw_label = model_id or subtask.target_object
    groups = alias_groups_for_label(
        raw_label,
        target_name_map=target_name_map,
        normalize_target_names=normalize_target_names,
    )
    strong_aliases = groups.get("strong") or []
    weak_aliases = groups.get("weak") or []
    aliases = _dedupe_preserve_order([*strong_aliases, *weak_aliases])
    query_label = strong_aliases[0] if strong_aliases else aliases[0] if aliases else normalize_label(raw_label)
    return QuerySpec(
        episode_id=episode.episode_id,
        subtask_id=subtask.subtask_id,
        task_type=subtask.task_type,
        target_object=subtask.target_object,
        target_object_id=str(subtask.target_object_id),
        query_label=query_label,
        image_path=subtask.image_path,
        language_prompt=subtask.language_prompt,
        alias_labels=aliases,
        strong_alias_labels=strong_aliases,
        weak_alias_labels=weak_aliases,
        sampled_region_id=metadata.get("sampled_region_id"),
        target_instance_id=metadata.get("target_instance_id"),
    )


def episode_to_queries(episode: Episode) -> List[QuerySpec]:
    return [query_from_subtask(episode, subtask) for subtask in episode.subtasks]
