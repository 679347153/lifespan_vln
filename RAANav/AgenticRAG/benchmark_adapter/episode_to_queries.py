from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.schemas import Episode, Subtask

from .common import as_path, normalize_label, token_overlap


DEFAULT_TARGET_NAME_MAP: Dict[str, List[str]] = {
    "alarm_clock_01": ["alarm_clock_01", "alarm_clock", "clock"],
    "antique_ceramic_vase_01": ["antique_ceramic_vase_01", "antique_ceramic_vase", "ceramic_vase", "vase"],
    "brass_pot_01": ["brass_pot_01", "brass_pot", "pot"],
    "brass_vase_03": ["brass_vase_03", "brass_vase", "vase"],
    "camera_01": ["camera_01", "camera"],
    "carrot_cake": ["carrot_cake", "cake"],
    "chess_set": ["chess_set", "chess"],
    "classicconsole_01": ["classicconsole_01", "classic_console", "console"],
    "food_apple_01": ["food_apple_01", "food_apple", "apple"],
    "food_pears_asian_01": ["food_pears_asian_01", "food_pears_asian", "asian_pears", "pears", "pear"],
    "horse_statue_01": ["horse_statue_01", "horse_statue", "statue"],
    "marble_bust_01": ["marble_bust_01", "marble_bust", "bust"],
    "megaphone_01": ["megaphone_01", "megaphone"],
    "metal_stool_02": ["metal_stool_02", "metal_stool", "stool"],
    "potted_plant_01": ["potted_plant_01", "potted_plant", "plant"],
    "potted_plant_02": ["potted_plant_02", "potted_plant", "plant"],
    "round_wooden_table_01": ["round_wooden_table_01", "round_wooden_table", "wooden_table", "table"],
    "side_table_tall_01": ["side_table_tall_01", "side_table_tall", "side_table", "table"],
    "tea_set_01": ["tea_set_01", "tea_set", "tea"],
    "throw_pillows_01": ["throw_pillows_01", "throw_pillows", "pillows", "pillow"],
    "wine_bottles_01": ["wine_bottles_01", "wine_bottles", "bottle"],
    "wooden_table_02": ["wooden_table_02", "wooden_table", "table"],
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


def load_target_name_map(path: Optional[str]) -> Dict[str, List[str]]:
    if not path:
        return {}
    p = as_path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Target name map must be a JSON object: {p}")
    out: Dict[str, List[str]] = {}
    for key, value in data.items():
        norm_key = normalize_label(key)
        if not norm_key:
            continue
        if isinstance(value, str):
            out[norm_key] = _dedupe_preserve_order([value, norm_key])
        elif isinstance(value, list):
            out[norm_key] = _dedupe_preserve_order([str(v) for v in value] + [norm_key])
        else:
            raise ValueError(f"Target name map values must be string or list: key={key!r}")
    return out


def aliases_for_label(
    label: Any,
    *,
    target_name_map: Optional[Dict[str, List[str]]] = None,
    normalize_target_names: bool = True,
) -> List[str]:
    norm = normalize_label(label)
    aliases: List[str] = []
    maps = [target_name_map or {}]
    if normalize_target_names:
        maps.append(DEFAULT_TARGET_NAME_MAP)
    for mapping in maps:
        if norm in mapping:
            aliases.extend(mapping[norm])
    if normalize_target_names:
        aliases.extend(_fallback_aliases(norm))
    else:
        aliases.append(norm)
    return _dedupe_preserve_order(aliases)


def query_label_matches(label: Any, query: QuerySpec, *, threshold: float = 0.6) -> bool:
    norm = _collapse_repeated_tokens(normalize_label(label))
    aliases = query.alias_labels or [query.query_label]
    for alias in aliases:
        if not alias:
            continue
        if norm == alias:
            return True
        if token_overlap(norm, alias) >= threshold:
            return True
    return False


def query_label_similarity(label: Any, query: QuerySpec) -> float:
    norm = _collapse_repeated_tokens(normalize_label(label))
    aliases = query.alias_labels or [query.query_label]
    if norm in aliases:
        return 1.0
    return max((token_overlap(norm, alias) for alias in aliases if alias), default=0.0)


def query_from_subtask(
    episode: Episode,
    subtask: Subtask,
    *,
    target_name_map: Optional[Dict[str, List[str]]] = None,
    normalize_target_names: bool = True,
) -> QuerySpec:
    metadata = subtask.metadata or {}
    model_id = metadata.get("model_id")
    raw_label = model_id or subtask.target_object
    aliases = aliases_for_label(
        raw_label,
        target_name_map=target_name_map,
        normalize_target_names=normalize_target_names,
    )
    query_label = aliases[0] if aliases else normalize_label(raw_label)
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
        sampled_region_id=metadata.get("sampled_region_id"),
        target_instance_id=metadata.get("target_instance_id"),
    )


def episode_to_queries(episode: Episode) -> List[QuerySpec]:
    return [query_from_subtask(episode, subtask) for subtask in episode.subtasks]
