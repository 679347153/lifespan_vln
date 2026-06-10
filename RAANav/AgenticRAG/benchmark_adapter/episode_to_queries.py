from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from benchmark.schemas import Episode, Subtask

from .common import normalize_label


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
    sampled_region_id: Any = None
    target_instance_id: Any = None


def query_from_subtask(episode: Episode, subtask: Subtask) -> QuerySpec:
    metadata = subtask.metadata or {}
    model_id = metadata.get("model_id")
    query_label = normalize_label(model_id or subtask.target_object)
    return QuerySpec(
        episode_id=episode.episode_id,
        subtask_id=subtask.subtask_id,
        task_type=subtask.task_type,
        target_object=subtask.target_object,
        target_object_id=str(subtask.target_object_id),
        query_label=query_label,
        image_path=subtask.image_path,
        language_prompt=subtask.language_prompt,
        sampled_region_id=metadata.get("sampled_region_id"),
        target_instance_id=metadata.get("target_instance_id"),
    )


def episode_to_queries(episode: Episode) -> List[QuerySpec]:
    return [query_from_subtask(episode, subtask) for subtask in episode.subtasks]

