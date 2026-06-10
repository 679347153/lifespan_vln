from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from benchmark.schemas import SplitManifest, read_split_manifest

from .common import as_path, read_json


@dataclass
class LayoutEntry:
    scene_name: str
    layout_id: str
    layout_index: int
    layout_path: Path
    manifest_path: Path
    seed: int = 0
    placed_count: int = 0
    failed_count: int = 0


class DatasetIndex:
    def __init__(self, split_manifest_path: Union[str, Path]) -> None:
        self.split_manifest_path = as_path(split_manifest_path)
        self.split: SplitManifest = read_split_manifest(self.split_manifest_path)
        self.manifest_paths: List[Path] = [as_path(p) for p in self.split.metadata.get("layout_manifests", [])]
        if not self.manifest_paths:
            raise ValueError(f"Split manifest has no metadata.layout_manifests: {self.split_manifest_path}")
        self.entries_by_scene: Dict[str, List[LayoutEntry]] = {}
        self.entries_by_key: Dict[tuple[str, str], LayoutEntry] = {}
        self._load_manifests()

    @property
    def version(self) -> str:
        return self.split.version

    @property
    def episodes_root(self) -> Path:
        root = self.split.metadata.get("episodes_root")
        if root:
            return as_path(root)
        return as_path(Path("benchmark") / "episodes" / self.version)

    def _load_manifests(self) -> None:
        for manifest_path in self.manifest_paths:
            manifest = read_json(manifest_path)
            scene = str(manifest.get("scene") or manifest_path.parent.parent.name)
            batch_dir = manifest.get("paths", {}).get("batch_dir")
            batch_base = as_path(batch_dir) if batch_dir else manifest_path.parent
            entries: List[LayoutEntry] = []
            for item in manifest.get("layouts", []):
                if item.get("status") not in (None, "ok"):
                    continue
                layout_path_raw = item.get("layout_path")
                if not layout_path_raw:
                    continue
                layout_path = as_path(layout_path_raw)
                if not layout_path.exists():
                    layout_path = batch_base / Path(layout_path_raw).name
                layout_index = int(item.get("layout_index", len(entries)))
                layout_id = layout_path.stem
                entry = LayoutEntry(
                    scene_name=scene,
                    layout_id=layout_id,
                    layout_index=layout_index,
                    layout_path=layout_path,
                    manifest_path=manifest_path,
                    seed=int(item.get("seed", 0) or 0),
                    placed_count=int(item.get("placed_count", 0) or 0),
                    failed_count=int(item.get("failed_count", 0) or 0),
                )
                entries.append(entry)
                self.entries_by_key[(scene, layout_id)] = entry
            entries.sort(key=lambda x: x.layout_index)
            self.entries_by_scene[scene] = entries

    def scenes(self) -> List[str]:
        return sorted(self.entries_by_scene)

    def layouts_for_scene(self, scene_name: str) -> List[LayoutEntry]:
        return list(self.entries_by_scene.get(scene_name, []))

    def resolve_layout(self, scene_name: str, layout_id: str = "", state_index: Optional[int] = None) -> LayoutEntry:
        if layout_id:
            entry = self.entries_by_key.get((scene_name, layout_id))
            if entry:
                return entry
        entries = self.entries_by_scene.get(scene_name, [])
        if state_index is not None:
            for entry in entries:
                if entry.layout_index == int(state_index):
                    return entry
        raise KeyError(f"Cannot resolve layout scene={scene_name!r}, layout_id={layout_id!r}, state_index={state_index!r}")

    def history_layouts(self, scene_name: str, state_index: int, seen_layout_count_before: int) -> List[LayoutEntry]:
        if seen_layout_count_before <= 0:
            return []
        prior = [e for e in self.layouts_for_scene(scene_name) if e.layout_index < int(state_index)]
        prior.sort(key=lambda x: x.layout_index)
        return prior[-int(seen_layout_count_before):]

    def validate_episode_layout(self, episode: Any) -> List[str]:
        warnings: List[str] = []
        try:
            entry = self.resolve_layout(episode.scene_name, episode.layout_id, episode.state_index)
        except Exception as exc:
            return [str(exc)]
        ep_path = Path(episode.scene_state.layout_path)
        if ep_path and ep_path.as_posix() != entry.layout_path.relative_to(as_path(".")).as_posix():
            if ep_path.name != entry.layout_path.name:
                warnings.append(
                    f"Episode {episode.episode_id} references {ep_path}, manifest resolves {entry.layout_path}"
                )
        return warnings


def load_dataset_index(split_manifest_path: Union[str, Path]) -> DatasetIndex:
    return DatasetIndex(split_manifest_path)
