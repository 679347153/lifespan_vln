from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ScenePaths:
    scene_dir: Path
    dataset_config: Path
    stage_glb: Path
    semantic_glb: Optional[Path] = None
    semantic_txt: Optional[Path] = None
    navmesh: Optional[Path] = None


def _find_dataset_config(root: Path, scene_dir: Path) -> Path:
    candidates = [
        scene_dir / "hm3d_annotated_basis.scene_dataset_config.json",
        scene_dir.parent / "hm3d_annotated_val_basis.scene_dataset_config.json",
        scene_dir.parent / "hm3d_annotated_basis.scene_dataset_config.json",
        root / "hm3d_annotated_basis.scene_dataset_config.json",
        root / "val" / "hm3d_annotated_val_basis.scene_dataset_config.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    found = sorted(root.rglob("*scene_dataset_config.json")) if root.exists() else []
    if found:
        return found[0]
    return candidates[0]


def resolve_scene_paths(scene_name: str, require_semantic: bool = False, root: Path = Path("hm3d")) -> Optional[ScenePaths]:
    root = Path(root)
    candidates = [
        root / "minival" / scene_name,
        root / "val" / scene_name,
        root / scene_name,
    ]
    scene_dir = next((p for p in candidates if p.is_dir()), None)
    if scene_dir is None:
        matches = sorted(p for p in root.rglob(scene_name) if p.is_dir()) if root.exists() else []
        scene_dir = matches[0] if matches else None
    if scene_dir is None:
        return None

    glbs = sorted(scene_dir.glob("*.basis.glb"))
    if not glbs:
        glbs = sorted(scene_dir.glob("*.glb"))
    if not glbs:
        return None
    stage_glb = glbs[0]
    semantic_glb = next(iter(sorted(scene_dir.glob("*.semantic.glb"))), None)
    semantic_txt = next(iter(sorted(scene_dir.glob("*.semantic.txt"))), None)
    if require_semantic and (semantic_glb is None or semantic_txt is None):
        return None
    navmesh = next(iter(sorted(scene_dir.glob("*.navmesh"))), None)
    return ScenePaths(
        scene_dir=scene_dir,
        dataset_config=_find_dataset_config(root, scene_dir),
        stage_glb=stage_glb,
        semantic_glb=semantic_glb,
        semantic_txt=semantic_txt,
        navmesh=navmesh,
    )

