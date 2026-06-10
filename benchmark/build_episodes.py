from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .metrics import TASK_TYPES
from .schemas import (
    Episode,
    Pose,
    SceneState,
    SplitManifest,
    Subtask,
    validate_episode,
    write_json,
)

IMAGE_EXTENSIONS = (".webp", ".jpg", ".jpeg", ".png", ".bmp")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    direct = path
    if direct.is_file():
        return direct
    return base_dir / path


def _object_name(obj: Dict[str, Any]) -> str:
    return str(obj.get("name") or obj.get("model_id") or obj.get("id") or "object")


def _object_id(obj: Dict[str, Any]) -> str:
    raw_id = str(obj.get("id", "")).strip()
    model_id = str(obj.get("model_id", "")).strip()
    name = str(obj.get("name", "")).strip()
    base = model_id or name or raw_id or "object"
    if raw_id and raw_id != base:
        return f"{base}#{raw_id}"
    return base


def _unique_object_ids(objects: Sequence[Dict[str, Any]]) -> List[str]:
    counts: Dict[str, int] = {}
    results: List[str] = []
    for obj in objects:
        base = _object_id(obj)
        counts[base] = counts.get(base, 0) + 1
        if counts[base] == 1:
            results.append(base)
        else:
            results.append(f"{base}@{counts[base] - 1}")
    return results


def _object_position(obj: Dict[str, Any]) -> Pose:
    pos = obj.get("position", [0.0, 0.0, 0.0])
    if not isinstance(pos, list) or len(pos) < 3:
        raise ValueError(f"Layout object has invalid position: {_object_id(obj)}")
    rot = obj.get("rotation", [0.0, 0.0, 0.0])
    yaw = float(rot[1]) if isinstance(rot, list) and len(rot) >= 2 else 0.0
    return Pose(float(pos[0]), float(pos[1]), float(pos[2]), yaw)


def _candidate_image_stems(obj: Dict[str, Any]) -> List[str]:
    raw = [_object_name(obj), str(obj.get("model_id", "")), _object_id(obj)]
    stems: List[str] = []
    for item in raw:
        item = Path(item).stem.strip()
        if not item:
            continue
        variants = [item]
        if item.endswith("_4k"):
            variants.append(item[:-3])
        else:
            variants.append(f"{item}_4k")
        for variant in variants:
            if variant and variant not in stems:
                stems.append(variant)
    return stems


def _resolve_image_path(obj: Dict[str, Any], images_dir: Path) -> Optional[Path]:
    if not images_dir.is_dir():
        return None
    for stem in _candidate_image_stems(obj):
        for suffix in IMAGE_EXTENSIONS:
            path = images_dir / f"{stem}{suffix}"
            if path.is_file():
                return path
    lowered = {stem.lower() for stem in _candidate_image_stems(obj)}
    for path in images_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            stem = path.stem.lower()
            if stem in lowered or stem.removesuffix("_4k") in lowered:
                return path
    return None


def _balanced_task_types(count: int, rng: random.Random) -> List[str]:
    types = [TASK_TYPES[i % len(TASK_TYPES)] for i in range(count)]
    rng.shuffle(types)
    return list(types)


def _language_prompt(obj: Dict[str, Any]) -> str:
    name = _object_name(obj)
    region = obj.get("sampled_region_id", "the current room")
    target = obj.get("target_instance_id", "a nearby support surface")
    return (
        f"Go to the {name}, which is usually placed near receptacle instance "
        f"{target} in region {region}."
    )


def _make_subtask(
    episode_id: str,
    subtask_index: int,
    task_type: str,
    obj: Dict[str, Any],
    images_dir: Path,
    success_radius: float,
) -> Subtask:
    name = _object_name(obj)
    obj_id = _object_id(obj)
    image_path = None
    prompt = f"Find {name}."
    language_prompt = None

    if task_type == "image_goal":
        image = _resolve_image_path(obj, images_dir)
        if image is None:
            raise FileNotFoundError(f"Cannot resolve image_goal reference image for {name}")
        image_path = str(image)
        prompt = f"Find the object shown in the reference image: {name}."
    elif task_type == "language_goal":
        language_prompt = _language_prompt(obj)
        prompt = language_prompt

    return Subtask(
        subtask_id=f"{episode_id}_subtask_{subtask_index:02d}",
        task_type=task_type,  # type: ignore[arg-type]
        target_object=name,
        target_object_id=obj_id,
        target_position=_object_position(obj),
        prompt=prompt,
        language_prompt=language_prompt,
        image_path=image_path,
        success_radius=success_radius,
        metadata={
            "layout_object_id": obj.get("id"),
            "model_id": obj.get("model_id", ""),
            "name": obj.get("name", ""),
            "sampled_region_id": obj.get("sampled_region_id"),
            "target_instance_id": obj.get("target_instance_id"),
            "layout_object": obj,
        },
    )


def _sample_object(objects: Sequence[Dict[str, Any]], rng: random.Random) -> Dict[str, Any]:
    if not objects:
        raise ValueError("Cannot sample from an empty object list")
    return dict(rng.choice(list(objects)))


def _layout_id(entry: Dict[str, Any], layout_path: Path, order_index: int) -> str:
    if entry.get("layout_id"):
        return str(entry["layout_id"])
    layout_index = entry.get("layout_index", order_index)
    seed = entry.get("seed", "unknown")
    return f"layout_{int(layout_index):03d}_seed_{seed}" if str(layout_index).isdigit() else layout_path.stem


def _scene_from_manifest(manifest: Dict[str, Any], manifest_path: Path) -> str:
    scene = str(manifest.get("scene_name") or manifest.get("scene") or "").strip()
    if scene:
        return Path(scene).name
    if manifest_path.parent.name.startswith("batch_"):
        return manifest_path.parent.parent.name
    return manifest_path.parent.name


def _iter_layout_entries(manifest: Dict[str, Any], manifest_path: Path) -> Iterable[Dict[str, Any]]:
    base_dir = manifest_path.parent
    layouts = manifest.get("layouts", [])
    if not isinstance(layouts, list):
        raise ValueError(f"manifest layouts must be a list: {manifest_path}")
    for idx, entry in enumerate(layouts):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("status", "ok")) != "ok":
            continue
        layout_value = entry.get("layout_path")
        if not layout_value:
            continue
        layout_path = _resolve_path(str(layout_value), base_dir)
        out = dict(entry)
        out["_layout_path"] = str(layout_path)
        out["_order_index"] = idx
        yield out


def _valid_layout_objects(layout: Dict[str, Any]) -> List[Dict[str, Any]]:
    objects = layout.get("objects", [])
    if not isinstance(objects, list):
        return []
    valid: List[Dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        pos = obj.get("position")
        if isinstance(pos, list) and len(pos) >= 3:
            valid.append(obj)
    return valid


def _make_episode(
    *,
    scene_name: str,
    split: str,
    version: str,
    layout_path: Path,
    layout_id: str,
    state_index: int,
    objects: Sequence[Dict[str, Any]],
    image_objects: Sequence[Dict[str, Any]],
    episode_index: int,
    rng: random.Random,
    min_subtasks: int,
    max_subtasks: int,
    images_dir: Path,
    success_radius: float,
    max_steps: int,
    manifest_path: Path,
) -> Episode:
    subtask_count = rng.randint(min_subtasks, max_subtasks)
    task_types = _balanced_task_types(subtask_count, rng)
    episode_id = f"{version}_{scene_name}_{layout_id}_ep_{episode_index:04d}"
    subtasks: List[Subtask] = []
    for subtask_index, task_type in enumerate(task_types):
        pool = image_objects if task_type == "image_goal" else objects
        obj = _sample_object(pool, rng)
        subtasks.append(
            _make_subtask(
                episode_id,
                subtask_index,
                task_type,
                obj,
                images_dir,
                success_radius,
            )
        )

    ep = Episode(
        episode_id=episode_id,
        split=split,  # type: ignore[arg-type]
        scene_name=scene_name,
        layout_id=layout_id,
        state_index=state_index,
        seen_layout_count_before=state_index,
        scene_state=SceneState(
            state_id=layout_id,
            layout_id=layout_id,
            layout_path=str(layout_path),
            time_index=state_index,
            movable_objects=_unique_object_ids(objects),
        ),
        seed=rng.randrange(0, 2**31),
        start_pose=Pose(0.0, 0.0, 0.0, 0.0),
        max_steps=max_steps,
        subtasks=subtasks,
        metadata={
            "benchmark_version": version,
            "source_manifest": str(manifest_path),
            "layout_object_count": len(objects),
            "task_types": task_types,
            "start_pose_note": "Use benchmark.runner or HabitatLayoutAdapter to sample/overwrite a navigable start pose when Habitat is available.",
        },
    )
    validate_episode(ep)
    return ep


def build_episodes(args: argparse.Namespace) -> Dict[str, Any]:
    rng = random.Random(int(args.seed))
    output_root = Path(args.output_root) / args.version
    split_dir = output_root / args.split
    images_dir = Path(args.images_dir)
    manifest_paths = [Path(x) for x in args.layout_manifest]
    episode_paths: List[Path] = []
    scenes: List[str] = []
    layout_count = 0

    if args.min_subtasks < 5 or args.max_subtasks > 10 or args.min_subtasks > args.max_subtasks:
        raise ValueError("--min-subtasks/--max-subtasks must define a range inside [5, 10]")

    for manifest_path in manifest_paths:
        manifest = _read_json(manifest_path)
        scene_name = _scene_from_manifest(manifest, manifest_path)
        if scene_name not in scenes:
            scenes.append(scene_name)
        for entry in _iter_layout_entries(manifest, manifest_path):
            layout_path = Path(entry["_layout_path"])
            layout = _read_json(layout_path)
            objects = _valid_layout_objects(layout)
            if len(objects) < int(args.min_objects):
                continue
            image_objects = [obj for obj in objects if _resolve_image_path(obj, images_dir) is not None]
            if not image_objects:
                raise FileNotFoundError(
                    f"No image_goal-capable objects found for {layout_path}; "
                    "check --images-dir or generate episodes without image_goal support later."
                )
            state_index = int(entry.get("layout_index", entry.get("_order_index", layout_count)))
            layout_id = _layout_id(entry, layout_path, state_index)
            layout_count += 1

            for local_ep_idx in range(int(args.episodes_per_layout)):
                episode_serial = len(episode_paths)
                ep = _make_episode(
                    scene_name=scene_name,
                    split=args.split,
                    version=args.version,
                    layout_path=layout_path,
                    layout_id=layout_id,
                    state_index=state_index,
                    objects=objects,
                    image_objects=image_objects,
                    episode_index=episode_serial,
                    rng=rng,
                    min_subtasks=int(args.min_subtasks),
                    max_subtasks=int(args.max_subtasks),
                    images_dir=images_dir,
                    success_radius=float(args.success_radius),
                    max_steps=int(args.max_steps),
                    manifest_path=manifest_path,
                )
                out_path = split_dir / scene_name / layout_id / f"{ep.episode_id}.json"
                write_json(out_path, ep)
                episode_paths.append(out_path)

    split_manifest = SplitManifest(
        version=args.version,
        seed=int(args.seed),
        train_scenes=scenes if args.split == "train" else [],
        val_scenes=scenes if args.split == "val" else [],
        test_scenes=scenes if args.split == "test" else [],
        episode_count=len(episode_paths),
        layout_count=layout_count,
        metadata={
            "episodes_root": str(output_root),
            "layout_manifests": [str(x) for x in manifest_paths],
            "images_dir": str(images_dir),
        },
    )
    split_manifest_path = Path("benchmark") / "splits" / f"benchmark_split_{args.version}.json"
    write_json(split_manifest_path, split_manifest)

    return {
        "version": args.version,
        "split": args.split,
        "episodes": len(episode_paths),
        "layouts": layout_count,
        "episodes_root": str(output_root),
        "split_manifest": str(split_manifest_path),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dynamic household object navigation episodes from layout manifests.")
    parser.add_argument("--layout-manifest", nargs="+", required=True, help="One or more results/layouts/<scene>/batch_*/manifest.json files")
    parser.add_argument("--version", default="v2", help="Benchmark version name")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output-root", default="benchmark/episodes")
    parser.add_argument("--images-dir", default="objects_images")
    parser.add_argument("--episodes-per-layout", type=int, default=3)
    parser.add_argument("--min-subtasks", type=int, default=5)
    parser.add_argument("--max-subtasks", type=int, default=10)
    parser.add_argument("--success-radius", type=float, default=1.2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--min-objects", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    summary = build_episodes(parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
