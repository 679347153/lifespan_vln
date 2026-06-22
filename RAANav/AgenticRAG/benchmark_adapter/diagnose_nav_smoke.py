from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_objects_from_map(data: Any) -> Iterable[Dict[str, Any]]:
    floors = data if isinstance(data, list) else data.get("floors", [])
    for floor in floors or []:
        for room in floor.get("rooms", []) or []:
            for obj in room.get("objects", []) or []:
                if isinstance(obj, dict):
                    yield obj


def _label_counts(output_dir: Path) -> Counter:
    counts: Counter = Counter()
    for name in ["map_live.json", "map_final.json", "map_local_rag.json"]:
        path = output_dir / name
        if not path.exists():
            continue
        data = _read_json(path)
        if name == "map_local_rag.json" and isinstance(data, dict):
            objects = data.get("objects", [])
        else:
            objects = list(_iter_objects_from_map(data))
        for obj in objects:
            label = str(obj.get("label", "")).strip().lower()
            if label:
                counts[label] += 1
    return counts


def _query_dirs(output_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(output_dir.glob("query_step*")):
        files = sorted(p.name for p in path.iterdir()) if path.is_dir() else []
        rows.append({"path": str(path), "files": files, "has_artifacts": bool(files)})
    return rows


def diagnose(output_dir: Path, target: str = "") -> Dict[str, Any]:
    output_dir = output_dir.resolve()
    counts = _label_counts(output_dir)
    target_norm = target.strip().lower()
    nav_result_path = output_dir / "nav_result.json"
    nav_result = _read_json(nav_result_path) if nav_result_path.exists() else {}
    recommended = [
        {"label": label, "count": count}
        for label, count in counts.most_common(10)
        if label not in {"wall", "floor", "ceiling", "unknown", "objects"}
    ]
    query_artifacts = _query_dirs(output_dir)
    return {
        "output_dir": str(output_dir),
        "target": target_norm or None,
        "target_observed_count": counts.get(target_norm, 0) if target_norm else None,
        "recommended_smoke_targets": recommended,
        "n_labels": len(counts),
        "n_objects_from_maps": sum(counts.values()),
        "query_artifacts": query_artifacts,
        "nav_result": {
            "target_found": nav_result.get("target_found"),
            "target_found_step": nav_result.get("target_found_step"),
            "total_map_objects": nav_result.get("total_map_objects"),
            "total_distance_m": nav_result.get("total_distance_m"),
        },
        "diagnosis": _diagnosis_text(target_norm, counts, query_artifacts),
    }


def _diagnosis_text(target: str, counts: Counter, query_artifacts: List[Dict[str, Any]]) -> str:
    if target and counts.get(target, 0) == 0:
        return (
            f"Target '{target}' is absent from the generated maps. "
            "Do not treat target failure as a navigation-quality result; choose a recommended_smoke_target first."
        )
    if not query_artifacts:
        return "No query_step artifacts found; the run likely did not produce a probability-field diagnostic."
    return "Smoke output is diagnosable. Check query_step artifacts and use a target with nonzero observed count."


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose a legacy Habitat navigation smoke output directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="")
    args = parser.parse_args()
    print(json.dumps(diagnose(Path(args.output_dir), args.target), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
