#!/usr/bin/env python3
"""
DAAAM → RAANav Bridge: 将 DAAAM 前端输出转换为 RAANav 语义地图 + GMM 热力图。

Input:  DAAAM summary.json (per-track avg_position_world, semantic_id, observations)
Output:
  1. RAANav-compatible semantic_map.json (Floor → Room → Object)
  2. GMM 概率热力图 (object_gmm_heatmap.png)
  3. Object 散点图 (object_scatter.png)

用法:
  cd /home/adminer/agentRAG/AgenticRAG
  conda run -n agentrag python scripts/daaam/daaam_to_raanav_map.py \
      --daaam-summary /tmp/daaam_world_test/summary.json \
      --output-dir /tmp/raanav_p1_test
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure RAANav root is on path
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

# ═══════════════════════════════════════════════════════════════
# Step 1: Load DAAAM summary, build RAANav semantic map
# ═══════════════════════════════════════════════════════════════

def _make_obj_id(track_id: int) -> str:
    return f"daaam_obj_{track_id}"

def build_semantic_map(summary: dict, min_label_confidence: float = 0.2, label_gate_mode: str = "soft") -> List[dict]:
    """Convert DAAAM summary.json tracks into a RAANav Floor→Room→Object structure.

    Since DAAAM has no room/floor segmentation, we assign all objects to a single
    default room (R0) on a single default floor (F0).

    Each object gets:
      - obj_id: f"daaam_obj_{track_id}"
      - label: f"obj_{semantic_id}" (no VLM → numeric placeholder)
      - pos_3d: [x, y, z] world
      - pos_2d: {"x": world_x, "y": world_z}  (RAANav uses y=world_z convention)
      - stability: 0.5 (neutral — no prior knowledge)
      - cfd: min(1.0, observations/30) (confidence grows with observations)
      - exist_prob: 1.0
      - N: observation count
      - region: axis-aligned bounding box polygon around the object position
    """
    tracks = summary.get("object_nodes") or summary.get("tracks", {})
    objects = []

    for track_id_str, track in tracks.items():
        if track.get("map_eligible") is False:
            continue
        avg_pos = track.get("avg_position_world")
        if avg_pos is None:
            continue

        wx, wy, wz = float(avg_pos[0]), float(avg_pos[1]), float(avg_pos[2])
        n_obs = int(track.get("total_observations", 1))
        sem_id = int(track.get("semantic_id", 0))
        avg_depth = track.get("avg_depth")
        # Soft gate by default: keep the candidate label available for target
        # queries, but expose confidence/status so RAANav scoring, stability and
        # negative feedback can suppress weak false positives. Hard mode is kept
        # only for conservative ablations where weak labels should be hidden.
        raw_label = track.get("label", f"obj_{sem_id}")
        label_conf = float(track.get("label_confidence", 0.0) or 0.0)
        semantic_status = "confirmed" if label_conf >= min_label_confidence else "low_confidence"
        if label_gate_mode == "hard" and semantic_status == "low_confidence":
            label = "unknown"
        else:
            label = raw_label
        clip_emb = track.get("clip_embedding")

        # Simple axis-aligned bounding box region (~20cm around center at 1m depth)
        # Scale with depth to account for perspective
        half_extent = 0.15 * (1.0 + (avg_depth or 1.0) * 0.3)
        region = [
            {"x": float(wx - half_extent), "y": float(wz - half_extent)},
            {"x": float(wx + half_extent), "y": float(wz - half_extent)},
            {"x": float(wx + half_extent), "y": float(wz + half_extent)},
            {"x": float(wx - half_extent), "y": float(wz + half_extent)},
        ]

        objects.append({
            "obj_id": _make_obj_id(track_id_str),
            "label": label,
            "raw_label": raw_label,
            "label_confidence": label_conf,
            "label_topk": track.get("label_topk", []),
            "semantic_status": semantic_status,
            "label_gate_mode": label_gate_mode,
            "pos_3d": [wx, wy, wz],
            "pos_2d": {"x": wx, "y": wz},
            "region": region,
            "stability": 0.5,
            "cfd": max(0.1, min(1.0, n_obs / 30.0) * (0.5 + 0.5 * label_conf)),
            "exist_prob": 1.0,
            "N": n_obs,
            "R_objs": {},
            "clip_embedding": clip_emb if clip_emb else [],
            "imgs": {},
            "description": {},
            "last_update_time": "2026-05-06T00:00:00Z",
        })

    floor = {
        "floor_id": "F0",
        "rooms": [{
            "room_id": "F0_R0",
            "objects": objects,
            "region": [],
            "room_name": {"1": "Default Room"},
            "imgs": {},
            "description": {},
            "N": 1,
            "floor_id": "F0",
        }],
        "z_range": [0.0, 5.0],
    }

    return [floor]


# ═══════════════════════════════════════════════════════════════
# Step 2: Build synthetic occupancy grid from object extents
# ═══════════════════════════════════════════════════════════════

def build_occ_grid(objects: List[dict], resolution: float = 0.05, margin: float = 2.0) -> Tuple[np.ndarray, dict]:
    """Create a synthetic occupancy grid covering the object extents.

    Since we have no habitat simulation, all cells are marked FREE (1) so the
    GMM can be placed anywhere.  Real deployment will use the live OCC grid.
    """
    xs = [o["pos_2d"]["x"] for o in objects]
    zs = [o["pos_2d"]["y"] for o in objects]

    if not xs:
        xs = [-5, 5]
        zs = [-5, 5]

    x_min, x_max = min(xs) - margin, max(xs) + margin
    z_min, z_max = min(zs) - margin, max(zs) + margin

    H = int(math.ceil((z_max - z_min) / resolution))
    W = int(math.ceil((x_max - x_min) / resolution))

    grid = np.ones((max(10, H), max(10, W)), dtype=np.uint8)  # all FREE

    meta = {
        "resolution": resolution,
        "origin_x": float(x_min),
        "origin_z": float(z_min),
        "shape": list(grid.shape),
    }
    return grid, meta


# ═══════════════════════════════════════════════════════════════
# Step 3: Score all objects for GMM (no CLIP needed — direct scoring)
# ═══════════════════════════════════════════════════════════════

def score_all_objects(objects: List[dict], config_path: str) -> Dict[str, Any]:
    """Score every object using RAANav's Calculate_obj_Score and build GMM inputs."""
    from pathlib import Path
    from GMM_map_Create.GMM_map_calcualte import Calculate_obj_Score

    gmm_scores = {}
    gmm_positions = {}

    for obj in objects:
        oid = obj["obj_id"]
        score = Calculate_obj_Score(
            N=obj.get("N", 1),
            stability=obj.get("stability", 0.5),
            cfd=obj.get("cfd"),
            config_path=Path(config_path) if isinstance(config_path, str) else config_path,
        )
        gmm_scores[oid] = float(score)
        gmm_positions[oid] = [obj["pos_2d"]["x"], obj["pos_2d"]["y"]]

    return {
        "gmm_scores": gmm_scores,
        "gmm_positions": gmm_positions,
        "n_objects": len(objects),
    }


# ═══════════════════════════════════════════════════════════════
# Step 4: Build GMM probability field (adapted from query_e2e.py)
# ═══════════════════════════════════════════════════════════════

def build_gmm_field(
    gmm_scores: Dict[str, float],
    gmm_positions: Dict[str, List[float]],
    grid: np.ndarray,
    grid_meta: dict,
    sigma_base: float = 1.0,
    score_amplify: float = 3.0,
) -> np.ndarray:
    """Build 2D Gaussian mixture probability field."""
    H, W = grid.shape
    resolution = grid_meta["resolution"]
    origin_x = grid_meta["origin_x"]
    origin_z = grid_meta["origin_z"]

    prob = np.zeros((H, W), dtype=np.float64)
    sigma_px = sigma_base / resolution

    for oid, score in gmm_scores.items():
        pos = gmm_positions.get(oid)
        if pos is None:
            continue
        wx, wz = pos[0], pos[1]
        c = int(round((wx - origin_x) / resolution))
        r = int(round((wz - origin_z) / resolution))
        if r < 0 or r >= H or c < 0 or c >= W:
            continue

        amp = score * score_amplify
        win = int(math.ceil(3 * sigma_px))
        r0, r1 = max(0, r - win), min(H, r + win + 1)
        c0, c1 = max(0, c - win), min(W, c + win + 1)
        yy, xx = np.mgrid[r0:r1, c0:c1]
        prob[r0:r1, c0:c1] += amp * np.exp(
            -((yy - r) ** 2 + (xx - c) ** 2) / (2 * sigma_px ** 2)
        )

    return prob


# ═══════════════════════════════════════════════════════════════
# Step 5: Visualization
# ═══════════════════════════════════════════════════════════════

def visualize_gmm(
    prob: np.ndarray,
    grid: np.ndarray,
    grid_meta: dict,
    objects: List[dict],
    output_dir: str,
    sigma: float,
):
    """Draw GMM heatmap + object scatter overlay."""
    import cv2

    H, W = prob.shape
    resolution = grid_meta["resolution"]
    origin_x = grid_meta["origin_x"]
    origin_z = grid_meta["origin_z"]

    # Base layer: white for free space
    base = np.full((H, W, 3), 255, dtype=np.uint8)

    # Heatmap overlay (jet colormap)
    if prob.max() > 0:
        prob_norm = (prob / prob.max() * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(prob_norm, cv2.COLORMAP_HOT)
        mask = prob_norm > 1
        alpha = 0.65
        for ch in range(3):
            base[:, :, ch] = np.where(
                mask,
                (alpha * heatmap[:, :, ch] + (1 - alpha) * base[:, :, ch]).astype(np.uint8),
                base[:, :, ch],
            )

    # Object positions as circles
    for obj in objects:
        pos = obj["pos_2d"]
        wx, wz = pos["x"], pos["y"]
        c = int(round((wx - origin_x) / resolution))
        r = int(round((wz - origin_z) / resolution))
        if 0 <= r < H and 0 <= c < W:
            radius = max(2, int(obj["cfd"] * 5))
            cv2.circle(base, (c, r), radius, (0, 120, 0), -1)
            cv2.circle(base, (c, r), radius + 1, (0, 0, 0), 1)

    # Title + grid info
    cv2.putText(base, f"RAANav GMM | {len(objects)} objects | sigma={sigma}m",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(base, f"range: [{origin_x:.1f}, {origin_z:.1f}] → "
                f"[{origin_x+W*resolution:.1f}, {origin_z+H*resolution:.1f}]m",
                (5, H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)

    out_path = os.path.join(output_dir, "object_gmm_heatmap.png")
    cv2.imwrite(out_path, base)
    print(f"  [VIS] GMM heatmap: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DAAAM → RAANav Bridge + GMM")
    parser.add_argument("--daaam-summary", required=True, help="Path to DAAAM summary.json")
    parser.add_argument("--output-dir", default="/tmp/raanav_p1_test")
    parser.add_argument("--sigma", type=float, default=1.0, help="GMM Gaussian sigma (m)")
    parser.add_argument("--config", default="config/map.yaml", help="RAANav config path (for GMM params)")
    parser.add_argument("--min-label-confidence", type=float, default=0.2, help="Confidence threshold used to mark semantic_status as confirmed vs low_confidence")
    parser.add_argument("--label-gate-mode", choices=["soft", "hard"], default="soft", help="soft keeps weak labels queryable with low cfd/status; hard exposes weak labels as unknown")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── Load DAAAM data ──
    print(f"[1/4] Loading DAAAM summary: {args.daaam_summary}")
    with open(args.daaam_summary, "r") as f:
        summary = json.load(f)
    print(f"  → {summary['total_tracks']} tracks")

    # ── Build semantic map ──
    print("[2/4] Building RAANav semantic map...")
    floors = build_semantic_map(summary, min_label_confidence=args.min_label_confidence, label_gate_mode=args.label_gate_mode)
    n_objects = len(floors[0]["rooms"][0]["objects"])
    map_path = os.path.join(output_dir, "semantic_map.json")
    with open(map_path, "w") as f:
        json.dump(floors, f, ensure_ascii=False, indent=2)
    print(f"  → {n_objects} objects → {map_path}")

    # ── Build synthetic OCC grid ──
    print("[3/4] Building occupancy grid + GMM...")
    objects = floors[0]["rooms"][0]["objects"]
    grid, grid_meta = build_occ_grid(objects)
    print(f"  → Grid: {grid_meta['shape']} @ {grid_meta['resolution']}m")

    # ── Score + GMM ──
    score_data = score_all_objects(objects, args.config)
    print(f"  → Scored {score_data['n_objects']} objects")

    prob = build_gmm_field(
        score_data["gmm_scores"],
        score_data["gmm_positions"],
        grid, grid_meta,
        sigma_base=args.sigma,
    )
    print(f"  → GMM field max: {prob.max():.4f}")

    # ── Visualize ──
    print("[4/4] Visualizing...")
    heatmap_path = visualize_gmm(prob, grid, grid_meta, objects, output_dir, args.sigma)

    # ── Save GMM data ──
    gmm_data = {
        "source": args.daaam_summary,
        "n_objects": n_objects,
        "sigma": args.sigma,
        "gmm_field_max": float(prob.max()),
        "gmm_field_mean": float(prob.mean()),
        "grid_meta": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in grid_meta.items()},
    }
    with open(os.path.join(output_dir, "gmm_result.json"), "w") as f:
        json.dump(gmm_data, f, ensure_ascii=False, indent=2)

    print(f"\nDone.")
    print(f"  Output: {output_dir}")
    print(f"    - {map_path}")
    print(f"    - {heatmap_path}")
    print(f"    - {output_dir}/gmm_result.json")


if __name__ == "__main__":
    main()
