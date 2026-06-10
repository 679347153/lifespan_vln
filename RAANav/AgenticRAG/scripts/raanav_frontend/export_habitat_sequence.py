#!/usr/bin/env python3
"""Export Habitat RGB-D-pose data to DAAAM ImageSequenceDataset format.

This is a data-source adapter for the RAANav final frontend. Habitat is used
only to render RGB-D frames and camera poses; object perception is handled by
`scripts/raanav_frontend/run_object_node_frontend.py` using the same
DAAAM-style segmentation/tracking/assignment/CLIP object-node pipeline that
will also consume rosbag and online ROS frames.

The default export mode is a continuous navmesh trajectory. Tracking frontends
assume temporally adjacent frames; random teleport samples are kept only as an
explicit legacy/debug mode.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import quaternion

_PROJ_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from semantic_map_Create.scene_extract import make_simulator
from scripts.nav_core.habitat_agent import HabitatAgent


def _matrix_to_line(T: np.ndarray) -> str:
    return " ".join(f"{float(v):.8f}" for v in T.reshape(-1))


def _agent_pose_matrix(agent: HabitatAgent) -> np.ndarray:
    state = agent.agent.get_state()
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(state.position, dtype=np.float64)
    T[:3, :3] = quaternion.as_rotation_matrix(state.rotation)
    # DAAAM unprojects pixels into an optical camera frame:
    #   x right, y down, z forward.
    # Habitat/OpenGL camera coordinates are:
    #   x right, y up, -z forward.
    # Convert DAAAM optical points into Habitat agent-local coordinates before
    # applying the agent pose.
    optical_to_habitat = np.eye(4, dtype=np.float64)
    optical_to_habitat[:3, :3] = np.diag([1.0, -1.0, -1.0])
    return T @ optical_to_habitat


def _save_camera_info(out_dir: Path, width: int, height: int, hfov_deg: float) -> None:
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    info = {
        "width": width,
        "height": height,
        "intrinsics": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "distortion": [],
        "source": "habitat",
        "hfov_deg": hfov_deg,
    }
    (out_dir / "camera_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def _write_frame(out_dir: Path, idx: int, obs: dict, pose_line: str, depth_scale: float) -> None:
    color = obs["color"][:, :, :3]
    depth_m = obs["depth"]
    if depth_m.ndim == 3:
        depth_m = depth_m[:, :, 0]
    rgb_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    depth_u16 = np.clip(depth_m * depth_scale, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    cv2.imwrite(str(out_dir / "rgb" / f"{idx:06d}.png"), rgb_bgr)
    cv2.imwrite(str(out_dir / "depth" / f"{idx:06d}.png"), depth_u16)
    with (out_dir / "pose" / "poses.txt").open("a", encoding="utf-8") as f:
        f.write(pose_line + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--dataset-config", default="/home/adminer/agentRAG/experiment_data/hm3d/hm3d_val_scene_dataset_config.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-positions", type=int, default=20)
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--trajectory-mode", choices=["continuous", "teleport_views", "target_views"], default="continuous")
    ap.add_argument("--num-frames", type=int, default=80)
    ap.add_argument("--path-step-size", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--depth-scale", type=float, default=1000.0)
    ap.add_argument("--hfov-deg", type=float, default=90.0)
    ap.add_argument("--same-floor-threshold", type=float, default=1.5)
    ap.add_argument("--target-points-json", default=None, help="JSON target list for target_views mode; each item needs label and pos_3d/position")
    ap.add_argument("--target-camera-radius", type=float, default=1.8, help="Camera radius around each target in target_views mode")
    ap.add_argument("--target-azimuths-deg", default="0,60,120,180,240,300", help="Comma-separated camera azimuths around target")
    ap.add_argument("--target-heading-offsets-deg", default="-20,0,20", help="Comma-separated heading offsets after facing target")
    ap.add_argument("--target-max-snap-distance", type=float, default=1.2, help="Skip snapped camera poses farther than this from requested pose in x/z")
    return ap.parse_args()


def _capture_frame(out_dir: Path, frame_idx: int, sim, agent: HabitatAgent, depth_scale: float, metadata: list, extra: dict) -> None:
    obs = sim.get_sensor_observations()
    T = _agent_pose_matrix(agent)
    _write_frame(out_dir, frame_idx, obs, _matrix_to_line(T), depth_scale)
    metadata.append({
        "frame_id": frame_idx,
        "agent_position": [float(x) for x in agent.get_position()],
        "heading_deg": float(agent.get_heading_deg()),
        **extra,
    })




def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _load_target_points(path: str) -> list[dict]:
    if not path:
        raise ValueError("--target-points-json is required for target_views mode")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "targets" in data:
            data = data["targets"]
        elif "objects" in data:
            data = data["objects"]
        else:
            data = [data]
    targets = []
    for idx, item in enumerate(data):
        pos = item.get("pos_3d") or item.get("position") or item.get("target_position")
        if pos is None and item.get("pos_2d"):
            p2 = item["pos_2d"]
            pos = [p2.get("x"), item.get("y", 0.0), p2.get("y")]
        if pos is None or len(pos) < 3:
            continue
        targets.append({
            "target_id": str(item.get("obj_id") or item.get("target_id") or item.get("id") or f"target_{idx}"),
            "label": str(item.get("label") or item.get("target_label") or "unknown"),
            "pos_3d": [float(pos[0]), float(pos[1]), float(pos[2])],
            "source": item.get("source"),
        })
    if not targets:
        raise ValueError(f"No usable targets found in {path}")
    return targets


def _export_target_views(args, out_dir: Path, sim, agent: HabitatAgent, start) -> tuple[int, list, dict]:
    """Debug/validation mode: render views around known target candidate positions.

    Target positions may come from a prior map or a manually curated list. They
    are not treated as ground truth labels here; they only guide camera poses so
    the frontend can be tested on likely target observations.
    """
    targets = _load_target_points(args.target_points_json)
    azimuths = _parse_float_list(args.target_azimuths_deg)
    heading_offsets = _parse_float_list(args.target_heading_offsets_deg)
    frame_idx = 0
    metadata_positions = []
    first_obs = None

    agent.set_position(start)
    fallback_y = float(agent.get_position()[1])

    for target in targets:
        tx, ty, tz = target["pos_3d"]
        for az in azimuths:
            rad = math.radians(az)
            requested = np.array([
                tx + math.sin(rad) * args.target_camera_radius,
                fallback_y,
                tz + math.cos(rad) * args.target_camera_radius,
            ], dtype=np.float32)
            snapped = sim.pathfinder.snap_point(requested)
            snapped = np.asarray(snapped, dtype=np.float32)
            snap_dist = float(np.linalg.norm((snapped - requested)[[0, 2]]))
            if not sim.pathfinder.is_navigable(snapped) or snap_dist > args.target_max_snap_distance:
                continue
            agent.set_position([float(snapped[0]), float(snapped[1]), float(snapped[2])])
            agent.face_toward([tx, float(snapped[1]), tz])
            base_heading = agent.get_heading_deg()
            for offset in heading_offsets:
                agent.set_heading(base_heading + offset)
                obs = sim.get_sensor_observations()
                if first_obs is None:
                    first_obs = obs
                T = _agent_pose_matrix(agent)
                _write_frame(out_dir, frame_idx, obs, _matrix_to_line(T), args.depth_scale)
                metadata_positions.append({
                    "frame_id": frame_idx,
                    "target_id": target["target_id"],
                    "target_label": target["label"],
                    "target_position": target["pos_3d"],
                    "camera_azimuth_deg": float(az),
                    "heading_offset_deg": float(offset),
                    "snap_distance_xz_m": snap_dist,
                    "agent_position": [float(x) for x in agent.get_position()],
                    "heading_deg": float(agent.get_heading_deg()),
                })
                frame_idx += 1

    if first_obs is None:
        raise RuntimeError("No Habitat target-view observations exported; check target positions and navmesh snapping")
    return frame_idx, metadata_positions, first_obs

def _export_continuous(args, out_dir: Path, sim, agent: HabitatAgent, start) -> tuple[int, list, dict]:
    """Export a temporally continuous path suitable for BotSort/ReID."""
    agent.set_position(start)
    base_y = float(agent.get_position()[1])
    frame_idx = 0
    segment_idx = 0
    metadata_positions = []
    first_obs = None

    while frame_idx < args.num_frames:
        target = agent.get_random_navigable_point(floor_y_threshold=args.same_floor_threshold)
        if abs(float(target[1]) - base_y) > args.same_floor_threshold:
            continue
        waypoints = agent.get_path_waypoints(target.tolist() if hasattr(target, "tolist") else target, step_size=args.path_step_size)
        if not waypoints or len(waypoints) < 2:
            continue

        for wp_idx, wp in enumerate(waypoints):
            if frame_idx >= args.num_frames:
                break
            agent.set_position(wp)
            if wp_idx + 1 < len(waypoints):
                agent.face_toward(waypoints[wp_idx + 1])
            else:
                agent.face_toward(wp)
            obs = sim.get_sensor_observations()
            if first_obs is None:
                first_obs = obs
            T = _agent_pose_matrix(agent)
            _write_frame(out_dir, frame_idx, obs, _matrix_to_line(T), args.depth_scale)
            metadata_positions.append({
                "frame_id": frame_idx,
                "segment_index": segment_idx,
                "waypoint_index": wp_idx,
                "agent_position": [float(x) for x in agent.get_position()],
                "heading_deg": float(agent.get_heading_deg()),
            })
            frame_idx += 1
        segment_idx += 1

    if first_obs is None:
        raise RuntimeError("No Habitat observations exported")
    return frame_idx, metadata_positions, first_obs


def _export_teleport_views(args, out_dir: Path, sim, agent: HabitatAgent, start) -> tuple[int, list, dict]:
    """Legacy/debug mode: random positions with panoramic views."""
    agent.set_position(start)
    base_y = float(agent.get_position()[1])
    frame_idx = 0
    positions = [start]
    for _ in range(max(0, args.num_positions - 1)):
        positions.append(agent.get_random_navigable_point(floor_y_threshold=args.same_floor_threshold))

    first_obs = None
    metadata_positions = []
    for pos_idx, pos in enumerate(positions):
        if abs(float(pos[1]) - base_y) > args.same_floor_threshold:
            continue
        agent.set_position(pos)
        heading0 = agent.get_heading_deg()
        step_deg = 360.0 / args.n_views
        for view_idx in range(args.n_views):
            agent.set_heading(heading0 + view_idx * step_deg)
            obs = sim.get_sensor_observations()
            if first_obs is None:
                first_obs = obs
            T = _agent_pose_matrix(agent)
            _write_frame(out_dir, frame_idx, obs, _matrix_to_line(T), args.depth_scale)
            metadata_positions.append({
                "frame_id": frame_idx,
                "position_index": pos_idx,
                "view_index": view_idx,
                "agent_position": [float(x) for x in agent.get_position()],
                "heading_deg": float(agent.get_heading_deg()),
            })
            frame_idx += 1

    if first_obs is None:
        raise RuntimeError("No Habitat observations exported")
    return frame_idx, metadata_positions, first_obs


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    scene_dir = Path(args.scene_dir)
    scene_name = scene_dir.name
    scene_stem = scene_name.split("-", 1)[1] if "-" in scene_name else scene_name
    basis_glb = scene_dir / f"{scene_stem}.basis.glb"
    if not basis_glb.exists():
        raise FileNotFoundError(basis_glb)

    out_dir = Path(args.out_dir)
    for sub in ["rgb", "depth", "pose"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    pose_file = out_dir / "pose" / "poses.txt"
    if pose_file.exists():
        pose_file.unlink()

    sim = make_simulator(str(basis_glb), args.dataset_config, enable_semantic=False, enable_depth=True)
    agent = HabitatAgent(sim, use_navmesh=True)

    try:
        start = agent.get_random_navigable_point()
        if args.trajectory_mode == "continuous":
            frame_idx, metadata_positions, first_obs = _export_continuous(args, out_dir, sim, agent, start)
        elif args.trajectory_mode == "target_views":
            frame_idx, metadata_positions, first_obs = _export_target_views(args, out_dir, sim, agent, start)
        else:
            frame_idx, metadata_positions, first_obs = _export_teleport_views(args, out_dir, sim, agent, start)

        if first_obs is None:
            raise RuntimeError("No Habitat observations exported")
        h, w = first_obs["color"].shape[:2]
        _save_camera_info(out_dir, w, h, args.hfov_deg)
        (out_dir / "metadata.json").write_text(json.dumps({
            "source": "habitat",
            "trajectory_mode": args.trajectory_mode,
            "scene_dir": str(scene_dir),
            "dataset_config": args.dataset_config,
            "frames": frame_idx,
            "n_views": args.n_views,
            "path_step_size": args.path_step_size,
            "depth_scale": args.depth_scale,
            "target_points_json": args.target_points_json,
            "positions": metadata_positions,
        }, indent=2), encoding="utf-8")
        print(f"Exported {frame_idx} RGB-D-pose frames to {out_dir}")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
