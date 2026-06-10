#!/usr/bin/env python3
"""
Frontend-Only DAAAM Pipeline (No Hydra / No Spark-DSG / No VLM Grounding).

Runs segmentation + tracking on an ImageSequenceDataset and outputs:
  1. Per-frame JSON: list of {track_id, semantic_id, pos_3d_camera, pos_3d_world, bbox, median_depth}
  2. Per-frame label visualization image (color-coded masks)
  3. Aggregate summary JSON with all tracks across all frames

Usage:
  LD_PRELOAD=/tmp/libvitjot_stub.so python AgenticRAG/scripts/daaam/run_frontend_only_pipeline.py \
      --data-path /tmp/daaam_apartment \
      --output-dir /tmp/daaam_frontend_output \
      --max-frames 50
"""

import sys
import os
import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from abc import ABC, abstractmethod

# ── DAAAM imports ──────────────────────────────────────────────
DAAAM_ROOT = Path("/home/adminer/agentRAG/参考开源代码/DAAAM")
DAAAM_SRC = DAAAM_ROOT / "src"
sys.path.insert(0, str(DAAAM_SRC))

import daaam
daaam.ROOT_DIR = DAAAM_ROOT
import daaam.config
daaam.config.ROOT_DIR = DAAAM_ROOT
import daaam.segmentation.services
daaam.segmentation.services.ROOT_DIR = DAAAM_ROOT
import daaam.tracking.services
daaam.tracking.services.ROOT_DIR = DAAAM_ROOT

from daaam.config import PipelineConfig
from daaam.datasets.loaders.image_sequence import ImageSequenceDataset
from daaam.segmentation import SegmentationService
from daaam.tracking import TrackingService
from daaam.utils.geometry import (
    compute_mask_centroid,
    unproject_pixel_to_3d,
    pose_to_matrix,
)
from daaam.utils.vision import bounding_box_from_mask
from daaam.utils.logging import get_default_logger


# ═══════════════════════════════════════════════════════════════
# PoseProvider — modular world-frame transform (swappable for ROS TF)
# ═══════════════════════════════════════════════════════════════

class PoseProvider(ABC):
    """Abstract interface for converting camera-frame 3D points to world frame.

    Design: Offline we read from pose/poses.txt (DatasetPoseProvider).
            Online we swap to a ROSTFPoseProvider that subscribes to /tf
            and maintains a cached tf2 buffer — same interface, zero
            changes to the frontend loop.
    """

    @abstractmethod
    def camera_to_world(
        self,
        point_camera: np.ndarray,      # [x, y, z] in camera frame
        timestamp: float,
        frame_transform: Optional[np.ndarray] = None,  # 7D pose [x,y,z,qx,qy,qz,qw] from dataset
    ) -> np.ndarray:
        """Return point in world frame as np.ndarray([x, y, z])."""
        ...

    @property
    @abstractmethod
    def has_poses(self) -> bool:
        """Whether pose information is available (i.e. world ≠ camera frame)."""
        ...


class IdentityPoseProvider(PoseProvider):
    """Fallback: world == camera (no pose data available)."""

    def camera_to_world(self, point_camera, timestamp, frame_transform=None):
        return point_camera

    @property
    def has_poses(self):
        return False


class DatasetPoseProvider(PoseProvider):
    """Reads camera→world transform from the dataset's 7D pose array.

    The dataset stores each pose as [x, y, z, qx, qy, qz, qw]
    representing world_T_camera.  This provider converts it to a 4×4
    matrix and transforms the 3D point.
    """

    def __init__(self):
        self._pose_count = 0

    def camera_to_world(self, point_camera, timestamp, frame_transform=None):
        if frame_transform is None:
            return point_camera  # no pose for this frame → passthrough

        world_T_camera = pose_to_matrix(frame_transform)
        point_homo = np.append(point_camera, 1.0)
        return (world_T_camera @ point_homo)[:3]

    @property
    def has_poses(self):
        return True


# ═══════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════

def extract_camera_intrinsics(camera_info: Optional[dict]) -> Optional[Dict[str, float]]:
    if camera_info is None:
        return None
    K = camera_info.get('intrinsics')
    if K is None:
        return None
    return {
        'fx': float(K[0][0]), 'fy': float(K[1][1]),
        'cx': float(K[0][2]), 'cy': float(K[1][2]),
    }


# ═══════════════════════════════════════════════════════════════
# Core: track processing with pluggable pose provider
# ═══════════════════════════════════════════════════════════════

def process_tracks(
    frame_data,                # DatasetFrame
    tracks: np.ndarray,        # M×8 from tracker
    masks: List[np.ndarray],
    object_labels: Dict[int, int],
    current_id: int,
    camera_intrinsics: Optional[Dict[str, float]],
    pose_provider: PoseProvider,
    depth_lb: float = 0.25,
    depth_ub: float = 5.0,
    logger=None,
) -> Tuple[List[dict], Dict[int, int], int]:
    """Replicate the frontend portion of PipelineOrchestrator._process_tracks().

    All camera→world transforms go through *pose_provider* so the same
    loop works offline (DatasetPoseProvider) and online (ROSTFPoseProvider).
    """
    results = []

    for track_row in tracks:
        track_id = int(track_row[4])
        mask_idx = int(track_row[7])

        if mask_idx >= len(masks):
            continue

        mask = masks[mask_idx].astype(bool)

        # ── Depth validation ──
        track_is_depth_valid = True
        median_depth = 0.0
        if frame_data.depth_image is not None:
            mask_sum = mask.sum()
            depth_values = frame_data.depth_image[mask]
            valid_count = np.sum(depth_values > 0)
            if valid_count < 0.25 * mask_sum:
                track_is_depth_valid = False
            else:
                median_depth = float(np.median(depth_values[depth_values > 0]))
                track_is_depth_valid = (depth_lb <= median_depth <= depth_ub)

        bbox = bounding_box_from_mask(mask)

        # ── Semantic label ──
        if track_id not in object_labels:
            object_labels[track_id] = current_id
            current_id += 1
        semantic_id = object_labels[track_id]

        # ── 3D position (camera frame) ──
        pos_3d_camera = None
        pos_3d_world = None
        centroid_pixel = None

        if track_is_depth_valid and camera_intrinsics and median_depth > 0:
            centroid = compute_mask_centroid(mask)
            if centroid:
                u, v = centroid
                centroid_pixel = [int(u), int(v)]
                point_camera = unproject_pixel_to_3d(
                    u, v, median_depth,
                    camera_intrinsics['fx'], camera_intrinsics['fy'],
                    camera_intrinsics['cx'], camera_intrinsics['cy']
                )
                pos_3d_camera = point_camera.tolist()

                # ── World frame via modular PoseProvider ──
                pos_3d_world = pose_provider.camera_to_world(
                    point_camera,
                    float(frame_data.timestamp),
                    frame_data.transform,
                ).tolist()

        results.append({
            "track_id": track_id,
            "semantic_id": semantic_id,
            "bbox": [int(x) for x in bbox],
            "median_depth": float(median_depth),
            "depth_valid": track_is_depth_valid,
            "pos_3d_camera": pos_3d_camera,
            "pos_3d_world": pos_3d_world,
            "centroid_pixel": centroid_pixel,
            "mask_area_pixels": int(mask.sum()),
        })

    return results, object_labels, current_id


# ═══════════════════════════════════════════════════════════════
# Visualization helpers
# ═══════════════════════════════════════════════════════════════

def create_label_image(
    tracks_results: List[dict],
    masks: List[np.ndarray],
    tracks_array: np.ndarray,
    image_shape: Tuple[int, int],
    object_labels: Dict[int, int],
) -> Tuple[np.ndarray, Dict[int, list]]:
    h, w = image_shape
    label_image = np.zeros((h, w), dtype=np.uint16)
    rng = np.random.RandomState(42)
    color_map = {}

    for track_row, result in zip(tracks_array, tracks_results):
        mask_idx = int(track_row[7])
        if mask_idx < len(masks):
            mask = masks[mask_idx].astype(bool)
            semantic_id = result["semantic_id"]
            label_image[mask] = semantic_id
            if semantic_id not in color_map:
                color_map[semantic_id] = rng.randint(0, 255, 3).tolist()

    return label_image, color_map


def save_visualization(
    label_image: np.ndarray,
    color_map: Dict[int, list],
    output_path: Path,
    rgb_image: Optional[np.ndarray] = None,
):
    import cv2
    h, w = label_image.shape
    vis = rgb_image.copy() if rgb_image is not None else np.zeros((h, w, 3), dtype=np.uint8)
    alpha = 0.4
    for sem_id, color in color_map.items():
        mask = (label_image == sem_id)
        if mask.any():
            vis[mask] = (vis[mask] * (1 - alpha) + np.array(color, dtype=np.uint8) * alpha).astype(np.uint8)
    cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DAAAM Frontend-Only Pipeline")
    parser.add_argument("--data-path", required=True, help="Path to ImageSequenceDataset")
    parser.add_argument("--output-dir", default="/tmp/daaam_frontend_output")
    parser.add_argument("--config", default=None, help="Path to pipeline_config.yaml")
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--without-reid", action="store_true", default=True)
    parser.add_argument("--save-vis", action="store_true", default=True)
    parser.add_argument("--vis-interval", type=int, default=5, help="Save vis every N frames")
    parser.add_argument("--pose-provider", default="dataset",
                        choices=["dataset", "identity"],
                        help="'dataset' reads pose/poses.txt; 'identity' sets world=camera")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(exist_ok=True)
    (output_dir / "vis").mkdir(exist_ok=True)

    logger = get_default_logger()

    # ── Config ──
    config_path = args.config or str(DAAAM_ROOT / "config" / "pipeline_config.yaml")
    config = PipelineConfig.from_yaml(config_path)
    config.segmentation.device = args.device
    config.tracking.device = args.device
    config.tracking.with_reid = not args.without_reid
    config.segmentation.model_name = "fastsam/FastSAM-s.pt"
    config.segmentation.model_config_path = "fastsam/fastsam_config.yaml"
    config.segmentation.imgsz = None
    logger.info(f"Config: device={args.device}, model={config.segmentation.model_name}")

    # ── Dataset ──
    logger.info(f"Loading dataset from {args.data_path}")
    dataset = ImageSequenceDataset(
        data_path=Path(args.data_path),
        depth_scale=args.depth_scale,
        compute_velocities=False,
    )
    logger.info(f"Dataset loaded: {len(dataset)} frames")
    camera_info = dataset.get_camera_info()
    camera_intrinsics = extract_camera_intrinsics(camera_info)
    logger.info(f"Camera info: {camera_info}")
    logger.info(f"Camera intrinsics: {camera_intrinsics}")

    # ── PoseProvider (modular — swap for ROS TF later) ──
    if args.pose_provider == "dataset":
        pose_provider = DatasetPoseProvider()
    else:
        pose_provider = IdentityPoseProvider()
    logger.info(f"Pose provider: {type(pose_provider).__name__} (has_poses={pose_provider.has_poses})")

    # Check first frame for pose availability
    has_pose_data = False
    if len(dataset) > 0:
        first = dataset[0]
        has_pose_data = first.transform is not None
    logger.info(f"Pose data present in dataset: {has_pose_data}")

    # ── Services ──
    logger.info("Initializing segmentation service...")
    seg_service = SegmentationService(config.segmentation, logger)
    logger.info("Initializing tracking service...")
    trk_service = TrackingService(config.tracking, logger)

    logger.info("Warming up...")
    seg_service.warmup()
    logger.info("Warmup complete")

    # ── State ──
    object_labels: Dict[int, int] = {}
    current_id = 1
    all_frame_results: List[dict] = []
    track_summary: Dict[int, dict] = defaultdict(lambda: {
        "track_id": 0,
        "semantic_id": 0,
        "first_seen_frame": None,
        "last_seen_frame": None,
        "total_observations": 0,
        "positions_world": [],
        "positions_camera": [],
        "bboxes": [],
        "median_depths": [],
    })

    n_frames = len(dataset)
    if args.max_frames:
        n_frames = min(n_frames, args.max_frames)
    logger.info(f"Processing {n_frames} frames...")

    start_time = time.time()

    for i in range(n_frames):
        frame_data = dataset[i]
        rgb = frame_data.rgb_image

        # 1. Segment
        t0 = time.time()
        dets, masks = seg_service.segment(rgb)
        t_seg = time.time() - t0

        # 2. Track
        t0 = time.time()
        tracks_array = trk_service.update(dets, rgb)
        t_trk = time.time() - t0

        # 3. Process tracks (3D positions via PoseProvider)
        frame_results, object_labels, current_id = process_tracks(
            frame_data=frame_data,
            tracks=tracks_array,
            masks=masks,
            object_labels=object_labels,
            current_id=current_id,
            camera_intrinsics=camera_intrinsics,
            pose_provider=pose_provider,
            depth_lb=config.depth.depth_lb,
            depth_ub=config.depth.depth_ub,
            logger=logger,
        )

        # 4. Visualization
        if frame_results:
            label_img, color_map = create_label_image(
                frame_results, masks, tracks_array, rgb.shape[:2], object_labels,
            )
        else:
            label_img = np.zeros(rgb.shape[:2], dtype=np.uint16)
            color_map = {}

        # 5. Save per-frame JSON
        frame_entry = {
            "frame_id": i,
            "timestamp": float(frame_data.timestamp),
            "n_detections": len(dets),
            "n_tracks": len(tracks_array),
            "tracks": frame_results,
            "timing": {"segmentation_ms": round(t_seg * 1000, 1), "tracking_ms": round(t_trk * 1000, 1)},
        }
        all_frame_results.append(frame_entry)
        with open(output_dir / "labels" / f"{i:06d}.json", "w") as f:
            json.dump(frame_entry, f, indent=2)

        if args.save_vis and i % args.vis_interval == 0:
            save_visualization(label_img, color_map, output_dir / "vis" / f"{i:06d}.png", rgb)

        # 6. Update track summary
        for tr in frame_results:
            tid = tr["track_id"]
            ts = track_summary[tid]
            ts["track_id"] = tid
            ts["semantic_id"] = tr["semantic_id"]
            if ts["first_seen_frame"] is None:
                ts["first_seen_frame"] = i
            ts["last_seen_frame"] = i
            ts["total_observations"] += 1
            if tr["pos_3d_world"] is not None:
                ts["positions_world"].append(tr["pos_3d_world"])
            if tr["pos_3d_camera"] is not None:
                ts["positions_camera"].append(tr["pos_3d_camera"])
            ts["bboxes"].append(tr["bbox"])
            if tr["median_depth"] > 0:
                ts["median_depths"].append(tr["median_depth"])

        elapsed = time.time() - start_time
        fps = (i + 1) / elapsed if elapsed > 0 else 0
        n_positions = sum(1 for t in frame_results if t["pos_3d_world"] is not None)
        print(f"\rFrame {i+1}/{n_frames} | dets={len(dets)} tracks={len(tracks_array)} "
              f"pos3d={n_positions} | seg={t_seg*1000:.0f}ms trk={t_trk*1000:.0f}ms | {fps:.1f} fps",
              end="", flush=True)
    print()

    # ── Summary ──
    summary_out = {}
    for tid, ts in track_summary.items():
        summary_out[str(tid)] = {
            "track_id": ts["track_id"],
            "semantic_id": ts["semantic_id"],
            "first_seen_frame": ts["first_seen_frame"],
            "last_seen_frame": ts["last_seen_frame"],
            "total_observations": ts["total_observations"],
            "avg_depth": float(np.mean(ts["median_depths"])) if ts["median_depths"] else None,
            "avg_position_world": (
                np.mean(ts["positions_world"], axis=0).tolist()
                if ts["positions_world"] else None
            ),
            "avg_position_camera": (
                np.mean(ts["positions_camera"], axis=0).tolist()
                if ts["positions_camera"] else None
            ),
            "positions_world_count": len(ts["positions_world"]),
            "position_world_std_m": (
                np.std(ts["positions_world"], axis=0).tolist()
                if len(ts["positions_world"]) > 1 else None
            ),
        }

    summary = {
        "dataset_path": str(args.data_path),
        "total_frames_processed": n_frames,
        "total_tracks": len(summary_out),
        "total_unique_semantic_ids": len(set(ts["semantic_id"] for ts in track_summary.values())),
        "has_world_poses": has_pose_data,
        "pose_provider": type(pose_provider).__name__,
        "config": {
            "model": config.segmentation.model_name,
            "device": args.device,
            "depth_scale": args.depth_scale,
        },
        "camera_info": camera_info,
        "tracks": summary_out,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "all_frames.json", "w") as f:
        json.dump(all_frame_results, f, indent=2)

    total_time = time.time() - start_time
    print(f"Done. Processed {n_frames} frames in {total_time:.1f}s ({n_frames/total_time:.1f} fps)")
    print(f"Total tracks: {len(summary_out)}, Unique semantic IDs: {summary['total_unique_semantic_ids']}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
