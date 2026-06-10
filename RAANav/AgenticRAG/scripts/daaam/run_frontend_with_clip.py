#!/usr/bin/env python3
"""
Enhanced DAAAM Frontend Pipeline with CLIP Zero-Shot Labeling.

Replaces GroundingDINO: FastSAM gives class-agnostic masks, CLIP assigns
semantic labels by classifying each masked crop against 65 indoor categories.

Usage:
  LD_PRELOAD=/tmp/libvitjot_stub.so conda run -n daaam_p0 python \
      AgenticRAG/scripts/daaam/run_frontend_with_clip.py \
      --data-path /tmp/daaam_apartment \
      --output-dir /tmp/daaam_clip_output \
      --max-frames 30
"""

import sys, os, argparse, json, time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np

# ── Paths ──
DAAAM_ROOT = Path("/home/adminer/agentRAG/参考开源代码/DAAAM")
sys.path.insert(0, str(DAAAM_ROOT / "src"))

import daaam; daaam.ROOT_DIR = DAAAM_ROOT
import daaam.config; daaam.config.ROOT_DIR = DAAAM_ROOT
import daaam.segmentation.services; daaam.segmentation.services.ROOT_DIR = DAAAM_ROOT
import daaam.tracking.services; daaam.tracking.services.ROOT_DIR = DAAAM_ROOT

from daaam.config import PipelineConfig
from daaam.datasets.loaders.image_sequence import ImageSequenceDataset
from daaam.segmentation import SegmentationService
from daaam.tracking import TrackingService
from daaam.utils.geometry import compute_mask_centroid, unproject_pixel_to_3d, pose_to_matrix
from daaam.utils.vision import bounding_box_from_mask
from daaam.utils.logging import get_default_logger

# Our module (in same directory)
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from clip_labeler import CLIPZeroShotLabeler


# ═══════════════════════════════════════════════════════════════
# PoseProvider (same modular interface)
# ═══════════════════════════════════════════════════════════════

class PoseProvider:
    def camera_to_world(self, pt, ts, ft=None): return pt
    @property
    def has_poses(self): return False

class IdentityPoseProvider(PoseProvider): pass

class DatasetPoseProvider(PoseProvider):
    def camera_to_world(self, pt, ts, ft=None):
        if ft is None: return pt
        T = pose_to_matrix(ft)
        return (T @ np.append(pt, 1.0))[:3]
    @property
    def has_poses(self): return True


# ═══════════════════════════════════════════════════════════════
# Core pipeline (same as before + CLIP labeling)
# ═══════════════════════════════════════════════════════════════

def extract_camera_intrinsics(camera_info):
    if not camera_info: return None
    K = camera_info.get('intrinsics')
    if not K: return None
    return {'fx': float(K[0][0]), 'fy': float(K[1][1]),
            'cx': float(K[0][2]), 'cy': float(K[1][2])}


def process_frame(
    frame_data, tracks_array, masks,
    object_labels, current_id,
    camera_intrinsics, pose_provider, clip_labeler,
    track_label_accum,  # mutable: {track_id: [labels]}
    depth_lb=0.25, depth_ub=5.0,
    label_every_n_frames=5,
    frame_idx=0,
    logger=None,
):
    results = []

    # CLIP labeling for this frame (every N frames to save time)
    frame_labels = None
    if clip_labeler is not None and frame_idx % label_every_n_frames == 0:
        rgb_for_label = frame_data.rgb_image
        frame_labels = clip_labeler.label_masks(rgb_for_label, [m.astype(bool) for m in masks])

    for track_row in tracks_array:
        track_id = int(track_row[4])
        mask_idx = int(track_row[7])
        if mask_idx >= len(masks): continue
        mask = masks[mask_idx].astype(bool)

        # Depth
        track_is_depth_valid = True
        median_depth = 0.0
        if frame_data.depth_image is not None:
            depth_vals = frame_data.depth_image[mask]
            valid = np.sum(depth_vals > 0)
            if valid < 0.25 * mask.sum():
                track_is_depth_valid = False
            else:
                median_depth = float(np.median(depth_vals[depth_vals > 0]))
                track_is_depth_valid = (depth_lb <= median_depth <= depth_ub)

        bbox = bounding_box_from_mask(mask)

        # Semantic ID
        if track_id not in object_labels:
            object_labels[track_id] = current_id
            current_id += 1
        sem_id = object_labels[track_id]

        # CLIP label accumulation
        if frame_labels is not None and mask_idx < len(frame_labels):
            fl = frame_labels[mask_idx]
            if fl['confidence'] > 0.05:  # only accumulate meaningful labels
                track_label_accum[track_id].append({
                    'label': fl['label'],
                    'confidence': fl['confidence'],
                    'clip_embedding': fl['clip_embedding'],
                })

        # 3D position
        pos_3d_camera = None
        pos_3d_world = None
        centroid_pixel = None
        if track_is_depth_valid and camera_intrinsics and median_depth > 0:
            centroid = compute_mask_centroid(mask)
            if centroid:
                u, v = centroid
                centroid_pixel = [int(u), int(v)]
                pc = unproject_pixel_to_3d(u, v, median_depth,
                    camera_intrinsics['fx'], camera_intrinsics['fy'],
                    camera_intrinsics['cx'], camera_intrinsics['cy'])
                pos_3d_camera = pc.tolist()
                pos_3d_world = pose_provider.camera_to_world(
                    pc, float(frame_data.timestamp), frame_data.transform).tolist()

        results.append({
            "track_id": track_id, "semantic_id": sem_id,
            "bbox": [int(x) for x in bbox],
            "median_depth": float(median_depth), "depth_valid": track_is_depth_valid,
            "pos_3d_camera": pos_3d_camera, "pos_3d_world": pos_3d_world,
            "centroid_pixel": centroid_pixel, "mask_area_pixels": int(mask.sum()),
        })

    return results, object_labels, current_id


def finalize_track_labels(track_label_accum):
    """Pick best label for each track from accumulated per-frame labels."""
    final_labels = {}
    for track_id, entries in track_label_accum.items():
        if not entries:
            final_labels[track_id] = {"label": f"obj", "confidence": 0.0, "clip_embedding": None}
            continue
        # Most common label
        label_counts = Counter(e['label'] for e in entries)
        best_label = label_counts.most_common(1)[0][0]
        # Best confidence entry for that label
        best = max((e for e in entries if e['label'] == best_label),
                   key=lambda e: e['confidence'])
        final_labels[track_id] = best
    return final_labels


# ═══════════════════════════════════════════════════════════════
# Visualization (simple OpenCV-based for real-time preview)
# ═══════════════════════════════════════════════════════════════

def save_frame_visualization(rgb, masks, labels, output_path, alpha=0.4):
    """Overlay colored masks with CLIP labels on RGB image."""
    import cv2
    H, W = rgb.shape[:2]
    vis = rgb.copy()
    rng = np.random.RandomState(42)

    for i, (mask, lbl) in enumerate(zip(masks, labels)):
        if mask is None: continue
        m = mask.astype(bool)
        if not m.any(): continue
        color = rng.randint(0, 255, 3).tolist()
        vis[m] = (vis[m] * (1 - alpha) + np.array(color, dtype=np.uint8) * alpha).astype(np.uint8)

        # Label text at mask centroid
        ys, xs = np.where(m)
        if len(ys) > 0:
            cy, cx = int(np.median(ys)), int(np.median(xs))
            cv2.putText(vis, lbl.get('label', '?')[:12],
                        (cx - 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default="/tmp/daaam_clip_output")
    parser.add_argument("--config", default=None)
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pose-provider", default="dataset", choices=["dataset","identity"])
    parser.add_argument("--no-clip", action="store_true", help="Disable CLIP labeling")
    parser.add_argument("--label-interval", type=int, default=5,
                        help="Run CLIP labeling every N frames")
    parser.add_argument("--save-vis", action="store_true", default=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)
    (out / "vis").mkdir(exist_ok=True)

    log = get_default_logger()

    # Config
    cfg_path = args.config or str(DAAAM_ROOT / "config" / "pipeline_config.yaml")
    config = PipelineConfig.from_yaml(cfg_path)
    config.segmentation.device = args.device
    config.tracking.device = args.device
    config.tracking.with_reid = False
    config.segmentation.model_name = "fastsam/FastSAM-s.pt"
    config.segmentation.model_config_path = "fastsam/fastsam_config.yaml"
    config.segmentation.imgsz = None

    # Dataset
    log.info(f"Loading {args.data_path}")
    dataset = ImageSequenceDataset(Path(args.data_path), depth_scale=args.depth_scale,
                                   compute_velocities=False)
    log.info(f"{len(dataset)} frames")
    cam_info = dataset.get_camera_info()
    K = extract_camera_intrinsics(cam_info)
    log.info(f"Intrinsics: {K}")

    # Pose
    pp = DatasetPoseProvider() if args.pose_provider == "dataset" else IdentityPoseProvider()
    has_poses = len(dataset) > 0 and dataset[0].transform is not None
    log.info(f"PoseProvider: {type(pp).__name__}, has_poses={has_poses}")

    # Services
    seg_svc = SegmentationService(config.segmentation, log)
    trk_svc = TrackingService(config.tracking, log)
    seg_svc.warmup()
    log.info("Warmup done")

    # CLIP Labeler
    clip_labeler = None
    if not args.no_clip:
        log.info("Loading CLIP labeler (ViT-B-16)...")
        clip_labeler = CLIPZeroShotLabeler(device=args.device)
        log.info("CLIP labeler ready")

    # State
    object_labels = {}
    current_id = 1
    track_label_accum = defaultdict(list)  # {track_id: [{label, confidence, clip_embedding}]}
    all_frame_results = []
    track_summary = defaultdict(lambda: {
        "track_id": 0, "semantic_id": 0,
        "first_seen_frame": None, "last_seen_frame": None,
        "total_observations": 0,
        "positions_world": [], "positions_camera": [],
        "bboxes": [], "median_depths": [],
    })

    n = len(dataset)
    if args.max_frames: n = min(n, args.max_frames)
    log.info(f"Processing {n} frames (CLIP every {args.label_interval} frames)...")

    t_start = time.time()
    for i in range(n):
        fd = dataset[i]
        rgb = fd.rgb_image

        t0 = time.time()
        dets, masks = seg_svc.segment(rgb)
        t_seg = time.time() - t0

        t0 = time.time()
        tracks_array = trk_svc.update(dets, rgb)
        t_trk = time.time() - t0

        fr, object_labels, current_id = process_frame(
            fd, tracks_array, masks, object_labels, current_id,
            K, pp, clip_labeler, track_label_accum,
            depth_lb=config.depth.depth_lb, depth_ub=config.depth.depth_ub,
            label_every_n_frames=args.label_interval, frame_idx=i, logger=log,
        )

        # Save per-frame JSON
        fe = {"frame_id": i, "timestamp": float(fd.timestamp),
              "n_detections": len(dets), "n_tracks": len(tracks_array),
              "tracks": fr,
              "timing": {"seg_ms": round(t_seg*1000,1), "trk_ms": round(t_trk*1000,1)}}
        all_frame_results.append(fe)
        with open(out / "labels" / f"{i:06d}.json", "w") as f:
            json.dump(fe, f, indent=2)

        # Visualization with CLIP labels (if available)
        if args.save_vis and i % max(1, args.label_interval) == 0 and clip_labeler is not None:
            # Get labels for the masks visible in this frame
            frame_labels_for_vis = clip_labeler.label_masks(rgb, [m.astype(bool) for m in masks])
            save_frame_visualization(rgb, masks, frame_labels_for_vis,
                                     out / "vis" / f"{i:06d}.png")

        # Track summary
        for tr in fr:
            tid = tr["track_id"]
            ts = track_summary[tid]
            ts["track_id"] = tid; ts["semantic_id"] = tr["semantic_id"]
            if ts["first_seen_frame"] is None: ts["first_seen_frame"] = i
            ts["last_seen_frame"] = i; ts["total_observations"] += 1
            if tr["pos_3d_world"]: ts["positions_world"].append(tr["pos_3d_world"])
            if tr["pos_3d_camera"]: ts["positions_camera"].append(tr["pos_3d_camera"])
            ts["bboxes"].append(tr["bbox"])
            if tr["median_depth"] > 0: ts["median_depths"].append(tr["median_depth"])

        elapsed = time.time() - t_start
        fps = (i+1)/elapsed if elapsed > 0 else 0
        n_pos = sum(1 for t in fr if t["pos_3d_world"])
        print(f"\r{i+1}/{n} | dets={len(dets)} trks={len(tracks_array)} "
              f"pos3d={n_pos} | seg={t_seg*1000:.0f}ms trk={t_trk*1000:.0f}ms | {fps:.1f}fps",
              end="", flush=True)
    print()

    # ── Finalize labels per track ──
    log.info("Finalizing track labels...")
    final_labels = finalize_track_labels(track_label_accum)
    n_labeled = sum(1 for v in final_labels.values() if v['confidence'] > 0)
    log.info(f"Labeled tracks: {n_labeled}/{len(final_labels)}")

    # ── Summary ──
    summary_out = {}
    for tid, ts in track_summary.items():
        lbl = final_labels.get(tid, {"label": "obj", "confidence": 0.0, "clip_embedding": None})
        summary_out[str(tid)] = {
            "track_id": ts["track_id"],
            "semantic_id": ts["semantic_id"],
            "label": lbl["label"],
            "label_confidence": lbl["confidence"],
            "clip_embedding": lbl.get("clip_embedding"),
            "first_seen_frame": ts["first_seen_frame"],
            "last_seen_frame": ts["last_seen_frame"],
            "total_observations": ts["total_observations"],
            "avg_depth": float(np.mean(ts["median_depths"])) if ts["median_depths"] else None,
            "avg_position_world": (np.mean(ts["positions_world"], axis=0).tolist()
                                   if ts["positions_world"] else None),
            "avg_position_camera": (np.mean(ts["positions_camera"], axis=0).tolist()
                                    if ts["positions_camera"] else None),
            "positions_world_count": len(ts["positions_world"]),
        }

    summary = {
        "dataset_path": str(args.data_path),
        "total_frames": n,
        "total_tracks": len(summary_out),
        "has_world_poses": has_poses,
        "pose_provider": type(pp).__name__,
        "clip_model": "ViT-B-16" if clip_labeler else "none",
        "config": {"model": config.segmentation.model_name,
                   "device": args.device, "depth_scale": args.depth_scale},
        "camera_info": cam_info,
        "tracks": summary_out,
    }

    with open(out / "summary.json", "w") as f: json.dump(summary, f, indent=2)
    with open(out / "all_frames.json", "w") as f: json.dump(all_frame_results, f, indent=2)

    total_t = time.time() - t_start
    print(f"Done. {n} frames in {total_t:.1f}s ({n/total_t:.1f} fps)")
    print(f"Tracks: {len(summary_out)}, Labeled: {n_labeled}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
