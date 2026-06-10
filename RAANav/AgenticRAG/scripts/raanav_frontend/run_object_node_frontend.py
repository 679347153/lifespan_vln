#!/usr/bin/env python3
"""RAANav final object-node frontend.

This is the main perception frontend after retiring GroundingDINO/MobileSAM.
It follows DAAAM's production idea while emitting RAANav-friendly object nodes:

  RGB-D-pose sequence
    -> FastSAM segmentation every frame
    -> BotSort tracking every frame
    -> assignment: keep representative high-quality observations per track
    -> CLIP object-node embedding/top-k label from representative crops
    -> summary.json consumable by scripts/daaam/daaam_to_raanav_map.py

DAM/VLM descriptions are intentionally not in the core path. The long-term
map learns from object nodes, temporal statistics, positions, embeddings and
co-occurrence rather than heavy natural-language descriptions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

DAAAM_ROOT = Path("/home/adminer/agentRAG/参考开源代码/DAAAM")
DAAAM_SRC = DAAAM_ROOT / "src"
if str(DAAAM_SRC) not in sys.path:
    sys.path.insert(0, str(DAAAM_SRC))

import daaam
import daaam.config
import daaam.segmentation.services
import daaam.tracking.services

daaam.ROOT_DIR = DAAAM_ROOT
daaam.config.ROOT_DIR = DAAAM_ROOT
daaam.segmentation.services.ROOT_DIR = DAAAM_ROOT
daaam.tracking.services.ROOT_DIR = DAAAM_ROOT

from daaam.config import PipelineConfig
from daaam.datasets.loaders.image_sequence import ImageSequenceDataset
from daaam.segmentation import SegmentationService
from daaam.tracking import TrackingService
from daaam.utils.geometry import compute_mask_centroid, unproject_pixel_to_3d, pose_to_matrix
from daaam.utils.vision import bounding_box_from_mask
from daaam.utils.logging import get_default_logger

_SCRIPT_DIR = Path(__file__).resolve().parent
_DAAAM_SCRIPTS = _SCRIPT_DIR.parent / "daaam"
if str(_DAAAM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_DAAAM_SCRIPTS))
from clip_labeler import CLIPZeroShotLabeler

WINDOW_TITLE = "RAANav Object-Node Frontend"


def extract_camera_intrinsics(camera_info: Optional[dict]) -> Optional[Dict[str, float]]:
    if not camera_info:
        return None
    K = camera_info.get("intrinsics")
    if K is None:
        return None
    return {
        "fx": float(K[0][0]), "fy": float(K[1][1]),
        "cx": float(K[0][2]), "cy": float(K[1][2]),
    }


def camera_to_world(point_camera: np.ndarray, frame_transform: Optional[np.ndarray]) -> Optional[List[float]]:
    if frame_transform is None:
        return point_camera.astype(float).tolist()
    T = pose_to_matrix(frame_transform)
    return (T @ np.append(point_camera, 1.0))[:3].astype(float).tolist()


def process_track_row(frame_data, track_row, masks, intrinsics, depth_lb: float, depth_ub: float) -> Optional[dict]:
    track_id = int(track_row[4])
    mask_idx = int(track_row[7])
    if mask_idx < 0 or mask_idx >= len(masks):
        return None
    mask = masks[mask_idx].astype(bool)
    if mask.sum() <= 0:
        return None

    bbox = bounding_box_from_mask(mask)
    median_depth = 0.0
    depth_valid = False
    pos_cam = None
    pos_world = None
    centroid_pixel = None

    if frame_data.depth_image is not None and intrinsics is not None:
        depth_values = frame_data.depth_image[mask]
        valid = depth_values[(depth_values > depth_lb) & (depth_values < depth_ub)]
        if len(valid) > 0:
            median_depth = float(np.median(valid))
            centroid = compute_mask_centroid(mask)
            if centroid:
                u, v = centroid
                centroid_pixel = [int(u), int(v)]
                pc = unproject_pixel_to_3d(
                    u, v, median_depth,
                    intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"],
                )
                pos_cam = pc.astype(float).tolist()
                pos_world = camera_to_world(pc, frame_data.transform)
                depth_valid = True

    return {
        "track_id": track_id,
        "mask_idx": mask_idx,
        "bbox": [int(x) for x in bbox],
        "mask_area_pixels": int(mask.sum()),
        "median_depth": float(median_depth),
        "depth_valid": bool(depth_valid),
        "pos_3d_camera": pos_cam,
        "pos_3d_world": pos_world,
        "centroid_pixel": centroid_pixel,
    }


def save_masked_crop(rgb: np.ndarray, mask: np.ndarray, path: Path, pad_ratio: float = 0.1, crop_mode: str = "context") -> Optional[dict]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    h, w = max(1, y2 - y1), max(1, x2 - x1)
    pad_y, pad_x = max(2, int(h * pad_ratio)), max(2, int(w * pad_ratio))
    H, W = rgb.shape[:2]
    y1, y2 = max(0, y1 - pad_y), min(H, y2 + pad_y + 1)
    x1, x2 = max(0, x1 - pad_x), min(W, x2 + pad_x + 1)
    crop = rgb[y1:y2, x1:x2].copy()
    crop_mode = crop_mode.lower()
    if crop_mode == "masked":
        crop_mask = mask[y1:y2, x1:x2]
        crop[~crop_mask] = 0
    elif crop_mode != "context":
        raise ValueError(f"Unsupported representative crop mode: {crop_mode}")
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    return {"path": str(path), "bbox_crop": [x1, y1, x2, y2], "crop_size": [int(x2 - x1), int(y2 - y1)], "crop_mode": crop_mode}


def maybe_add_representative(reps: List[dict], candidate: dict, max_reps: int) -> None:
    reps.append(candidate)
    reps.sort(key=lambda x: (x.get("mask_area_pixels", 0), x.get("depth_valid", False)), reverse=True)
    del reps[max_reps:]


def label_representatives(labeler: CLIPZeroShotLabeler, crop_paths: List[str], top_k: int = 5) -> dict:
    valid_images = []
    valid_paths = []
    for p in crop_paths:
        img = Image.open(p).convert("RGB")
        if img.size[0] < 16 or img.size[1] < 16:
            continue
        valid_images.append(img)
        valid_paths.append(p)
    if not valid_images:
        return {"label": "unknown", "label_confidence": 0.0, "label_topk": [], "clip_embedding": None, "labeled_crop_paths": []}

    tensors = torch.stack([labeler.preprocess(img) for img in valid_images]).to(labeler.device)
    with torch.no_grad():
        image_features = labeler.model.encode_image(tensors)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        pooled = image_features.mean(dim=0, keepdim=True)
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        sims = pooled @ labeler.text_features.T
        if hasattr(labeler.model, "logit_scale"):
            probs = (sims * labeler.model.logit_scale.exp()).softmax(dim=-1)[0]
        else:
            probs = (sims * 100.0).softmax(dim=-1)[0]
        vals, idxs = torch.topk(probs, k=min(top_k, probs.numel()))

    top = []
    for score, idx in zip(vals.cpu().tolist(), idxs.cpu().tolist()):
        top.append({"label": labeler.canonical_labels[int(idx)], "confidence": float(score)})
    return {
        "label": top[0]["label"] if top else "unknown",
        "label_confidence": top[0]["confidence"] if top else 0.0,
        "label_topk": top,
        "clip_embedding": pooled.cpu().numpy()[0].astype(float).tolist(),
        "labeled_crop_paths": valid_paths,
    }


def build_spatial_object_nodes(observations: List[dict], radius_m: float, min_obs: int, max_reps: int, labeler: CLIPZeroShotLabeler, top_k: int) -> Dict[str, dict]:
    clusters: List[dict] = []
    for obs in observations:
        pos = obs.get("pos_3d_world")
        if pos is None:
            continue
        p = np.asarray(pos, dtype=np.float64)
        best_idx = None
        best_dist = float("inf")
        for idx, cluster in enumerate(clusters):
            c = np.asarray(cluster["centroid_world"], dtype=np.float64)
            d = float(np.linalg.norm((p - c)[[0, 2]]))
            if d < best_dist:
                best_dist = d
                best_idx = idx
        if best_idx is None or best_dist > radius_m:
            clusters.append({
                "observations": [obs],
                "positions_world": [p.tolist()],
                "centroid_world": p.tolist(),
                "representatives": [],
            })
        else:
            cluster = clusters[best_idx]
            cluster["observations"].append(obs)
            cluster["positions_world"].append(p.tolist())
            cluster["centroid_world"] = np.mean(cluster["positions_world"], axis=0).astype(float).tolist()

    nodes = {}
    for idx, cluster in enumerate(clusters):
        obs_list = cluster["observations"]
        if len(obs_list) < min_obs:
            continue
        positions = np.asarray(cluster["positions_world"], dtype=np.float64)
        std = np.std(positions, axis=0).astype(float).tolist() if len(positions) > 1 else [0.0, 0.0, 0.0]
        reps = []
        for obs in sorted(obs_list, key=lambda o: o.get("mask_area_pixels", 0), reverse=True):
            p = obs.get("representative_crop_path")
            if p and Path(p).exists():
                reps.append({
                    "path": p,
                    "frame_id": obs.get("frame_id"),
                    "track_id": obs.get("track_id"),
                    "mask_area_pixels": obs.get("mask_area_pixels", 0),
                    "depth_valid": obs.get("depth_valid", False),
                })
            if len(reps) >= max_reps:
                break
        lbl = label_representatives(labeler, [r["path"] for r in reps], top_k=top_k) if reps else {
            "label": "unknown", "label_confidence": 0.0, "label_topk": [], "clip_embedding": None, "labeled_crop_paths": [],
        }
        node_id = f"node_{idx}"
        nodes[node_id] = {
            "node_id": node_id,
            "track_ids": sorted({int(o.get("track_id", -1)) for o in obs_list if o.get("track_id") is not None}),
            "total_observations": len(obs_list),
            "first_seen_frame": int(min(o.get("frame_id", 0) for o in obs_list)),
            "last_seen_frame": int(max(o.get("frame_id", 0) for o in obs_list)),
            "avg_position_world": np.mean(positions, axis=0).astype(float).tolist(),
            "position_world_std_m": std,
            "position_world_std_xz_m": float((std[0] ** 2 + std[2] ** 2) ** 0.5),
            "representatives": reps,
            "map_eligible": True,
            "quality_reasons": [],
            **lbl,
        }
    return nodes


def normalize_tracks_array(tracks) -> np.ndarray:
    """Normalize DAAAM/BotSort output so empty frames never break downstream indexing."""
    if tracks is None:
        return np.empty((0, 8), dtype=np.float32)
    arr = np.asarray(tracks)
    if arr.size == 0:
        return np.empty((0, 8), dtype=np.float32)
    if arr.ndim == 1:
        if arr.shape[0] >= 8:
            return arr.reshape(1, -1)
        return np.empty((0, 8), dtype=np.float32)
    if arr.shape[1] < 8:
        padded = np.zeros((arr.shape[0], 8), dtype=arr.dtype)
        padded[:, :arr.shape[1]] = arr
        return padded
    return arr


def save_frame_vis(rgb: np.ndarray, masks: List[np.ndarray], tracks_array: np.ndarray, path: Path) -> None:
    vis = rgb.copy()
    rng = np.random.default_rng(42)
    for row in tracks_array:
        mask_idx = int(row[7])
        if mask_idx < 0 or mask_idx >= len(masks):
            continue
        color = rng.integers(40, 255, size=3, dtype=np.uint8)
        mask = masks[mask_idx].astype(bool)
        vis[mask] = (0.55 * vis[mask] + 0.45 * color).astype(np.uint8)
        x1, y1, x2, y2 = [int(v) for v in row[:4]]
        tid = int(row[4])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color.tolist(), 1)
        cv2.putText(vis, f"T{tid}", (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color.tolist(), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def filter_object_like_masks(
    dets: np.ndarray,
    masks: List[np.ndarray],
    max_area_ratio: float,
    max_edge_area_ratio: float,
    max_bbox_aspect_ratio: float,
    min_bbox_fill_ratio: float,
) -> Tuple[np.ndarray, List[np.ndarray], dict]:
    if len(masks) == 0 or len(dets) == 0:
        return dets, masks, {"input": len(masks), "kept": len(masks), "dropped_large": 0, "dropped_edge": 0, "dropped_shape": 0}
    if len(dets) != len(masks):
        return dets, masks, {"input": len(masks), "kept": len(masks), "dropped_large": 0, "dropped_edge": 0, "dropped_shape": 0, "warning": "dets_masks_len_mismatch"}

    H, W = masks[0].shape[:2]
    image_area = float(H * W)
    keep = []
    dropped_large = 0
    dropped_edge = 0
    dropped_shape = 0
    for idx, mask in enumerate(masks):
        m = mask.astype(bool)
        area_ratio = float(m.sum()) / image_area
        if area_ratio > max_area_ratio:
            dropped_large += 1
            continue
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bw = max(1, x2 - x1 + 1)
        bh = max(1, y2 - y1 + 1)
        aspect = max(bw / bh, bh / bw)
        fill = float(m.sum()) / float(bw * bh)
        if aspect > max_bbox_aspect_ratio or fill < min_bbox_fill_ratio:
            dropped_shape += 1
            continue
        touches_edge = x1 <= 2 or y1 <= 2 or x2 >= W - 3 or y2 >= H - 3
        if touches_edge and area_ratio > max_edge_area_ratio:
            dropped_edge += 1
            continue
        keep.append(idx)

    if not keep:
        return np.empty((0, dets.shape[1]), dtype=dets.dtype), [], {
            "input": len(masks), "kept": 0, "dropped_large": dropped_large, "dropped_edge": dropped_edge, "dropped_shape": dropped_shape,
        }
    return dets[keep], [masks[i] for i in keep], {
        "input": len(masks), "kept": len(keep), "dropped_large": dropped_large, "dropped_edge": dropped_edge, "dropped_shape": dropped_shape,
    }


def color_for_track(track_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(int(track_id) * 7919 + 17)
    color = rng.integers(50, 255, size=3, dtype=np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def render_track_overlay(rgb: np.ndarray, masks: List[np.ndarray], tracks_array: np.ndarray, width: int, height: int) -> np.ndarray:
    vis = rgb.copy()
    for row in tracks_array:
        mask_idx = int(row[7])
        if mask_idx < 0 or mask_idx >= len(masks):
            continue
        tid = int(row[4])
        color = color_for_track(tid)
        mask = masks[mask_idx].astype(bool)
        vis[mask] = (0.56 * vis[mask] + 0.44 * np.array(color, dtype=np.uint8)).astype(np.uint8)
        x1, y1, x2, y2 = [int(v) for v in row[:4]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"T{tid}", (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return cv2.resize(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), (width, height))


def world_xy_from_track(track: dict) -> Optional[Tuple[float, float]]:
    pos = track.get("avg_position_world")
    if pos is None or len(pos) < 3:
        pts = track.get("positions_world") or []
        if not pts:
            return None
        arr = np.asarray(pts, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 3:
            return None
        pos = arr.mean(axis=0).tolist()
    return float(pos[0]), float(pos[2])


def render_topdown_panel(track_summary: Dict[int, dict], frame_transform: Optional[np.ndarray], width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    pts = []
    for tid, ts in track_summary.items():
        xy = world_xy_from_track(ts)
        if xy is not None:
            pts.append((tid, xy[0], xy[1], int(ts.get("total_observations", 0))))

    agent_xy = None
    if frame_transform is not None:
        try:
            T = pose_to_matrix(frame_transform)
            agent_xy = (float(T[0, 3]), float(T[2, 3]))
        except Exception:
            agent_xy = None

    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    if agent_xy:
        xs.append(agent_xy[0])
        ys.append(agent_xy[1])
    if not xs:
        cv2.putText(panel, "No 3D object nodes yet", (22, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (90, 90, 90), 2)
        return panel

    pad_m = 0.8
    x0, x1 = min(xs) - pad_m, max(xs) + pad_m
    y0, y1 = min(ys) - pad_m, max(ys) + pad_m
    if abs(x1 - x0) < 1e-3:
        x0 -= 1.0
        x1 += 1.0
    if abs(y1 - y0) < 1e-3:
        y0 -= 1.0
        y1 += 1.0

    def to_px(x: float, y: float) -> Tuple[int, int]:
        px = int((x - x0) / (x1 - x0) * (width - 40) + 20)
        py = int(height - ((y - y0) / (y1 - y0) * (height - 50) + 30))
        return px, py

    for tid, x, y, n_obs in pts:
        px, py = to_px(x, y)
        bgr = tuple(reversed(color_for_track(tid)))
        radius = int(np.clip(4 + n_obs // 2, 5, 14))
        cv2.circle(panel, (px, py), radius, bgr, -1)
        cv2.circle(panel, (px, py), radius + 1, (30, 30, 30), 1)
        cv2.putText(panel, f"T{tid}:{n_obs}", (px + 6, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (25, 25, 25), 1)

    if agent_xy:
        ax, ay = to_px(*agent_xy)
        cv2.drawMarker(panel, (ax, ay), (0, 0, 255), markerType=cv2.MARKER_TRIANGLE_UP, markerSize=18, thickness=2)
        cv2.putText(panel, "camera", (ax + 8, ay + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 180), 1)

    cv2.putText(panel, "Top-down object nodes (x/z)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1)
    cv2.putText(panel, f"range x[{x0:.1f},{x1:.1f}] z[{y0:.1f},{y1:.1f}]", (8, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)
    return panel


def render_representatives_panel(track_summary: Dict[int, dict], width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.putText(panel, "Representative crops / stable tracks", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1)
    rows = []
    for tid, ts in sorted(track_summary.items(), key=lambda kv: kv[1].get("total_observations", 0), reverse=True):
        reps = ts.get("representatives") or []
        if reps:
            rows.append((tid, ts, reps[0].get("path")))
        if len(rows) >= 4:
            break
    if not rows:
        cv2.putText(panel, "Waiting for representative crops...", (18, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 1)
        return panel

    cell_h = max(60, (height - 34) // len(rows))
    for i, (tid, ts, img_path) in enumerate(rows):
        y = 32 + i * cell_h
        crop = None
        if img_path and Path(img_path).exists():
            crop = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if crop is None:
            crop = np.zeros((cell_h - 8, cell_h - 8, 3), dtype=np.uint8)
        crop = cv2.resize(crop, (cell_h - 8, cell_h - 8))
        panel[y:y + cell_h - 8, 8:cell_h] = crop
        obs = int(ts.get("total_observations", 0))
        npos = len(ts.get("positions_world") or [])
        txt_x = cell_h + 12
        cv2.putText(panel, f"T{tid} obs={obs} pos3d={npos}", (txt_x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1)
        if len(ts.get("positions_world") or []) > 1:
            std = np.std(ts["positions_world"], axis=0)
            cv2.putText(panel, f"std xyz=({std[0]:.2f},{std[1]:.2f},{std[2]:.2f})m", (txt_x, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (170, 220, 255), 1)
    return panel


def render_stats_panel(frame_idx: int, n_frames: int, dets, tracks, frame_tracks: List[dict], timing: dict, track_summary: Dict[int, dict], t_start: float, width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 18, dtype=np.uint8)
    elapsed = max(1e-6, time.time() - t_start)
    fps = (frame_idx + 1) / elapsed
    total_tracks = len(track_summary)
    stable_tracks = sum(1 for t in track_summary.values() if int(t.get("total_observations", 0)) >= 3)
    pos_tracks = sum(1 for t in track_summary.values() if t.get("positions_world"))
    lines = [
        "RAANav object-node frontend",
        f"frame: {frame_idx + 1}/{n_frames}   fps: {fps:.2f}",
        f"detections: {len(dets)}   active tracks: {len(tracks)}",
        f"total tracks: {total_tracks}   stable>=3: {stable_tracks}",
        f"tracks with 3D pos: {pos_tracks}",
        f"seg: {timing['segmentation_ms']:.1f} ms   track: {timing['tracking_ms']:.1f} ms",
        f"current pos3d observations: {sum(1 for t in frame_tracks if t.get('pos_3d_world') is not None)}",
        "Core path: FastSAM -> BotSort -> reps -> CLIP -> RAANav map",
        "No GroundingDINO/MobileSAM, no DAM/VLM in core path",
    ]
    y = 26
    for i, line in enumerate(lines):
        color = (235, 235, 235) if i == 0 else (190, 220, 210)
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 if i else 0.62, color, 1)
        y += 28 if i == 0 else 24
    recent = sorted(track_summary.items(), key=lambda kv: kv[1].get("last_seen_frame") or -1, reverse=True)[:5]
    y += 8
    cv2.putText(panel, "recent tracks", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 220, 160), 1)
    y += 22
    for tid, ts in recent:
        cv2.putText(panel, f"T{tid}: obs={ts.get('total_observations',0)} last={ts.get('last_seen_frame')}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)
        y += 20
    return panel


def compose_dashboard(rgb: np.ndarray, masks: List[np.ndarray], tracks: np.ndarray, track_summary: Dict[int, dict], frame_data, frame_idx: int, n_frames: int, dets, frame_tracks: List[dict], timing: dict, t_start: float, panel_w: int = 640, panel_h: int = 360) -> np.ndarray:
    canvas = np.zeros((panel_h * 2, panel_w * 2, 3), dtype=np.uint8)
    canvas[0:panel_h, 0:panel_w] = render_track_overlay(rgb, masks, tracks, panel_w, panel_h)
    canvas[0:panel_h, panel_w:panel_w * 2] = render_representatives_panel(track_summary, panel_w, panel_h)
    canvas[panel_h:panel_h * 2, 0:panel_w] = render_topdown_panel(track_summary, frame_data.transform, panel_w, panel_h)
    canvas[panel_h:panel_h * 2, panel_w:panel_w * 2] = render_stats_panel(frame_idx, n_frames, dets, tracks, frame_tracks, timing, track_summary, t_start, panel_w, panel_h)
    return canvas


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True, help="DAAAM ImageSequenceDataset root")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--depth-scale", type=float, default=1000.0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--without-reid", action="store_true")
    ap.add_argument("--min-obs-per-track", type=int, default=3)
    ap.add_argument("--representatives-per-track", type=int, default=3)
    ap.add_argument("--representative-crop-mode", choices=["context", "masked"], default="context", help="Crop saved for CLIP labels/debug; context keeps RGB around the mask, masked zeros background")
    ap.add_argument("--top-k-labels", type=int, default=5)
    ap.add_argument("--save-vis", action="store_true")
    ap.add_argument("--vis-interval", type=int, default=10)
    ap.add_argument("--max-mask-area-ratio", type=float, default=0.18)
    ap.add_argument("--max-edge-mask-area-ratio", type=float, default=0.06)
    ap.add_argument("--max-bbox-aspect-ratio", type=float, default=8.0)
    ap.add_argument("--min-bbox-fill-ratio", type=float, default=0.22)
    ap.add_argument("--max-map-position-std-m", type=float, default=1.5)
    ap.add_argument("--spatial-cluster-radius-m", type=float, default=0.75)
    ap.add_argument("--live-window", action="store_true", help="Show a live OpenCV dashboard on the local display")
    ap.add_argument("--record-dashboard", action="store_true", help="Record the live dashboard to dashboard/live.mp4")
    ap.add_argument("--dashboard-dir", default=None, help="Dashboard frame/video output dir; defaults to output-dir/dashboard")
    ap.add_argument("--dashboard-fps", type=float, default=5.0)
    ap.add_argument("--dashboard-save-frames", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    for sub in ["labels", "vis", "representatives"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    dashboard_dir = Path(args.dashboard_dir) if args.dashboard_dir else out / "dashboard"
    video_writer = None
    if args.record_dashboard or args.dashboard_save_frames:
        dashboard_dir.mkdir(parents=True, exist_ok=True)
    if args.record_dashboard:
        video_path = dashboard_dir / "live.mp4"
        video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.dashboard_fps, (1280, 720))
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open dashboard video writer: {video_path}")
        print(f"Recording dashboard video: {video_path}")
    if args.live_window:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, 1280, 720)

    logger = get_default_logger()
    cfg_path = args.config or str(DAAAM_ROOT / "config" / "pipeline_config.yaml")
    config = PipelineConfig.from_yaml(cfg_path)
    config.segmentation.device = args.device
    config.tracking.device = args.device
    config.tracking.with_reid = not args.without_reid
    config.segmentation.model_name = "fastsam/FastSAM-s.pt"
    config.segmentation.model_config_path = "fastsam/fastsam_config.yaml"
    config.segmentation.imgsz = None

    dataset = ImageSequenceDataset(Path(args.data_path), depth_scale=args.depth_scale, compute_velocities=False)
    intrinsics = extract_camera_intrinsics(dataset.get_camera_info())

    seg_service = SegmentationService(config.segmentation, logger)
    trk_service = TrackingService(config.tracking, logger)
    seg_service.warmup()

    track_summary: Dict[int, dict] = defaultdict(lambda: {
        "track_id": 0,
        "first_seen_frame": None,
        "last_seen_frame": None,
        "total_observations": 0,
        "positions_world": [],
        "positions_camera": [],
        "median_depths": [],
        "bboxes": [],
        "representatives": [],
    })
    all_frames = []
    object_observations = []
    n_frames = len(dataset) if args.max_frames is None else min(len(dataset), args.max_frames)
    t_start = time.time()

    for i in range(n_frames):
        frame = dataset[i]
        rgb = frame.rgb_image
        t0 = time.time()
        dets, masks = seg_service.segment(rgb)
        dets_raw_count = len(dets)
        mask_filter_stats = {"input": len(masks), "kept": len(masks), "dropped_large": 0, "dropped_edge": 0}
        dets, masks, mask_filter_stats = filter_object_like_masks(
            dets,
            masks,
            max_area_ratio=args.max_mask_area_ratio,
            max_edge_area_ratio=args.max_edge_mask_area_ratio,
            max_bbox_aspect_ratio=args.max_bbox_aspect_ratio,
            min_bbox_fill_ratio=args.min_bbox_fill_ratio,
        )
        t_seg = time.time() - t0
        t0 = time.time()
        if len(dets) == 0:
            tracks = np.empty((0, 8), dtype=np.float32)
        else:
            tracks = normalize_tracks_array(trk_service.update(dets, rgb))
        t_trk = time.time() - t0

        frame_tracks = []
        for row in tracks:
            rec = process_track_row(frame, row, masks, intrinsics, config.depth.depth_lb, config.depth.depth_ub)
            if rec is None:
                continue
            tid = rec["track_id"]
            ts = track_summary[tid]
            ts["track_id"] = tid
            if ts["first_seen_frame"] is None:
                ts["first_seen_frame"] = i
            ts["last_seen_frame"] = i
            ts["total_observations"] += 1
            ts["bboxes"].append(rec["bbox"])
            if rec["median_depth"] > 0:
                ts["median_depths"].append(rec["median_depth"])
            if rec["pos_3d_world"] is not None:
                ts["positions_world"].append(rec["pos_3d_world"])
            if rec["pos_3d_camera"] is not None:
                ts["positions_camera"].append(rec["pos_3d_camera"])

            crop_path = out / "representatives" / f"track_{tid}" / f"f{i:06d}_area{rec['mask_area_pixels']}.png"
            crop_info = save_masked_crop(rgb, masks[rec["mask_idx"]].astype(bool), crop_path, crop_mode=args.representative_crop_mode)
            if crop_info:
                rep = {**crop_info, "frame_id": i, "mask_area_pixels": rec["mask_area_pixels"], "depth_valid": rec["depth_valid"]}
                maybe_add_representative(ts["representatives"], rep, args.representatives_per_track)
                rec["representative_crop_path"] = crop_info["path"]
            if rec.get("pos_3d_world") is not None and rec.get("representative_crop_path"):
                object_observations.append({**rec, "frame_id": i})

            frame_tracks.append({k: v for k, v in rec.items() if k != "mask_idx"})

        frame_entry = {
            "frame_id": i,
            "timestamp": float(frame.timestamp),
            "n_detections_raw": dets_raw_count,
            "n_detections": len(dets),
            "n_tracks": len(tracks),
            "tracks": frame_tracks,
            "mask_filter": mask_filter_stats,
            "timing": {"segmentation_ms": round(t_seg * 1000, 1), "tracking_ms": round(t_trk * 1000, 1)},
        }
        all_frames.append(frame_entry)
        (out / "labels" / f"{i:06d}.json").write_text(json.dumps(frame_entry, indent=2), encoding="utf-8")
        if args.save_vis and i % max(1, args.vis_interval) == 0:
            save_frame_vis(rgb, masks, tracks, out / "vis" / f"{i:06d}.png")
        if args.live_window or args.record_dashboard or args.dashboard_save_frames:
            dashboard = compose_dashboard(
                rgb=rgb,
                masks=masks,
                tracks=tracks,
                track_summary=track_summary,
                frame_data=frame,
                frame_idx=i,
                n_frames=n_frames,
                dets=dets,
                frame_tracks=frame_tracks,
                timing=frame_entry["timing"],
                t_start=t_start,
            )
            if args.live_window:
                cv2.imshow(WINDOW_TITLE, dashboard)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("\nLive dashboard requested stop.")
                    break
            if video_writer is not None:
                video_writer.write(dashboard)
            if args.dashboard_save_frames:
                cv2.imwrite(str(dashboard_dir / f"dashboard_{i:06d}.jpg"), dashboard)

        fps = (i + 1) / max(1e-6, time.time() - t_start)
        n_pos = sum(1 for t in frame_tracks if t.get("pos_3d_world") is not None)
        print(f"\r{i+1}/{n_frames} dets={len(dets)}/{dets_raw_count} tracks={len(tracks)} pos3d={n_pos} seg={t_seg*1000:.0f}ms trk={t_trk*1000:.0f}ms {fps:.1f}fps", end="", flush=True)
    print()

    print("Loading CLIP object-node labeler...")
    labeler = CLIPZeroShotLabeler(device=args.device)
    object_nodes = build_spatial_object_nodes(
        object_observations,
        radius_m=args.spatial_cluster_radius_m,
        min_obs=args.min_obs_per_track,
        max_reps=args.representatives_per_track,
        labeler=labeler,
        top_k=args.top_k_labels,
    )
    tracks_out = {}
    for tid, ts in sorted(track_summary.items()):
        reps = ts["representatives"]
        crop_paths = [r["path"] for r in reps]
        if ts["total_observations"] >= args.min_obs_per_track:
            lbl = label_representatives(labeler, crop_paths, top_k=args.top_k_labels)
        else:
            lbl = {"label": "unknown", "label_confidence": 0.0, "label_topk": [], "clip_embedding": None, "labeled_crop_paths": []}
        position_std = np.std(ts["positions_world"], axis=0).astype(float).tolist() if len(ts["positions_world"]) > 1 else None
        std_xz = None
        if position_std is not None:
            std_xz = float((position_std[0] ** 2 + position_std[2] ** 2) ** 0.5)
        quality_reasons = []
        if ts["total_observations"] < args.min_obs_per_track:
            quality_reasons.append("too_few_observations")
        if len(ts["positions_world"]) < args.min_obs_per_track:
            quality_reasons.append("too_few_3d_positions")
        if std_xz is not None and std_xz > args.max_map_position_std_m:
            quality_reasons.append("unstable_3d_position")
        if std_xz is None and len(ts["positions_world"]) > 1:
            quality_reasons.append("missing_position_std")
        map_eligible = len(quality_reasons) == 0

        tracks_out[str(tid)] = {
            "track_id": tid,
            "semantic_id": tid,
            "first_seen_frame": ts["first_seen_frame"],
            "last_seen_frame": ts["last_seen_frame"],
            "total_observations": ts["total_observations"],
            "avg_depth": float(np.mean(ts["median_depths"])) if ts["median_depths"] else None,
            "avg_position_world": np.mean(ts["positions_world"], axis=0).astype(float).tolist() if ts["positions_world"] else None,
            "avg_position_camera": np.mean(ts["positions_camera"], axis=0).astype(float).tolist() if ts["positions_camera"] else None,
            "position_world_std_m": position_std,
            "position_world_std_xz_m": std_xz,
            "positions_world_count": len(ts["positions_world"]),
            "representatives": reps,
            "map_eligible": map_eligible,
            "quality_reasons": quality_reasons,
            **lbl,
        }

    summary = {
        "schema": "raanav_object_node_frontend.v1",
        "frontend": "DAAAM-style object-node frontend (FastSAM + BotSort + assignment + CLIP)",
        "dataset_path": str(args.data_path),
        "total_frames_processed": n_frames,
        "total_tracks": len(tracks_out),
        "labeled_tracks": sum(1 for t in tracks_out.values() if t.get("label_confidence", 0) > 0),
        "map_eligible_tracks": sum(1 for t in tracks_out.values() if t.get("map_eligible")),
        "total_object_nodes": len(object_nodes),
        "map_eligible_object_nodes": sum(1 for t in object_nodes.values() if t.get("map_eligible")),
        "has_world_poses": any(t.get("avg_position_world") is not None for t in tracks_out.values()),
        "config": {
            "segmentation": config.segmentation.model_name,
            "tracking": "BotSort",
            "with_reid": config.tracking.with_reid,
            "assignment": {
                "min_obs_per_track": args.min_obs_per_track,
                "representatives_per_track": args.representatives_per_track,
                "representative_crop_mode": args.representative_crop_mode,
            },
            "grounding": "CLIP object-node embedding/top-k label; no DAM/VLM in core path",
            "device": args.device,
            "depth_scale": args.depth_scale,
        },
        "camera_info": dataset.get_camera_info(),
        "tracks": tracks_out,
        "object_nodes": object_nodes,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "all_frames.json").write_text(json.dumps(all_frames, indent=2), encoding="utf-8")

    label_counts = Counter(t.get("label", "unknown") for t in tracks_out.values() if t.get("label_confidence", 0) > 0)
    print(f"Done. tracks={len(tracks_out)} labeled={summary['labeled_tracks']} output={out}")
    print("Top labels:")
    for label, count in label_counts.most_common(15):
        print(f"  {label}: {count}")
    if video_writer is not None:
        video_writer.release()
    if args.live_window:
        cv2.destroyWindow(WINDOW_TITLE)


if __name__ == "__main__":
    main()
