"""感知模块: 观测→语义地图物体、CLIP编码、帧内去重、房间分配 — 从 sim_nav_loop.py 提取."""
from __future__ import annotations

import copy
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from semantic_map import Floor, Room, Object
from semantic_map_Create.virtual_clock import VirtualClock

# 结构性物体过滤集合
STRUCTURAL_CATEGORIES = {
    "wall", "floor", "ceiling", "unknown", "misc", "void", "objects",
    "column", "beam", "railing", "stair", "stairs",
}

# 全局对象图片保存目录 (由调用方设置)
_OBJ_CROP_DIR: Optional[str] = None


def set_object_crop_dir(path: str):
    """设置检测物体裁剪图保存目录."""
    global _OBJ_CROP_DIR
    _OBJ_CROP_DIR = path
    os.makedirs(path, exist_ok=True)


def save_object_crop(
    rgb: np.ndarray,
    bbox_xyxy: List[float],
    obj_id: str,
    step: int,
    padding: int = 15,
) -> Optional[str]:
    """保存检测到的物体的裁剪图片, 返回保存路径. 无目录设置时返回 None."""
    if _OBJ_CROP_DIR is None:
        return None
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    x1p, y1p = max(0, x1 - padding), max(0, y1 - padding)
    x2p, y2p = min(W, x2 + padding), min(H, y2 + padding)
    crop = rgb[y1p:y2p, x1p:x2p]
    if crop.size == 0:
        return None
    safe_id = obj_id.replace("/", "_").replace(" ", "_")
    fname = f"{safe_id}_s{step:04d}.jpg"
    fpath = os.path.join(_OBJ_CROP_DIR, fname)
    try:
        cv2.imwrite(fpath, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        return fpath
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 观测 → 语义地图构建
# ---------------------------------------------------------------------------

def objects_from_observation(
    visible_objs: List[Dict[str, Any]],
    agent_pos: np.ndarray,
    room_id: str,
    clock: Optional[VirtualClock],
    existing_ids: Set[str],
    n_views: int = 1,
) -> Tuple[List[Object], Set[str]]:
    """将可见物体列表转换为 Object 实例.

    使用 GT 信息 (AABB/OBB) 从 semantic scene 获取物体属性.
    """
    from semantic_map_Create.scene_extract import STABILITY_PRIOR

    now_str = clock.now_iso() if clock else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    objects = []
    seen_labels: Dict[str, int] = {}
    total_pixels = 640 * 480 * n_views

    for v in visible_objs:
        obj_ref = v["object_ref"]
        label = v["label"]
        sem_id = v["semantic_id"]

        if label not in seen_labels:
            seen_labels[label] = 0
        seen_labels[label] += 1
        obj_id = f"{label}_{sem_id}_{room_id}"

        center = obj_ref.aabb.center()
        sizes = obj_ref.aabb.size()
        pos_3d = [float(center[0]), float(center[1]), float(center[2])]
        pos_2d = {"x": float(center[0]), "y": float(center[2])}

        half_x = float(sizes[0]) / 2.0
        half_z = float(sizes[2]) / 2.0
        cx, cz = float(center[0]), float(center[2])
        min_x, max_x = cx - half_x, cx + half_x
        min_z, max_z = cz - half_z, cz + half_z
        region = [
            {"x": round(min_x, 5), "y": round(min_z, 5)},
            {"x": round(max_x, 5), "y": round(min_z, 5)},
            {"x": round(max_x, 5), "y": round(max_z, 5)},
            {"x": round(min_x, 5), "y": round(max_z, 5)},
            {"x": round(min_x, 5), "y": round(min_z, 5)},
        ]

        stability = STABILITY_PRIOR.get(label, 0.7)
        pixel_ratio = v["pixel_count"] / total_pixels
        cfd = min(1.0, pixel_ratio * 10)

        obj = Object(
            obj_id=obj_id,
            label=label,
            region=region,
            stability=stability,
            clip_embedding=[],
            cfd=round(cfd, 4),
            room_id=room_id,
            R_objs={},
            imgs={},
            N=1,
            description={},
            last_update_time=now_str,
            cooccur_stats={},
            exist_prob=1.0,
            pos_3d=pos_3d,
            pos_2d=pos_2d,
        )
        objects.append(obj)
        existing_ids.add(obj_id)

    return objects, existing_ids


def objects_from_detection(
    detections: List[Dict[str, Any]],
    agent_pos: np.ndarray,
    room_id: str,
    clock: Optional[VirtualClock],
    existing_ids: Set[str],
    rgb: Optional[np.ndarray] = None,
    step: int = 0,
) -> Tuple[List[Object], Set[str]]:
    """将视觉检测结果转换为 Object 实例 (替代 GT 版 objects_from_observation).

    Args:
        rgb: 原始 RGB 帧, 用于保存物体裁剪图 (可选, 传入时自动保存)
        step: 当前步数, 用于图片命名
    """
    from semantic_map_Create.scene_extract import STABILITY_PRIOR

    now_str = clock.now_iso() if clock else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    objects = []
    seen_counter: Dict[str, int] = {}

    for det in detections:
        label = det["label"]
        if label in STRUCTURAL_CATEGORIES or not label:
            continue

        pos_3d = det.get("pos_3d")
        if pos_3d is None:
            continue

        if label not in seen_counter:
            seen_counter[label] = 0
        seen_counter[label] += 1
        obj_id = f"{label}_det{seen_counter[label]}_{room_id}"

        pos_2d = det.get("pos_2d", {"x": pos_3d[0], "y": pos_3d[2]})

        depth_m = det.get("depth_median", 2.0)
        bbox = det.get("bbox_xyxy", [0, 0, 100, 100])
        bbox_w_px = bbox[2] - bbox[0]
        bbox_h_px = bbox[3] - bbox[1]
        fov_w = 2.0 * depth_m * math.tan(math.radians(45))
        obj_w = max(0.1, (bbox_w_px / 640.0) * fov_w)
        obj_h = max(0.1, (bbox_h_px / 480.0) * fov_w * 480 / 640)
        half_w = obj_w / 2.0
        half_h = obj_h / 2.0
        cx, cz = pos_2d["x"], pos_2d["y"]
        region = [
            {"x": round(cx - half_w, 5), "y": round(cz - half_h, 5)},
            {"x": round(cx + half_w, 5), "y": round(cz - half_h, 5)},
            {"x": round(cx + half_w, 5), "y": round(cz + half_h, 5)},
            {"x": round(cx - half_w, 5), "y": round(cz + half_h, 5)},
            {"x": round(cx - half_w, 5), "y": round(cz - half_h, 5)},
        ]

        stability = STABILITY_PRIOR.get(label, 0.5)
        cfd = round(float(det.get("confidence", 0.5)), 4)

        obj = Object(
            obj_id=obj_id,
            label=label,
            region=region,
            stability=stability,
            clip_embedding=det.get("clip_embedding", []),
            cfd=cfd,
            room_id=room_id,
            R_objs={},
            imgs={},
            N=1,
            description={},
            last_update_time=now_str,
            cooccur_stats={},
            exist_prob=1.0,
            pos_3d=pos_3d,
            pos_2d=pos_2d,
        )

        # 保存物体裁剪图 (用于 CLIP 图像匹配)
        # 优先使用检测时附带的 _rgb (来自对应视角), 其次使用全局 rgb
        det_rgb = det.get("_rgb", rgb)
        if det_rgb is not None and det.get("bbox_xyxy"):
            crop_path = save_object_crop(det_rgb, det["bbox_xyxy"], obj_id, step)
            if crop_path:
                obj.imgs[str(obj.N)] = crop_path

        objects.append(obj)
        existing_ids.add(obj_id)

    return objects, existing_ids


# ---------------------------------------------------------------------------
# CLIP 视觉编码
# ---------------------------------------------------------------------------

def _crop_and_mask_for_clip(
    rgb: np.ndarray,
    bbox_xyxy: List[float],
    mask: Optional[np.ndarray],
    padding: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """从 RGB 图裁剪检测区域, 返回 (local_crop, masked_crop)."""
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    x1p = max(0, x1 - padding)
    y1p = max(0, y1 - padding)
    x2p = min(W, x2 + padding)
    y2p = min(H, y2 + padding)

    local_crop = rgb[y1p:y2p, x1p:x2p].copy()

    if mask is not None:
        masked_rgb = rgb.copy()
        masked_rgb[~mask] = 0
        masked_crop = masked_rgb[y1p:y2p, x1p:x2p].copy()
    else:
        masked_crop = local_crop.copy()

    return local_crop, masked_crop


def compute_clip_embeddings_for_detections(
    rgb: np.ndarray,
    detections: List[Dict[str, Any]],
    clip_model: Any,
    clip_processor: Any,
    device: str,
    mode: str = "mask_only",
    masked_weight: float = 0.75,
    bbox_padding: int = 20,
) -> None:
    """为每个检测结果计算 CLIP 视觉 embedding, 就地写入 det["clip_embedding"]."""
    import torch as _torch
    from PIL import Image

    if not detections:
        return

    images_to_encode = []
    det_indices = []
    image_roles = []

    for i, det in enumerate(detections):
        bbox = det.get("bbox_xyxy")
        mask = det.get("mask")
        if bbox is None:
            det["clip_embedding"] = []
            continue

        local_crop, masked_crop = _crop_and_mask_for_clip(
            rgb, bbox, mask, padding=bbox_padding,
        )

        if mode == "mask_only":
            images_to_encode.append(Image.fromarray(masked_crop))
            det_indices.append(i)
            image_roles.append("masked")
        else:
            images_to_encode.append(Image.fromarray(masked_crop))
            det_indices.append(i)
            image_roles.append("masked")
            images_to_encode.append(Image.fromarray(local_crop))
            det_indices.append(i)
            image_roles.append("local")

    if not images_to_encode:
        return

    BATCH = 32
    all_feats = []
    for s in range(0, len(images_to_encode), BATCH):
        batch_imgs = images_to_encode[s : s + BATCH]
        inputs = clip_processor(images=batch_imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with _torch.no_grad():
            feats = clip_model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        all_feats.append(feats.cpu())
    all_feats = _torch.cat(all_feats, dim=0)

    if mode == "mask_only":
        for idx_in_batch, det_i in enumerate(det_indices):
            detections[det_i]["clip_embedding"] = all_feats[idx_in_batch].numpy().tolist()
    else:
        accum: Dict[int, Dict[str, Any]] = {}
        for idx_in_batch, (det_i, role) in enumerate(zip(det_indices, image_roles)):
            if det_i not in accum:
                accum[det_i] = {}
            accum[det_i][role] = all_feats[idx_in_batch]
        for det_i, parts in accum.items():
            m_feat = parts.get("masked")
            l_feat = parts.get("local")
            if m_feat is not None and l_feat is not None:
                fused = masked_weight * m_feat + (1.0 - masked_weight) * l_feat
                fused = fused / fused.norm().clamp(min=1e-8)
                detections[det_i]["clip_embedding"] = fused.numpy().tolist()
            elif m_feat is not None:
                detections[det_i]["clip_embedding"] = m_feat.numpy().tolist()
            else:
                detections[det_i]["clip_embedding"] = []

    for det in detections:
        if "clip_embedding" not in det:
            det["clip_embedding"] = []


# ---------------------------------------------------------------------------
# 帧内去重
# ---------------------------------------------------------------------------

def dedup_intra_frame(objects: List[Object], dist_threshold: float = 0.5) -> List[Object]:
    """帧内去重: 合并同 label + pos_3d 距离 < threshold 的物体."""
    if len(objects) <= 1:
        return objects

    by_label: Dict[str, List[Object]] = {}
    for obj in objects:
        by_label.setdefault(obj.label, []).append(obj)

    result = []
    for label, group in by_label.items():
        if len(group) == 1:
            result.append(group[0])
            continue

        group.sort(key=lambda o: getattr(o, 'cfd', 0.5), reverse=True)

        clusters: List[List[Object]] = []
        for obj in group:
            p = getattr(obj, 'pos_3d', None)
            if p is None or len(p) < 3:
                clusters.append([obj])
                continue

            merged = False
            for cluster in clusters:
                center = getattr(cluster[0], 'pos_3d', None)
                if center and len(center) >= 3:
                    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(p[:3], center[:3])))
                    if dist < dist_threshold:
                        cluster.append(obj)
                        merged = True
                        break
            if not merged:
                clusters.append([obj])

        for cluster in clusters:
            if len(cluster) == 1:
                result.append(cluster[0])
            else:
                base = copy.deepcopy(cluster[0])
                all_pos = [getattr(o, 'pos_3d', [0, 0, 0]) for o in cluster if getattr(o, 'pos_3d', None)]
                if all_pos:
                    base.pos_3d = [
                        round(sum(p[i] for p in all_pos) / len(all_pos), 6)
                        for i in range(3)
                    ]
                    base.pos_2d = {"x": base.pos_3d[0], "y": base.pos_3d[2]}
                base.cfd = max(getattr(o, 'cfd', 0) for o in cluster)
                result.append(base)

    return result


# ---------------------------------------------------------------------------
# 房间分配
# ---------------------------------------------------------------------------

def assign_room_ids(
    objects: List[Object],
    floors_history: List[Floor],
    fallback_room_id: str = "R0",
    max_dist: float = 5.0,
) -> None:
    """根据 pos_3d 最近邻将检测物体分配到历史房间 (就地修改 room_id)."""
    hist_pts: List[Tuple[np.ndarray, str]] = []
    for fl in floors_history:
        for rm in fl.rooms:
            for ho in rm.objects:
                p = getattr(ho, "pos_3d", None)
                if p is not None and len(p) >= 3:
                    hist_pts.append((np.array(p[:3], dtype=np.float32), rm.room_id))
    if not hist_pts:
        return

    hist_coords = np.stack([pt[0] for pt in hist_pts])
    hist_rids = [pt[1] for pt in hist_pts]

    for obj in objects:
        p = getattr(obj, "pos_3d", None)
        if p is None or len(p) < 3:
            obj.room_id = fallback_room_id
            continue
        q = np.array(p[:3], dtype=np.float32)
        dists = np.linalg.norm(hist_coords - q, axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] <= max_dist:
            obj.room_id = hist_rids[idx]
        else:
            obj.room_id = fallback_room_id


# ---------------------------------------------------------------------------
# 构建 floors_now
# ---------------------------------------------------------------------------

def build_floors_now(
    objects: List[Object],
    room_id: str = "R0",
    floor_id: str = "F0",
) -> List[Floor]:
    """从当前观测构建 floors_now (支持多房间: 按 obj.room_id 分组)."""
    groups: Dict[str, List[Object]] = defaultdict(list)
    for obj in objects:
        rid = getattr(obj, "room_id", None) or room_id
        groups[rid].append(obj)
    rooms = [
        Room(room_id=rid, room_name=None, objects=objs, region=None, floor_id=floor_id)
        for rid, objs in groups.items()
    ]
    floor = Floor(floor_id=floor_id, rooms=rooms)
    return [floor]


# ---------------------------------------------------------------------------
# 楼层检测 (参考 HOV-SG)
# ---------------------------------------------------------------------------

def detect_floors(
    objects: List[Object],
    bin_size_m: float = 0.3,
    min_objects_per_floor: int = 5,
    smooth_sigma: float = 2.0,
) -> List[Dict[str, Any]]:
    """从物体 Y 坐标检测楼层边界.

    算法 (参考 HOV-SG 高度直方图法):
      1. 收集所有物体 Y 坐标
      2. 构建高度直方图 (bin_size_m) 并高斯平滑
      3. 找平滑直方图的最深谷底 → 楼层分割点
      4. 若谷底密度 < 两侧峰值均值的 50% → 分割为两层

    Returns:
        按 y_min 从低到高排列的楼层信息列表:
        [{"floor_id": "F0", "y_min": ..., "y_max": ..., "obj_indices": [...]}, ...]
    """
    from scipy.ndimage import gaussian_filter1d

    # 收集有效 Y 坐标
    y_with_idx: List[Tuple[float, int]] = []
    for i, obj in enumerate(objects):
        if obj.pos_3d and len(obj.pos_3d) >= 2:
            y_with_idx.append((obj.pos_3d[1], i))

    if len(y_with_idx) < 2 * min_objects_per_floor:
        return [{"floor_id": "F0",
                 "y_min": 0.0, "y_max": 4.0,
                 "obj_indices": list(range(len(objects)))}]

    y_vals_all = np.array([v[0] for v in y_with_idx])
    y_min_all, y_max_all = float(y_vals_all.min()), float(y_vals_all.max())
    y_span = y_max_all - y_min_all

    # 跨度太小 (<2m) 不可能是多楼层
    if y_span < 2.0:
        return [{"floor_id": "F0",
                 "y_min": y_min_all, "y_max": y_max_all,
                 "obj_indices": [v[1] for v in y_with_idx]}]

    # 构建高度直方图
    n_bins = max(10, int(y_span / bin_size_m))
    hist, edges = np.histogram(y_vals_all, bins=n_bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0

    # 高斯平滑 (参考 HOV-SG)
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=smooth_sigma)

    # 排除两端 20% → 在中间 60% 范围找最深谷底
    margin = max(2, int(n_bins * 0.2))
    search_range = hist_smooth[margin:-margin]
    if len(search_range) < 3:
        return [{"floor_id": "F0",
                 "y_min": y_min_all, "y_max": y_max_all,
                 "obj_indices": [v[1] for v in y_with_idx]}]

    valley_idx_local = int(np.argmin(search_range))
    valley_idx = valley_idx_local + margin
    valley_y = float(bin_centers[valley_idx])
    valley_val = hist_smooth[valley_idx]

    # 两侧峰值
    left_peak = float(np.max(hist_smooth[:valley_idx]))
    right_peak = float(np.max(hist_smooth[valley_idx + 1:]))
    peak_mean = (left_peak + right_peak) / 2.0

    # 谷底密度需显著低于两侧峰值 (< 50%)
    if peak_mean < 1.0 or valley_val > peak_mean * 0.5:
        return [{"floor_id": "F0",
                 "y_min": y_min_all, "y_max": y_max_all,
                 "obj_indices": [v[1] for v in y_with_idx]}]

    # 分割
    lower_indices = [v[1] for v in y_with_idx if v[0] < valley_y]
    upper_indices = [v[1] for v in y_with_idx if v[0] >= valley_y]

    if len(lower_indices) < min_objects_per_floor or len(upper_indices) < min_objects_per_floor:
        return [{"floor_id": "F0",
                 "y_min": y_min_all, "y_max": y_max_all,
                 "obj_indices": [v[1] for v in y_with_idx]}]

    lower_ys = [v[0] for v in y_with_idx if v[0] < valley_y]
    upper_ys = [v[0] for v in y_with_idx if v[0] >= valley_y]

    return [
        {"floor_id": "F0",
         "y_min": min(lower_ys),
         "y_max": valley_y,
         "obj_indices": lower_indices},
        {"floor_id": "F1",
         "y_min": valley_y,
         "y_max": max(upper_ys),
         "obj_indices": upper_indices},
    ]
