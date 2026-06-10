"""HOV-SG 风格房间分割 — 基于占据栅格的距离变换 + 分水岭算法.

Pipeline (参考 HOV-SG graph.py + graph_utils.py):
  1. 占据栅格 → 二值墙骨架 (walls_skeleton)
  2. 距离变换 (Distance Transform)
  3. Otsu 自动阈值 → 种子区域
  4. 分水岭 (Watershed) 分割
  5. 过滤: 最小面积阈值
  6. 输出: 房间标签图 + 房间多边形 / 中心坐标

适配 AgenticRAG 的 OccupancyGrid (2D, resolution=0.05m).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def segment_rooms_from_occ_grid(
    occ_grid,
    min_room_area_m2: float = 1.0,
    morph_kernel_size: int = 3,
    morph_iterations: int = 1,
    wall_threshold_ratio: float = 0.25,
    blur_ksize: Tuple[int, int] = (5, 5),
    blur_sigma: float = 1.0,
    dist_blur_ksize: Tuple[int, int] = (11, 1),
    dist_blur_sigma: float = 10.0,
) -> Dict[str, Any]:
    """从 OccupancyGrid 执行房间分割.

    Args:
        occ_grid: OccupancyGrid 实例 (grid: 0=unknown, 1=free, 2=occupied)
        min_room_area_m2: 最小房间面积 (平方米)
        morph_kernel_size: 形态学闭运算核大小
        morph_iterations: 形态学闭运算迭代次数

    Returns:
        dict: {
            "room_labels": np.ndarray (H, W) — 房间标签 (0=背景/墙壁, 1..N=房间ID)
            "n_rooms": int,
            "room_centers": list of (world_x, world_z),
            "room_areas_m2": list of float,
            "room_pixel_counts": list of int,
            "walls_skeleton": np.ndarray (H, W, uint8) — 墙壁骨架图
        }
    """
    grid = occ_grid.grid.copy()
    resolution = occ_grid.resolution
    rows, cols = grid.shape

    # ===== Step 1: 构建墙壁骨架 =====
    # occupied (2) → 白色 (障碍/墙), free (1) → 黑色, unknown (0) → 按障碍处理
    walls_binary = np.zeros((rows, cols), dtype=np.uint8)
    walls_binary[grid != 1] = 255  # 非 free 区域都当作墙

    # 高斯模糊 + 形态学闭运算 (填补小缝隙)
    walls_blurred = cv2.GaussianBlur(walls_binary, blur_ksize, blur_sigma)
    _, walls_skeleton = cv2.threshold(
        walls_blurred, int(wall_threshold_ratio * 255), 255, cv2.THRESH_BINARY
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (morph_kernel_size, morph_kernel_size))
    walls_skeleton = cv2.morphologyEx(
        walls_skeleton, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations
    )

    # ===== Step 2: 距离变换 =====
    # 反转: 自由空间 → 255, 墙壁 → 0
    free_space = cv2.bitwise_not(walls_skeleton)
    free_space = np.uint8(free_space)

    dist = cv2.distanceTransform(free_space, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist_normalized = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 高斯模糊距离图 (找房间中心线)
    dist_blurred = cv2.GaussianBlur(dist_normalized, dist_blur_ksize, dist_blur_sigma)

    # ===== Step 3: Otsu 自动阈值 =====
    _, dist_thresh = cv2.threshold(
        dist_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ===== Step 4: 寻找种子区域 (connected components) =====
    dist_8u = dist_thresh.astype(np.uint8)
    contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 最小面积过滤 (像素)
    min_area_px = (min_room_area_m2 / (resolution ** 2))
    contours = [c for c in contours if cv2.contourArea(c) > min_area_px]

    if len(contours) == 0:
        # 没有找到有效房间种子 → 整个 free 区域作为一个房间
        room_labels = np.zeros((rows, cols), dtype=np.int32)
        room_labels[grid == 1] = 1
        center_ys, center_xs = np.where(grid == 1)
        if len(center_ys) > 0:
            cx = np.mean(center_xs)
            cy = np.mean(center_ys)
            world_center = occ_grid.grid_to_world(np.array([cx, cy]))
            centers = [(float(world_center[0]), float(world_center[1]))]
        else:
            centers = [(0.0, 0.0)]
        free_count = int(np.sum(grid == 1))
        area_m2 = free_count * resolution * resolution
        return {
            "room_labels": room_labels,
            "n_rooms": 1,
            "room_centers": centers,
            "room_areas_m2": [area_m2],
            "room_pixel_counts": [free_count],
            "walls_skeleton": walls_skeleton,
        }

    # ===== Step 5: 分水岭分割 =====
    markers = np.zeros((rows, cols), dtype=np.int32)
    for i, contour in enumerate(contours):
        cv2.drawContours(markers, contours, i, (i + 1), -1)  # 种子标签

    # 背景标记 (确保分水岭不会全标为 -1)
    # 在角落放一个背景种子
    bg_label = len(contours) + 1
    cv2.circle(markers, (2, 2), 1, bg_label, -1)

    # 分水岭需要 3-channel 输入
    walls_bgr = cv2.cvtColor(walls_skeleton, cv2.COLOR_GRAY2BGR)
    cv2.watershed(walls_bgr, markers)
    # markers: -1=边界, 0=未定, bg_label=背景, 1..N=房间

    # ===== Step 6: 提取房间信息 =====
    room_labels = np.zeros((rows, cols), dtype=np.int32)
    room_centers = []
    room_areas = []
    room_pixel_counts = []
    valid_room_id = 0

    for i in range(len(contours)):
        label_val = i + 1
        mask = (markers == label_val)
        px_count = int(np.sum(mask))
        area_m2 = px_count * resolution * resolution
        if area_m2 < min_room_area_m2:
            continue
        valid_room_id += 1
        room_labels[mask] = valid_room_id

        ys, xs = np.where(mask)
        cx = np.mean(xs)
        cy = np.mean(ys)
        world_center = occ_grid.grid_to_world(np.array([cx, cy]))
        room_centers.append((float(world_center[0]), float(world_center[1])))
        room_areas.append(area_m2)
        room_pixel_counts.append(px_count)

    n_rooms = valid_room_id

    return {
        "room_labels": room_labels,
        "n_rooms": n_rooms,
        "room_centers": room_centers,
        "room_areas_m2": room_areas,
        "room_pixel_counts": room_pixel_counts,
        "walls_skeleton": walls_skeleton,
    }


def assign_objects_to_rooms(
    objects,
    room_labels: np.ndarray,
    occ_grid,
) -> None:
    """根据 room_labels 为物体分配 room_id (就地修改).

    对每个物体的 pos_2d 查找 room_labels 中的房间标签.
    """
    for obj in objects:
        pos_2d = getattr(obj, "pos_2d", None)
        if pos_2d is None:
            continue
        if isinstance(pos_2d, dict):
            wx, wz = pos_2d.get("x", 0), pos_2d.get("y", 0)
        else:
            wx, wz = float(pos_2d[0]), float(pos_2d[1])

        gc = occ_grid.world_to_grid(np.array([wx, wz]))
        c, r = int(gc[0]), int(gc[1])
        rows, cols = room_labels.shape
        if 0 <= r < rows and 0 <= c < cols:
            label = room_labels[r, c]
            if label > 0:
                obj.room_id = f"R{label}"


def visualize_room_segmentation(
    room_labels: np.ndarray,
    walls_skeleton: np.ndarray,
    room_centers: List[Tuple[float, float]],
    occ_grid,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """可视化房间分割结果.

    Returns:
        BGR image
    """
    rows, cols = room_labels.shape
    n_rooms = int(room_labels.max())

    # 为每个房间生成不同颜色
    np.random.seed(42)
    colors = []
    for _ in range(n_rooms + 1):
        colors.append(tuple(int(c) for c in np.random.randint(60, 230, size=3)))

    vis = np.zeros((rows, cols, 3), dtype=np.uint8)

    # 画墙壁 (灰色)
    vis[walls_skeleton > 0] = (80, 80, 80)

    # 为每个房间上色
    for room_id in range(1, n_rooms + 1):
        mask = room_labels == room_id
        vis[mask] = colors[room_id]

    # 画房间中心
    for i, (wx, wz) in enumerate(room_centers):
        gc = occ_grid.world_to_grid(np.array([wx, wz]))
        cx, cy = int(gc[0]), int(gc[1])
        cv2.circle(vis, (cx, cy), 5, (255, 255, 255), -1)
        cv2.putText(vis, f"R{i+1}", (cx + 7, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if save_path:
        cv2.imwrite(save_path, vis)

    return vis


# ------------------------------------------------------------------
# CLIP 房间语义标签 (HOV-SG 风格)
# ------------------------------------------------------------------

# 候选房间类型 (英文, 覆盖常见室内场景)
ROOM_TYPE_CANDIDATES = [
    "kitchen",
    "living room",
    "bedroom",
    "bathroom",
    "dining room",
    "hallway",
    "office",
    "laundry room",
    "closet",
    "garage",
    "storage room",
    "entrance",
    "balcony",
    "staircase",
]


def label_rooms_with_clip(
    rooms: List[dict],
    objects_by_room: dict,
    clip_model,
    clip_processor,
    device: str = "cuda",
    top_k: int = 3,
) -> List[dict]:
    """为每个房间基于房间内物体的 CLIP 特征分配语义标签.

    方法 (参考 HOV-SG room semantic labeling):
      1. 对每个房间, 收集其内所有物体的 CLIP 视觉 embedding
      2. 计算这些 embedding 的均值作为房间"视觉语义向量"
      3. 将候选房间类型转为 CLIP 文本 embedding
      4. 余弦相似度匹配, 返回最高匹配的房间类型

    Args:
        rooms: segment_rooms_from_grid 的输出 list
        objects_by_room: {room_id: [Object, ...]} — 每个房间的物体列表
        clip_model: HuggingFace CLIPModel
        clip_processor: HuggingFace CLIPProcessor
        device: torch device
        top_k: 返回 top-k 候选类型

    Returns:
        rooms 列表 (就地修改), 每个 room dict 增加:
          "room_label": str — 最佳语义标签
          "room_label_scores": list of (label, score) — top-k 候选
    """
    import torch as _torch

    # --- 预计算所有候选房间类型的文本 embedding ---
    prompts = [f"a photo of a {rt}" for rt in ROOM_TYPE_CANDIDATES]
    text_inputs = clip_processor(text=prompts, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with _torch.no_grad():
        text_feats = clip_model.get_text_features(**text_inputs)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    # text_feats: (N_types, D)

    for room in rooms:
        rid = room["room_id"]
        objs = objects_by_room.get(rid, [])

        # 收集该房间所有物体的 CLIP embedding
        embeddings = []
        for obj in objs:
            emb = getattr(obj, "clip_embedding", None)
            if emb and len(emb) > 0:
                embeddings.append(emb)

        if not embeddings:
            # 没有 CLIP 特征 → 回退到基于物体标签的文本匹配
            labels = [getattr(obj, "label", "") for obj in objs if getattr(obj, "label", "")]
            if labels:
                room_text = f"a room containing {', '.join(labels[:20])}"
                rt_inputs = clip_processor(text=[room_text], return_tensors="pt", padding=True)
                rt_inputs = {k: v.to(device) for k, v in rt_inputs.items()}
                with _torch.no_grad():
                    room_feat = clip_model.get_text_features(**rt_inputs)
                    room_feat = room_feat / room_feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            else:
                room["room_label"] = "unknown"
                room["room_label_scores"] = []
                continue
        else:
            # 均值池化所有物体的视觉 embedding → 房间语义向量
            emb_tensor = _torch.tensor(embeddings, dtype=_torch.float32).to(device)
            room_feat = emb_tensor.mean(dim=0, keepdim=True)
            room_feat = room_feat / room_feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # 余弦相似度匹配
        sims = (room_feat @ text_feats.T).squeeze(0)  # (N_types,)
        top_indices = sims.argsort(descending=True)[:top_k]
        scores = [(ROOM_TYPE_CANDIDATES[i], float(sims[i])) for i in top_indices]
        room["room_label"] = scores[0][0]
        room["room_label_scores"] = scores

    return rooms
