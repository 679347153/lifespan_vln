"""深度探索模式 (Stage 1 / Deep Exploration)

在该模式下, 机器人对环境进行**完整覆盖式探索**, 没有目标物体:
  - 核心目的: 探索房间、建图, 不是找东西
  - 与任务模式(sim_nav_loop.py)的区别:
      深度探索: 无目标, frontier驱动全覆盖, 建立完整环境模型
      任务模式: 有目标(如"chair"), GMM引导, 找到即停
  - 产出:
    1. 完整的占据栅格地图 (从深度传感器构建)
    2. 房间分割 (HOV-SG分水岭算法)
    3. Voronoi导航图
    4. 完整物体图谱 (含CLIP embedding)

两种后端:
  a) --use-navmesh (推荐/回退方案):
     用navmesh做移动 (等同于真实部署时手动驾驶), 自有传感器做感知
     frontier驱动探索: 深度→栅格→frontier检测→navmesh寻路→走过去
  b) --no-navmesh (纯自主):
     纯深度构建栅格 + A*导航, 更接近完全自主但难度更大

使用方式:
  cd /home/adminer/agentRAG/AgenticRAG
  conda run -n agentrag python scripts/deep_exploration.py \\
      --scene-dir /path/to/00814-p53SfW6mjZe \\
      --output-dir deep_exploration_output \\
      --max-steps 200 --step-size 1.0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from semantic_map import Floor, Room, Object
from semantic_map_Create.scene_extract import make_simulator
from semantic_map_Create.occupancy_grid import OccupancyGrid, CameraIntrinsics
from semantic_map_Create.virtual_clock import VirtualClock, set_global_clock

DEFAULT_DATASET_CONFIG = "/home/adminer/agentRAG/experiment_data/hm3d/hm3d_val_scene_dataset_config.json"

STRUCTURAL_CATEGORIES = {
    "wall", "floor", "ceiling", "door", "window", "door frame",
    "beam", "pillar", "stairs", "stairs railing", "stair handle",
    "tiled floor", "carpet", "step", "grate", "ceiling duct",
    "shower wall", "bath wall", "wall panel",
}

INDOOR_OBJECTS = None  # 延迟加载, 由 run_deep_exploration() 从配置初始化


# ---------------------------------------------------------------------------
# 房间分割: 基于占据栅格连通分量
# ---------------------------------------------------------------------------

def segment_rooms_from_grid(
    occ_grid: OccupancyGrid,
    min_room_area_m2: float = 0.5,
    save_debug_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """从占据栅格分割房间 — 基于 HOV-SG 的高度切片密度图+分水岭算法.

    核心改进 (对齐 HOV-SG segment_rooms):
      HOV-SG 对 3D 点云做 1.5m+ 高度切片, 只保留墙壁结构, 排除低矮家具.
      我们使用 occ_grid.wall_density (在 update_from_habitat_depth 中仅累积
      ≥1.2m 高度的深度点), 效果等价于 HOV-SG 的 2D 密度直方图.

    算法:
      1. wall_density → 归一化 → 高斯模糊 → 阈值 → 墙壁骨架 (walls_skeleton)
      2. explored 区域 → 外边界 (outside_boundary)
      3. full_map = walls_skeleton | bitwise_not(outside_boundary)
      4. 距离变换 → Otsu 阈值 → 房间种子
      5. 分水岭 → 房间区域

    Args:
        occ_grid: 已构建好的占据栅格 (含 wall_density)
        min_room_area_m2: 最小房间面积 (平方米)
        save_debug_dir: 若提供, 保存中间调试图像

    Returns:
        list of room dicts:
          {"room_id": "R0", "polygon": [...], "center": [x, z],
           "area_m2": float, "grid_mask": np.ndarray}
    """
    # --- Step 0: 从 wall_density 构建墙壁骨架 (类似 HOV-SG 的 2D histogram) ---
    wd = occ_grid.wall_density.copy()
    if wd.max() > 0:
        # 归一化到 [0, 255], 与 HOV-SG histogram 处理一致
        walls = cv2.normalize(wd, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        walls = cv2.GaussianBlur(walls, (5, 5), 1)
        wall_threshold = 0.25 * float(walls.max()) if walls.max() > 0 else 0
        _, walls_skeleton = cv2.threshold(
            walls, int(wall_threshold), 255, cv2.THRESH_BINARY
        )
    else:
        # wall_density 全为零 (可能 navmesh 模式无深度更新), 回退到 grid==OCCUPIED
        walls_skeleton = (occ_grid.grid == 2).astype(np.uint8) * 255

    # 闭运算清理 (HOV-SG: MORPH_CROSS, (3,3), 1 iteration)
    k_wall = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    walls_skeleton = cv2.morphologyEx(walls_skeleton, cv2.MORPH_CLOSE, k_wall, iterations=1)

    # copyMakeBorder: 添加 10px 零值边框 (HOV-SG 标准做法)
    walls_skeleton = cv2.copyMakeBorder(walls_skeleton, 10, 10, 10, 10,
                                        cv2.BORDER_CONSTANT, value=0)

    # --- Step 1: 构建外边界 (HOV-SG: xyz_full → histogram2d → 大模糊) ---
    # 使用 full_density (所有深度投影点的累积密度, 等价于 HOV-SG 的 xyz_full)
    fd = occ_grid.full_density.copy()
    if fd.max() > 0:
        # 先将 full_density 二值化并大核膨胀, 桥接深度观测间的间隙
        # (HOV-SG 有完整3D重建点云, 我们只有深度相机轨迹观测, 需要更强连接)
        fd_binary = (fd > 0).astype(np.uint8) * 255
        k_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fd_connected = cv2.morphologyEx(fd_binary, cv2.MORPH_CLOSE, k_bridge, iterations=2)
        # 大核高斯模糊平滑边界 (HOV-SG: (21,21), sigma=2)
        fd_smooth = cv2.GaussianBlur(fd_connected, (21, 21), 2)
        _, outside_boundary = cv2.threshold(fd_smooth, 0, 255, cv2.THRESH_BINARY)
    else:
        # full_density 为空 (navmesh 模式无深度更新), 回退到 grid==FREE
        free_u8 = (occ_grid.grid == 1).astype(np.uint8) * 255
        free_blur = cv2.GaussianBlur(free_u8, (21, 21), 2)
        _, outside_boundary = cv2.threshold(free_blur, 0, 255, cv2.THRESH_BINARY)

    # copyMakeBorder: 添加 10px 零值边框 (与 walls_skeleton 对齐)
    outside_boundary = cv2.copyMakeBorder(outside_boundary, 10, 10, 10, 10,
                                          cv2.BORDER_CONSTANT, value=0)

    # 对外边界做闭运算填补小孔 (HOV-SG: kernel=(5,5), iterations=3)
    k_boundary = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    outside_boundary = cv2.morphologyEx(outside_boundary, cv2.MORPH_CLOSE, k_boundary, iterations=3)

    # 提取外轮廓并填充 (确保内部连通)
    contours_boundary, _ = cv2.findContours(outside_boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outside_boundary = np.zeros_like(outside_boundary)
    cv2.drawContours(outside_boundary, contours_boundary, -1, 255, -1)

    # --- Step 2: 合并墙壁骨架 + 外边界 → full_map ---
    # HOV-SG: full_map = bitwise_or(walls_skeleton, bitwise_not(outside_boundary))
    # 含义: 障碍 = 墙壁 OR 建筑外部
    full_map = cv2.bitwise_or(walls_skeleton, cv2.bitwise_not(outside_boundary))

    # 闭运算清理 (HOV-SG: kernel=(3,3), iterations=2)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    full_map = cv2.morphologyEx(full_map, cv2.MORPH_CLOSE, k_close, iterations=2)

    # --- Step 3: 距离变换 ---
    bw = cv2.bitwise_not(full_map)
    bw = bw.astype(np.uint8)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # 归一化到 [0, 255]
    cv2.normalize(dist, dist, 0, 255, cv2.NORM_MINMAX)
    dist = np.uint8(dist)

    # --- Step 4: 高斯模糊 + Otsu 阈值 (HOV-SG 参数) ---
    blur = cv2.GaussianBlur(dist, (11, 11), 10)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Step 5: 提取房间种子轮廓 ---
    thresh_8u = thresh.astype(np.uint8)
    contours_seeds, _ = cv2.findContours(thresh_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 面积过滤 (HOV-SG: min 0.5 m²)
    min_area_px = min_room_area_m2 / (occ_grid.resolution ** 2)
    contours_seeds = [c for c in contours_seeds if cv2.contourArea(c) > min_area_px]

    print(f"[房间分割] HOV-SG 分水岭, 检测到 {len(contours_seeds)} 个房间种子 "
          f"(过滤 < {min_room_area_m2} m²)")

    if len(contours_seeds) == 0:
        print(f"[房间分割] 未检测到房间种子, 将整个已观测区域视为 R0")
        observed = (occ_grid.full_density > 0).astype(np.uint8) if occ_grid.full_density.max() > 0 \
            else (occ_grid.grid == 1).astype(np.uint8)
        room_mask = observed
        poly = compute_room_polygon_from_mask(room_mask, occ_grid)
        ys, xs = np.where(room_mask > 0)
        center_world = occ_grid.grid_to_world(np.array([xs.mean(), ys.mean()]))
        area_m2 = np.sum(room_mask > 0) * (occ_grid.resolution ** 2)
        return [{
            "room_id": "R0", "polygon": poly,
            "center": [float(center_world[0]), float(center_world[1])],
            "area_m2": float(area_m2), "grid_mask": room_mask,
        }]

    # --- Step 6: 分水岭算法 ---
    markers = np.zeros(full_map.shape, dtype=np.int32)
    for i, c in enumerate(contours_seeds):
        cv2.drawContours(markers, [c], 0, (i + 1), -1)
    # 背景标记 (HOV-SG: 在角落放一个小圆)
    cv2.circle(markers, (3, 3), 1, len(contours_seeds) + 1, -1)

    full_map_bgr = cv2.cvtColor(full_map, cv2.COLOR_GRAY2BGR)
    cv2.watershed(full_map_bgr, markers)

    # 裁剪掉 copyMakeBorder 添加的 10px 边框, 恢复到原始栅格尺寸
    markers = markers[10:-10, 10:-10]

    # --- Step 7: 提取房间 ---
    rooms = []
    # 构建已观测区域掩码 (full_density > 0 表示被深度相机实际观测过)
    observed_mask = (occ_grid.full_density > 0).astype(np.uint8) if occ_grid.full_density.max() > 0 \
        else (occ_grid.grid == 1).astype(np.uint8)
    for i in range(len(contours_seeds)):
        label_id = i + 1
        room_mask = (markers == label_id).astype(np.uint8)
        # 限制在已观测区域内
        room_mask = room_mask & observed_mask

        area_m2 = np.sum(room_mask > 0) * (occ_grid.resolution ** 2)
        if area_m2 < min_room_area_m2:
            continue

        poly = compute_room_polygon_from_mask(room_mask, occ_grid)
        if poly is None:
            continue

        ys, xs = np.where(room_mask > 0)
        center_gc = np.array([xs.mean(), ys.mean()])
        center_world = occ_grid.grid_to_world(center_gc)

        rooms.append({
            "room_id": f"R{len(rooms)}",
            "polygon": poly,
            "center": [float(center_world[0]), float(center_world[1])],
            "area_m2": float(area_m2),
            "grid_mask": room_mask,
        })

    # --- Debug 输出 ---
    if save_debug_dir:
        os.makedirs(save_debug_dir, exist_ok=True)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # 带 padding 的中间图像 (walls_skeleton, outside_boundary, full_map, dist 等)
        # 裁剪回原始尺寸用于可视化
        _crop = lambda img: img[10:-10, 10:-10] if img.shape[0] > 20 else img
        for name, img in [("1_walls_skeleton", _crop(walls_skeleton)),
                          ("1b_outside_boundary", _crop(outside_boundary)),
                          ("2_full_map", _crop(full_map)),
                          ("3_distance", _crop(dist)), ("4_blur", _crop(blur)),
                          ("5_thresh", _crop(thresh_8u)),
                          ("6_markers", markers.astype(np.float32))]:
            plt.figure(figsize=(8, 8))
            cmap = "jet" if "dist" in name or "marker" in name else "gray"
            plt.imshow(img, cmap=cmap, origin="lower")
            plt.title(name)
            plt.savefig(os.path.join(save_debug_dir, f"{name}.png"), dpi=100)
            plt.close()

    print(f"[房间分割] 最终 {len(rooms)} 个房间:")
    for r in rooms:
        print(f"  {r['room_id']}: {r['area_m2']:.1f} m², "
              f"center=[{r['center'][0]:.1f}, {r['center'][1]:.1f}]")

    return rooms


def compute_room_polygon_from_mask(
    room_mask: np.ndarray,
    occ_grid: OccupancyGrid,
) -> Optional[List[Dict[str, float]]]:
    """从房间二值掩码计算 凸包多边形 (世界坐标).

    Args:
        room_mask: (H, W) uint8, 房间区域 = 1
        occ_grid: 用于坐标转换

    Returns:
        多边形顶点列表 [{"x": float, "y": float}, ...] (x-z 平面)
        或 None (点太少)
    """
    contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 合并所有轮廓点
    all_pts = np.vstack(contours).squeeze()
    if all_pts.ndim != 2 or len(all_pts) < 3:
        return None

    # 凸包
    hull = cv2.convexHull(all_pts)
    hull_pts = hull.squeeze()
    if hull_pts.ndim != 2:
        return None

    # 栅格坐标 → 世界坐标 (注意: contour 点是 (col, row) 格式)
    polygon = []
    for pt in hull_pts:
        world_xz = occ_grid.grid_to_world(np.array([float(pt[0]), float(pt[1])]))
        polygon.append({"x": float(world_xz[0]), "y": float(world_xz[1])})

    return polygon


# ---------------------------------------------------------------------------
# 全覆盖路径规划
# ---------------------------------------------------------------------------

def plan_coverage_waypoints(
    sim,
    spacing: float = 1.5,
    max_waypoints: int = 100,
    occ_grid=None,
    agent_y: float = 0.0,
) -> List[List[float]]:
    """规划全覆盖路径点.

    navmesh 模式 (occ_grid=None): 在 navmesh 上均匀采样.
    occ_grid 模式: 在已知 free 区域上均匀采样.

    策略:
      1. 获取边界 (navmesh bounds 或 occ_grid extent)
      2. 在边界内按 spacing 生成规则网格
      3. 过滤: 只保留可导航的点
      4. 贪心最近邻排序 (减少行走距离)
    """
    import habitat_sim

    candidates = []

    if occ_grid is not None:
        # occ_grid 模式: 从已知 free 区域采样
        free_ys, free_xs = np.where(occ_grid.grid == 1)
        if len(free_ys) == 0:
            print("[全覆盖规划] occ_grid 无 free cell, 跳过")
            return []
        free_world = occ_grid.grid_to_world(np.stack([free_xs, free_ys], axis=-1))
        xmin, zmin = free_world.min(axis=0)
        xmax, zmax = free_world.max(axis=0)

        x = float(xmin) + spacing / 2
        while x < float(xmax):
            z = float(zmin) + spacing / 2
            while z < float(zmax):
                if occ_grid.is_navigable_at(x, z):
                    candidates.append([x, agent_y, z])
                z += spacing
            x += spacing

        if not candidates:
            # fallback: 随机从 free cells 采样
            indices = np.random.choice(len(free_ys), size=min(max_waypoints, len(free_ys)), replace=False)
            for i in indices:
                w = occ_grid.grid_to_world(np.array([free_xs[i], free_ys[i]]))
                candidates.append([float(w[0]), agent_y, float(w[1])])
    else:
        # navmesh 模式
        pf = sim.pathfinder
        bounds = pf.get_bounds()
        bmin, bmax = bounds

        x = float(bmin[0]) + spacing / 2
        while x < float(bmax[0]):
            z = float(bmin[2]) + spacing / 2
            while z < float(bmax[2]):
                query = np.array([x, float(bmin[1]), z])
                snapped = pf.snap_point(query)
                if not np.isnan(snapped).any() and pf.is_navigable(snapped, max_y_delta=0.5):
                    candidates.append([float(snapped[0]), float(snapped[1]), float(snapped[2])])
                z += spacing
            x += spacing

        if not candidates:
            for _ in range(max_waypoints):
                pt = pf.get_random_navigable_point()
                if not np.isnan(pt).any():
                    candidates.append([float(pt[0]), float(pt[1]), float(pt[2])])

    # 去重: 距离 < spacing/2 的点合并
    unique = []
    for c in candidates:
        too_close = False
        for u in unique:
            d = math.sqrt((c[0]-u[0])**2 + (c[2]-u[2])**2)
            if d < spacing * 0.4:
                too_close = True
                break
        if not too_close:
            unique.append(c)

    # 截断
    if len(unique) > max_waypoints:
        unique = unique[:max_waypoints]

    # 贪心最近邻排序 (TSP 近似)
    if len(unique) <= 1:
        return unique

    ordered = [unique[0]]
    remaining = set(range(1, len(unique)))

    while remaining:
        cur = ordered[-1]
        best_idx = min(remaining,
                       key=lambda i: (unique[i][0]-cur[0])**2 + (unique[i][2]-cur[2])**2)
        ordered.append(unique[best_idx])
        remaining.remove(best_idx)

    print(f"[全覆盖规划] {len(ordered)} 个路径点 (间距 {spacing}m, 范围 {len(candidates)} → 去重 {len(unique)})")
    return ordered


# ---------------------------------------------------------------------------
# 完整物体图谱构建
# ---------------------------------------------------------------------------

def build_complete_object_map(
    scene_dir: str,
    dataset_config: str,
    clip_mode: str = "full",
    output_dir: str = "deep_exploration_output",
    max_steps: int = 200,
    step_size: float = 1.0,
    n_views: int = 4,
    config_path: str = "",
    use_navmesh: bool = True,
    coverage_target: float = 0.95,
    frontier_min_area_m2: float = 0.3,
    stale_threshold: int = 15,
    live_viz: bool = False,
) -> Dict[str, Any]:
    """深度探索: frontier驱动全覆盖建图, 无目标物体.

    核心循环 (frontier-based, 参考 BeliefMapNav):
      1. 360°环视 → 深度→栅格更新 + 物体检测
      2. 从栅格检测 frontier (已探索与未探索的边界)
      3. 选择最佳 frontier (距已访问位置最远的)
      4. navmesh/A* 规划路径 → 逐步走过去 (每步做感知)
      5. 重复, 直到无 frontier 或达到 max_steps
      6. 后处理: 房间分割 + Voronoi图 + 物体重分配

    Args:
        scene_dir: HM3D 场景目录
        dataset_config: scene_dataset_config.json
        clip_mode: CLIP编码模式
        output_dir: 输出目录
        max_steps: 最大行走步数
        step_size: 路径采样间距 (米)
        n_views: 每步环视视角数 (4=每90°)
        config_path: map.yaml 路径
        use_navmesh: True=navmesh移动+自有感知 (推荐)
        coverage_target: 覆盖率目标 (0-1, 达到则提前停)
        frontier_min_area_m2: 最小 frontier 区域面积
        stale_threshold: 连续N步无新 free cell → 判定探索停滞
    """
    import yaml
    from scripts.sim_nav_loop import (
        HabitatAgent, objects_from_detection, dedup_intra_frame,
        build_floors_now, compute_clip_embeddings_for_detections,
        set_object_crop_dir,
    )
    from scripts.nav_core.perception import detect_floors
    from semantic_map_Update.map_Update import run_merge
    from semantic_map_Create.astar_planner import GridAStarPlanner

    # CLIP 模型缓存 (本模块内复用)
    clip_cache: Dict[str, Any] = {}

    def ensure_clip():
        if clip_cache:
            return
        import torch as _torch
        from transformers import CLIPProcessor, CLIPModel
        _dev = "cuda" if _torch.cuda.is_available() else "cpu"
        _m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_dev).eval()
        _p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_cache["model"] = _m
        clip_cache["processor"] = _p
        clip_cache["device"] = _dev
        print(f"[CLIP] loaded on {_dev}")

    os.makedirs(output_dir, exist_ok=True)
    rgb_dir = os.path.join(output_dir, "rgb_frames")
    os.makedirs(rgb_dir, exist_ok=True)
    set_object_crop_dir(os.path.join(output_dir, "object_crops"))

    # 加载配置
    if not config_path:
        config_path = os.path.join(_PROJ_ROOT, "config", "map.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    deep_cfg = cfg.get("exploration_mode", {}).get("deep", {})
    clip_encode_mode = deep_cfg.get("clip_mode", clip_mode)
    clip_cfg = cfg.get("clip_visual", {})
    clip_masked_weight = float(clip_cfg.get("masked_weight", 0.75))
    clip_bbox_padding = int(clip_cfg.get("bbox_padding", 20))
    dedup_dist = float(clip_cfg.get("dedup_dist_threshold", 0.5))

    # 虚拟时钟
    clock = VirtualClock(step_hours=1.0)
    set_global_clock(clock)

    # 解析场景路径
    scene_name = os.path.basename(scene_dir)
    parts = scene_name.split("-", 1)
    scene_stem = parts[1] if len(parts) > 1 else scene_name
    basis_glb = os.path.join(scene_dir, f"{scene_stem}.basis.glb")

    nav_mode_str = "navmesh移动+自有感知" if use_navmesh else "纯深度+A*"
    print(f"\n{'='*60}")
    print(f"  深度探索模式 (Stage 1) — 全覆盖建图, 无目标")
    print(f"  场景: {scene_name}")
    print(f"  导航后端: {nav_mode_str}")
    print(f"  CLIP模式: {clip_encode_mode}")
    print(f"  最大步数: {max_steps}, 步长: {step_size}m")
    print(f"  环视视角: {n_views}×{360//n_views}°")
    print(f"  覆盖率目标: {coverage_target*100:.0f}%")
    print(f"  frontier最小面积: {frontier_min_area_m2}m²")
    print(f"{'='*60}\n")

    # 初始化仿真器
    sim = make_simulator(basis_glb, dataset_config,
                         enable_semantic=False, enable_depth=True)
    agent = HabitatAgent(sim, use_navmesh=use_navmesh)
    cam_intrinsics = CameraIntrinsics(hfov_deg=90.0, height=480, width=640)

    # 初始化占据栅格 + 起始位置
    start_pos = agent.get_random_navigable_point() if use_navmesh else agent.get_position()
    agent.set_position(start_pos)
    if use_navmesh:
        # navmesh 模式: 从 GT navmesh 构建全局占据栅格 (完整空间信息)
        occ_grid = OccupancyGrid.from_navmesh_fast(
            sim, resolution=0.05, agent_radius=0.18, num_samples=50000,
        )
    else:
        # 纯自主模式: 从 agent 位置初始化空白栅格, 深度帧逐步填充
        occ_grid = OccupancyGrid.from_agent_position(
            agent_pos=start_pos, resolution=0.05, agent_radius=0.18,
            initial_extent=15.0,
        )
    agent.set_occ_grid(occ_grid)
    
    # A* planner (仅 non-navmesh 模式需要)
    astar_planner = None
    if not use_navmesh:
        astar_planner = GridAStarPlanner(
            occ_grid, obstacle_weight=5.0, obstacle_decay_cells=6,
            use_navmesh_grid=False, unknown_cost=0.3,
        )

    print(f"[起始位置] {[round(x, 2) for x in start_pos]}")
    print(f"[占据栅格] {occ_grid.shape}, free={np.sum(occ_grid.grid==1)}, occ={np.sum(occ_grid.grid==2)}")

    # 加载检测器 + 检测标签集
    global INDOOR_OBJECTS
    from semantic_map_Create.perception import get_detector
    from scripts.sim_nav_loop import _load_detect_labels
    INDOOR_OBJECTS = _load_detect_labels(cfg)
    detector = get_detector(device="cuda")
    detect_text_prompt = " . ".join(INDOOR_OBJECTS)
    detect_labels_set = set(lbl.lower() for lbl in INDOOR_OBJECTS)
    print(f"[检测标签] {len(INDOOR_OBJECTS)} 类: {', '.join(INDOOR_OBJECTS[:10])}...")

    # 探索状态
    floors_history: List[Floor] = []
    all_obj_ids: Set[str] = set()
    visited_positions: List[List[float]] = [list(start_pos)]
    total_detected = 0
    total_merged = 0
    micro_step = 0
    t_start = time.time()
    stale_counter = 0
    prev_free_count = 0
    disabled_frontiers: Set[Tuple[int, int]] = set()  # 不可达frontier的栅格坐标

    # --- 实时可视化 ---
    viz = None
    if live_viz:
        from scripts.live_visualizer import LiveVisualizer
        viz_dir = os.path.join(output_dir, "live_viz")
        viz = LiveVisualizer(
            panel_size=360, wait_ms=1, save_dir=viz_dir, record_video=True,
        )
        print(f"[实时可视化] 已启用, 录制: {viz_dir}/exploration.mp4")

    # --- Voronoi 全局导航图 (从 navmesh/栅格一次性构建, 探索期间不变) ---
    from semantic_map_Create.voronoi_graph import VoronoiNavigator, build_voronoi_graph
    voronoi_nav = VoronoiNavigator(occ_grid)
    room_seg_result_init = None
    try:
        from semantic_map_Create.room_segmentation import segment_rooms_from_occ_grid
        room_seg_result_init = segment_rooms_from_occ_grid(occ_grid)
        if room_seg_result_init["n_rooms"] > 0:
            G = build_voronoi_graph(occ_grid, room_labels=room_seg_result_init["room_labels"])
            voronoi_nav.set_graph(G)
            print(f"[Voronoi] 全局导航图构建完成: {G.number_of_nodes()} 节点, "
                  f"{G.number_of_edges()} 边")
        else:
            print(f"[Voronoi] 房间分割无结果, Voronoi 导航不可用")
    except Exception as e:
        print(f"[Voronoi] 构建失败: {e}")

    # ---------------------------------------------------------------
    # 辅助: 在当前位置做一次完整感知周期
    # ---------------------------------------------------------------
    def observe_and_update(step_idx: int) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """360°环视 → 深度→栅格 + 物体检测 → 地图合并.
        
        Returns: (visible detections list, front_rgb — 正前方视角 RGB)
        """
        nonlocal floors_history, all_obj_ids, total_detected, total_merged

        pos = agent.get_position()
        heading_start = agent.get_heading_deg()
        all_obs = agent.panoramic_observe(n_views)

        # --- 深度 → 占据栅格 + fog-of-war ---
        heading = heading_start
        for obs in all_obs:
            if "depth" in obs:
                occ_grid.update_from_habitat_depth(
                    depth=obs["depth"],
                    intrinsics=cam_intrinsics,
                    agent_pos=pos,
                    agent_heading_deg=heading,
                    max_depth=5.0,
                )
            heading += 360.0 / n_views

        # --- 物体检测 ---
        visible = []
        front_rgb = None
        heading = heading_start
        _vi = 0
        for obs in all_obs:
            rgb = obs.get("color")
            depth_map = obs.get("depth")
            if rgb is None:
                heading += 360.0 / n_views
                continue

            rgb_np = rgb[:, :, :3].copy()
            if _vi == 0:
                front_rgb = rgb_np
            if depth_map is not None:
                dets = detector.detect_with_depth(
                    rgb_np, depth_map, detect_text_prompt,
                    cam_intrinsics, np.array(pos), heading,
                    max_depth=5.0,
                    box_threshold=0.35, text_threshold=0.30,
                )
            else:
                dets = detector.detect(rgb_np, detect_text_prompt,
                                      box_threshold=0.35, text_threshold=0.30)

            # CLIP 编码
            if dets:
                ensure_clip()
                compute_clip_embeddings_for_detections(
                    rgb_np, dets,
                    clip_cache["model"], clip_cache["processor"],
                    clip_cache["device"],
                    mode=clip_encode_mode,
                    masked_weight=clip_masked_weight,
                    bbox_padding=clip_bbox_padding,
                )

            _img_area = rgb_np.shape[0] * rgb_np.shape[1]
            for d in dets:
                lbl = d.get("label", "")
                if lbl.startswith("##") or len(lbl) <= 1:
                    continue
                if lbl.lower() in STRUCTURAL_CATEGORIES:
                    continue
                if lbl not in detect_labels_set:
                    continue
                _bb = d.get("bbox_xyxy")
                if _bb is not None:
                    _bw = max(_bb[2] - _bb[0], 0)
                    _bh = max(_bb[3] - _bb[1], 0)
                    _ba = _bw * _bh
                    if _ba < _img_area * 0.001 or _ba > _img_area * 0.8:
                        continue
                d["_view_index"] = _vi
                d["_rgb"] = rgb_np
                visible.append(d)
            heading += 360.0 / n_views
            _vi += 1

        total_detected += len(visible)

        # 构建 floors_now 并增量合并
        if visible:
            obs_objects, all_obj_ids = objects_from_detection(
                visible, pos, "R0", clock, all_obj_ids, step=step_idx,
            )
            obs_objects = dedup_intra_frame(obs_objects, dist_threshold=dedup_dist)
            total_merged += len(obs_objects)

            hist_floor_id = floors_history[0].floor_id if floors_history else "F0"
            floors_now = build_floors_now(obs_objects, room_id="R0", floor_id=hist_floor_id)

            if floors_history:
                try:
                    floors_merged, warns = run_merge(
                        floors_now, floors_history, cfg,
                        shape_check=False, allow_new_floors=True, allow_new_rooms=True,
                    )
                    floors_history = floors_merged
                except Exception as e:
                    print(f"    [合并错误] {e}")
            else:
                floors_history = floors_now

        clock.step()
        return visible, front_rgb

    # ---------------------------------------------------------------
    # 辅助: 选择最佳 frontier
    # ---------------------------------------------------------------
    def select_best_frontier() -> Optional[List[float]]:
        """从栅格检测 frontier, 选择距已访问位置最远的.
        
        策略 (参考 BeliefMapNav):
          1. 检测 frontier (已探索与未探索的边界)
          2. 过滤已标记不可达的 frontier
          3. 选择距离所有已访问位置最远的 (鼓励广度探索)
        """
        frontiers = occ_grid.get_frontiers(min_area_m2=frontier_min_area_m2)
        if len(frontiers) == 0:
            return None

        # 过滤不可达 frontier
        valid_frontiers = []
        for f in frontiers:
            gc = occ_grid.world_to_grid(np.array([f[0], f[1]]))
            key = (int(gc[0]) // 5, int(gc[1]) // 5)  # 5-cell 分桶
            if key not in disabled_frontiers:
                valid_frontiers.append(f)
        
        if not valid_frontiers:
            return None

        # 选择距已访问位置最远的 frontier
        best = None
        best_score = -np.inf
        for f in valid_frontiers:
            if visited_positions:
                dists = [math.sqrt((f[0]-vp[0])**2 + (f[1]-vp[2])**2)
                         for vp in visited_positions]
                score = min(dists)  # 距最近已访问点的距离
            else:
                score = 0.0
            if score > best_score:
                best_score = score
                best = f

        return best

    # ---------------------------------------------------------------
    # 辅助: 规划路径到 frontier
    # ---------------------------------------------------------------
    def plan_path_to(target_world_xz: List[float]) -> Optional[List[list]]:
        """从当前位置规划到目标的路径, 返回 3D waypoints.
        
        路径规划层级:
          1. Voronoi 图路径 (coarse, 走房间中心/门口)
          2. navmesh shortest_path (精确)
          3. A* on occupancy grid (纯自主)
        """
        pos = agent.get_position()
        y_val = float(pos[1])
        target_3d = [target_world_xz[0], y_val, target_world_xz[1]]

        # 1. Try Voronoi routing (if graph available)
        if voronoi_nav.has_graph():
            vpath = voronoi_nav.plan_path_3d(list(pos), target_3d, step_size=step_size)
            if vpath and len(vpath) > 0:
                return vpath

        # 2. Fallback: navmesh or A*
        if use_navmesh:
            waypoints = agent.get_path_waypoints(target_3d, step_size=step_size)
            return waypoints
        else:
            # A* on occupancy grid
            if astar_planner is not None:
                astar_planner.invalidate_cost_map()
                start_3d = [float(pos[0]), y_val, float(pos[2])]
                try:
                    waypoints_3d = astar_planner.plan_3d(start_3d, target_3d, step_size=step_size)
                    if waypoints_3d and len(waypoints_3d) > 1:
                        return waypoints_3d[1:]
                except Exception:
                    pass
            # fallback: navmesh shortest path
            return agent.get_path_waypoints(target_3d, step_size=step_size)

    # ---------------------------------------------------------------
    # Phase 0: 起始位置首次观测
    # ---------------------------------------------------------------
    print(f"[Phase 0] 起始位置首次360°环视...")
    visible, _front_rgb = observe_and_update(micro_step)
    free_count = int(np.sum(occ_grid.grid == 1))
    explored_count = int(np.sum(occ_grid.explored)) if hasattr(occ_grid, 'explored') else free_count
    n_obj = sum(len(rm.objects) for fl in floors_history for rm in fl.rooms) if floors_history else 0
    print(f"  检测到 {len(visible)} 个物体, 地图物体: {n_obj}")
    print(f"  free cells: {free_count}, explored: {explored_count}")
    prev_free_count = free_count

    # 保存初始 RGB + 栅格
    obs_snap = agent.observe()
    if "color" in obs_snap:
        cv2.imwrite(
            os.path.join(rgb_dir, f"step_{micro_step:04d}.jpg"),
            cv2.cvtColor(obs_snap["color"][:, :, :3], cv2.COLOR_RGB2BGR),
        )

    # ---------------------------------------------------------------
    # 主循环: Frontier 驱动全覆盖探索
    # ---------------------------------------------------------------
    print(f"\n[主循环] Frontier + Voronoi 驱动全覆盖探索开始...")
    no_frontier_count = 0
    MAX_NO_FRONTIER = 3  # 连续N轮无frontier则停止

    while micro_step < max_steps:
        # --- 1. 选择下一个目标: Voronoi 交叉点 > Frontier ---
        target_xz = None
        target_label = "frontier"

        # 优先: Voronoi 未访问的交叉点/跨房间节点 (门口、走廊交汇)
        if voronoi_nav.has_graph():
            vtargets = voronoi_nav.get_unvisited_exploration_targets(
                list(agent.get_position()), visited_positions,
                prefer_junctions=True, max_targets=3,
            )
            if vtargets:
                vt = vtargets[0]
                target_xz = [vt["pos_world"][0], vt["pos_world"][1]]
                target_label = f"voronoi-{vt['type']}(d{vt['degree']})"

        # 回退: Frontier 目标
        if target_xz is None:
            frontier_target = select_best_frontier()

            if frontier_target is None:
                no_frontier_count += 1
                print(f"\n  [导航] 无有效目标 (Voronoi+Frontier均空, 连续{no_frontier_count}/{MAX_NO_FRONTIER})")
                if no_frontier_count >= MAX_NO_FRONTIER:
                    print(f"  ✓ 无更多探索目标, 探索完成!")
                    break
                # 随机走一步, 可能打开新区域
                if use_navmesh:
                    rp = sim.pathfinder.get_random_navigable_point()
                    if not np.isnan(rp).any():
                        agent.navigate_to(list(rp))
                        visited_positions.append(list(agent.get_position()))
                micro_step += 1
                _, _front_rgb = observe_and_update(micro_step)
                continue
            else:
                target_xz = frontier_target
                target_label = "frontier"
                no_frontier_count = 0
        else:
            no_frontier_count = 0

        # --- 2. 规划路径到目标 ---
        waypoints = plan_path_to(target_xz)
        if waypoints is None or len(waypoints) == 0:
            # 标记该目标不可达
            gc = occ_grid.world_to_grid(np.array([target_xz[0], target_xz[1]]))
            disabled_frontiers.add((int(gc[0]) // 5, int(gc[1]) // 5))
            print(f"  [路径] 目标不可达, 已禁用, 尝试下一个")
            micro_step += 1
            continue

        n_wp = len(waypoints)
        dist_est = n_wp * step_size
        print(f"\n{'='*50}")
        print(f"  [宏目标] {target_label} [{target_xz[0]:.1f}, {target_xz[1]:.1f}], "
              f"步={micro_step}/{max_steps}")
        print(f"  [路径] {n_wp}个waypoint (~{dist_est:.1f}m)")

        # --- 3. 沿路径逐步走, 每步做感知 ---
        for wi, wp in enumerate(waypoints):
            if micro_step >= max_steps:
                break

            agent.move_to_waypoint_teleport(wp)
            pos = agent.get_position()
            visited_positions.append([float(pos[0]), float(pos[1]), float(pos[2])])

            # 360°环视 + 感知
            visible, _front_rgb = observe_and_update(micro_step)

            # 每5步或最后一步保存 RGB + 栅格快照
            if micro_step % 5 == 0 or wi == n_wp - 1:
                obs_snap = agent.observe()
                if "color" in obs_snap:
                    cv2.imwrite(
                        os.path.join(rgb_dir, f"step_{micro_step:04d}.jpg"),
                        cv2.cvtColor(obs_snap["color"][:, :, :3], cv2.COLOR_RGB2BGR),
                    )
                # 保存栅格快照 (每5步, 用于 GIF)
                grid_img = occ_grid.to_image(navigable=True, show_layers=True)
                gc_agent = occ_grid.world_to_grid(np.array([pos[0], pos[2]]))
                cv2.circle(grid_img, (int(gc_agent[0]), int(gc_agent[1])), 4, (0, 255, 0), -1)
                cv2.imwrite(os.path.join(rgb_dir, f"grid_{micro_step:04d}.jpg"), grid_img)

            # --- 实时可视化更新 ---
            if viz is not None:
                _rgb = _front_rgb  # 使用环视中的正前方帧, 避免重新observe导致帧不匹配
                _all_objs = []
                _n_obj = 0
                if floors_history:
                    for fl in floors_history:
                        for rm in fl.rooms:
                            for obj in rm.objects:
                                _n_obj += 1
                                if obj.pos_3d:
                                    _all_objs.append({
                                        "label": obj.label,
                                        "pos_3d": obj.pos_3d,
                                        "color": (0, 200, 255),
                                    })
                _front_dets = [d for d in visible if d.get("_view_index", 0) == 0]
                _progress = micro_step / max_steps if max_steps > 0 else 0
                _n_floors = len(floors_history) if floors_history else 1
                _cur_floor = floors_history[0].floor_id if floors_history else "F0"
                viz_continue = viz.update(
                    step=micro_step,
                    rgb=_rgb,
                    detections=_front_dets,
                    occ_grid=occ_grid,
                    agent_pos=np.array(pos),
                    nav_target=[target_xz[0], 0.0, target_xz[1]] if target_xz else None,
                    nav_mode=target_label,
                    target_name="exploration",
                    n_objects=_n_obj,
                    visited_positions=visited_positions,
                    objects_on_map=_all_objs,
                    exploration_progress=_progress,
                    n_floors=_n_floors,
                    current_floor_id=_cur_floor,
                )
                if not viz_continue:
                    print("[LiveViz] 用户退出")
                    micro_step = max_steps  # force exit
                    break

            # 进度报告 (每10步)
            if micro_step % 10 == 0:
                free_count = int(np.sum(occ_grid.grid == 1))
                n_obj = sum(len(rm.objects) for fl in floors_history for rm in fl.rooms) if floors_history else 0
                n_frontiers = len(occ_grid.get_frontiers(min_area_m2=frontier_min_area_m2))
                elapsed = time.time() - t_start
                print(f"  step {micro_step}: free={free_count}, obj={n_obj}, "
                      f"frontiers={n_frontiers}, elapsed={elapsed:.0f}s")

                # 停滞检测: free cell增长停滞
                if free_count <= prev_free_count + 10:
                    stale_counter += 1
                else:
                    stale_counter = 0
                prev_free_count = free_count

                if stale_counter >= stale_threshold // 10:
                    print(f"  ⚠ 探索停滞 ({stale_counter}次无增长)")

            micro_step += 1

        # --- 到达 frontier 后, 额外环视 ---
        if micro_step < max_steps:
            visible, _front_rgb = observe_and_update(micro_step)

    sim.close()
    if viz is not None:
        viz.close()
        print(f"[实时可视化] 录制完成")

    # ---------------------------------------------------------------
    # Phase 2: 房间分割 + Voronoi 导航图
    # ---------------------------------------------------------------
    final_free = int(np.sum(occ_grid.grid == 1))
    final_occ = int(np.sum(occ_grid.grid == 2))
    print(f"\n{'='*60}")
    print(f"  Phase 2: 房间分割 + Voronoi 导航图")
    print(f"  栅格状态: free={final_free}, occupied={final_occ}")
    print(f"  探索步数: {micro_step}/{max_steps}")
    print(f"{'='*60}")

    rooms = segment_rooms_from_grid(
        occ_grid, min_room_area_m2=2.0,
        save_debug_dir=os.path.join(output_dir, "room_segmentation_debug"),
    )

    # 复用启动时构建的全局 Voronoi 图 (不重建)
    voronoi_nav_graph = voronoi_nav.graph
    room_seg_result = room_seg_result_init
    if voronoi_nav_graph is not None:
        print(f"  [Voronoi] 复用全局导航图: {voronoi_nav_graph.number_of_nodes()} 节点")
    else:
        print(f"  [Voronoi] 无全局导航图")

    # ---------------------------------------------------------------
    # 楼层检测 + 按楼层分配房间
    # ---------------------------------------------------------------
    all_objects = []
    for fl in floors_history:
        for rm in fl.rooms:
            all_objects.extend(rm.objects)

    floor_ranges = detect_floors(all_objects)
    n_floors_detected = len(floor_ranges)
    print(f"\n[楼层检测] 检测到 {n_floors_detected} 个楼层 (最大Y间距分割)")
    for fr in floor_ranges:
        print(f"  {fr['floor_id']}: y ∈ [{fr['y_min']:.2f}, {fr['y_max']:.2f}], "
              f"{len(fr['obj_indices'])} 个物体")

    # 确定机器人所在楼层 (探索的楼层)
    agent_y_values = [p[1] for p in visited_positions if len(p) >= 2]
    agent_y_median = float(np.median(agent_y_values)) if agent_y_values else 0.0
    explored_floor_id = floor_ranges[0]["floor_id"]  # fallback
    for fr in floor_ranges:
        if fr["y_min"] <= agent_y_median <= fr["y_max"]:
            explored_floor_id = fr["floor_id"]
            break
    print(f"  机器人所在楼层: {explored_floor_id} (agent_y_median={agent_y_median:.2f})")

    # 房间分割结果 (rooms) 仅对探索楼层有效
    room_polygons = [{"room_id": r["room_id"], "polygon": r["polygon"]} for r in rooms]

    # 按楼层构建 Floor 对象
    room_obj_map: Dict[str, List[Object]] = {}
    reassigned = 0
    new_floors = []

    for fr in floor_ranges:
        floor_objs = [all_objects[i] for i in fr["obj_indices"]]

        if fr["floor_id"] == explored_floor_id and rooms:
            # 已探索楼层: 用 room polygon 分配房间
            for obj in floor_objs:
                new_rid = assign_room_id_by_polygon(obj.pos_3d, room_polygons)
                if new_rid and new_rid != obj.room_id:
                    obj.room_id = new_rid
                    reassigned += 1

            # 按 room_id 分组
            floor_room_map: Dict[str, List[Object]] = {r["room_id"]: [] for r in rooms}
            floor_room_map["R_unassigned"] = []
            for obj in floor_objs:
                rid = obj.room_id if obj.room_id in floor_room_map else "R_unassigned"
                floor_room_map[rid].append(obj)
            room_obj_map.update(floor_room_map)

            floor_rooms = []
            for r in rooms:
                rid = r["room_id"]
                if floor_room_map.get(rid):
                    floor_rooms.append(Room(
                        room_id=rid,
                        objects=floor_room_map[rid],
                        region=r["polygon"],
                        floor_id=fr["floor_id"],
                    ))
            if floor_room_map["R_unassigned"]:
                floor_rooms.append(Room(
                    room_id="R_unassigned",
                    objects=floor_room_map["R_unassigned"],
                    region=[],
                    floor_id=fr["floor_id"],
                ))
        else:
            # 未探索楼层: 所有物体放入单个房间
            uncharted_rid = f"{fr['floor_id']}_R0"
            for obj in floor_objs:
                obj.room_id = uncharted_rid
            floor_rooms = [Room(
                room_id=uncharted_rid,
                objects=floor_objs,
                region=[],
                floor_id=fr["floor_id"],
            )]
            room_obj_map[uncharted_rid] = floor_objs

        floor = Floor(
            floor_id=fr["floor_id"],
            rooms=floor_rooms,
            z_range={"z_min": fr["y_min"], "z_max": fr["y_max"]},
        )
        new_floors.append(floor)
        n_objs_on_floor = sum(len(rm.objects) for rm in floor_rooms)
        print(f"  {fr['floor_id']}: {len(floor_rooms)} 个房间, {n_objs_on_floor} 个物体")

    floors_history = new_floors
    print(f"[房间分配] {reassigned} 个物体 room_id 被更新")

    # --- CLIP 房间语义标签 ---
    if rooms and clip_encode_mode != "none":
        try:
            ensure_clip()
            from semantic_map_Create.room_segmentation import label_rooms_with_clip
            objects_by_room = {r["room_id"]: room_obj_map.get(r["room_id"], []) for r in rooms}
            label_rooms_with_clip(
                rooms, objects_by_room,
                clip_cache["model"], clip_cache["processor"], clip_cache["device"],
            )
            # 将标签同步到 Room.room_name
            for fl in floors_history:
                for rm in fl.rooms:
                    for r in rooms:
                        if r["room_id"] == rm.room_id and "room_label" in r:
                            rm.room_name = {"en": r["room_label"]}
                            break
            print("[CLIP 房间标签]")
            for r in rooms:
                label = r.get("room_label", "?")
                scores = r.get("room_label_scores", [])
                top3 = ", ".join(f"{s[0]}({s[1]:.2f})" for s in scores[:3])
                print(f"  {r['room_id']} → {label} [{top3}]")
        except Exception as e:
            print(f"[CLIP 房间标签] 失败: {e}")

    # ---------------------------------------------------------------
    # 导出
    # ---------------------------------------------------------------
    final_obj_count = sum(len(rm.objects) for fl in floors_history for rm in fl.rooms)
    total_distance = sum(
        math.sqrt(sum((a - b)**2 for a, b in zip(visited_positions[i], visited_positions[i+1])))
        for i in range(len(visited_positions) - 1)
    ) if len(visited_positions) > 1 else 0.0

    print(f"\n{'='*60}")
    print(f"  深度探索完成!")
    print(f"  楼层数: {len(floors_history)}")
    for fl in floors_history:
        fl_objs = sum(len(rm.objects) for rm in fl.rooms)
        zr = fl.z_range or {}
        print(f"    {fl.floor_id}: {len(fl.rooms)} 房间, {fl_objs} 物体, "
              f"y=[{zr.get('z_min',0):.2f}, {zr.get('z_max',0):.2f}]")
    print(f"  物体总数: {final_obj_count}")
    print(f"  房间数: {len(rooms)} (已探索楼层)")
    print(f"  Voronoi节点: {voronoi_nav_graph.number_of_nodes() if voronoi_nav_graph else 0}")
    print(f"  总检测: {total_detected}, 合并入图: {total_merged}")
    print(f"  探索步数: {micro_step}, 行走距离: {total_distance:.1f}m")
    print(f"  栅格: free={final_free}, occ={final_occ}")
    print(f"  耗时: {time.time()-t_start:.1f}s")
    print(f"{'='*60}")

    # 保存语义地图 JSON
    map_json = os.path.join(output_dir, f"{scene_name}_semantic_map.json")
    map_data = [fl.to_dict() for fl in floors_history]
    with open(map_json, "w", encoding="utf-8") as f:
        json.dump(map_data, f, ensure_ascii=False, indent=2)
    print(f"[已保存] {map_json}")

    # 保存房间分割信息
    rooms_json = os.path.join(output_dir, f"{scene_name}_rooms.json")
    rooms_export = [{k: v for k, v in r.items() if k != "grid_mask"} for r in rooms]
    with open(rooms_json, "w", encoding="utf-8") as f:
        json.dump(rooms_export, f, ensure_ascii=False, indent=2)
    print(f"[已保存] {rooms_json}")

    # 保存占据栅格
    grid_dir = os.path.join(output_dir, "occupancy_grid")
    os.makedirs(grid_dir, exist_ok=True)
    np.save(os.path.join(grid_dir, "occupancy_grid.npy"), occ_grid.grid)
    np.savez(os.path.join(grid_dir, "occupancy_meta.npz"),
             resolution=occ_grid.resolution,
             origin=np.array(occ_grid._origin))
    # 保存密度图 (用于离线重跑房间分割)
    np.save(os.path.join(grid_dir, "wall_density.npy"), occ_grid.wall_density)
    np.save(os.path.join(grid_dir, "full_density.npy"), occ_grid.full_density)
    # 可视化占据栅格
    grid_img = occ_grid.to_image(navigable=True, show_layers=True)
    cv2.imwrite(os.path.join(grid_dir, "occupancy_grid.png"), grid_img)
    print(f"[已保存] {grid_dir}/")

    # 保存房间分割可视化
    _save_room_visualization(occ_grid, rooms, output_dir, scene_name,
                             floors_history=floors_history,
                             explored_floor_id=explored_floor_id)

    # 保存轨迹可视化 (在占据栅格上)
    try:
        from scripts.nav_core.visualization import (
            visualize_trajectory_on_occ_grid as _viz_traj_occ,
        )
        _viz_traj_occ(occ_grid, visited_positions, [], "exploration", output_dir)
    except Exception as e:
        print(f"[轨迹可视化] 失败: {e}")

    # 保存 Voronoi 导航图可视化
    if voronoi_nav_graph is not None and voronoi_nav_graph.number_of_nodes() > 0:
        try:
            from semantic_map_Create.voronoi_graph import visualize_voronoi_graph
            visualize_voronoi_graph(
                voronoi_nav_graph, occ_grid,
                room_labels=room_seg_result["room_labels"] if room_seg_result else None,
                save_path=os.path.join(output_dir, "voronoi_graph.png"),
            )
            print(f"[已保存] voronoi_graph.png")
        except Exception as e:
            print(f"[Voronoi可视化] 失败: {e}")

        # 保存 Voronoi 导航图 JSON (供任务模式加载复用)
        try:
            voronoi_nav.save(os.path.join(output_dir, "voronoi_nav_graph.json"))
        except Exception as e:
            print(f"[Voronoi序列化] 失败: {e}")

    # 保存房间分割结果可视化
    if room_seg_result is not None and room_seg_result.get("n_rooms", 0) > 0:
        try:
            from semantic_map_Create.room_segmentation import visualize_room_segmentation
            visualize_room_segmentation(
                room_seg_result["room_labels"],
                room_seg_result["walls_skeleton"],
                room_seg_result["room_centers"],
                occ_grid,
                save_path=os.path.join(output_dir, "room_segmentation.png"),
            )
            print(f"[已保存] room_segmentation.png")
        except Exception:
            pass

    # --- HOV-SG 风格分层语义地图可视化 ---
    try:
        _generate_hierarchical_scene_graph(
            occ_grid, rooms, floors_history, voronoi_nav_graph, room_seg_result,
            visited_positions, output_dir, scene_name,
        )
    except Exception as e:
        print(f"[分层语义地图] 失败: {e}")
        import traceback; traceback.print_exc()

    # --- 探索过程 GIF ---
    try:
        _generate_exploration_gif(output_dir, rgb_dir)
    except Exception as e:
        print(f"[GIF生成] 失败: {e}")

    # --- 3D 场景图 GIF (HOV-SG 风格) ---
    try:
        _generate_3d_scene_graph_gif(
            occ_grid, rooms, floors_history, visited_positions,
            output_dir, scene_name,
        )
    except Exception as e:
        print(f"[3D GIF] 失败: {e}")

    return {
        "floors": floors_history,
        "rooms": rooms,
        "voronoi_graph": voronoi_nav_graph,
        "stats": {
            "steps": micro_step,
            "total_distance_m": round(total_distance, 1),
            "total_detected": total_detected,
            "total_merged": total_merged,
            "final_objects": final_obj_count,
            "n_rooms": len(rooms),
            "n_voronoi_nodes": voronoi_nav_graph.number_of_nodes() if voronoi_nav_graph else 0,
            "free_cells": final_free,
            "occupied_cells": final_occ,
            "elapsed_s": round(time.time() - t_start, 1),
        },
    }


def _save_room_visualization(
    occ_grid: OccupancyGrid,
    rooms: List[Dict[str, Any]],
    output_dir: str,
    scene_name: str,
    floors_history: Optional[List] = None,
    explored_floor_id: Optional[str] = None,
):
    """保存房间分割彩色可视化 (含楼层信息)."""
    H, W = occ_grid.shape
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    vis[occ_grid.grid == 2] = [40, 40, 40]  # obstacles = dark gray

    colors = [
        (66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
        (171, 71, 188), (0, 172, 193), (255, 112, 67), (158, 157, 36),
        (121, 85, 72), (96, 125, 139),
    ]
    for i, r in enumerate(rooms):
        mask = r.get("grid_mask")
        if mask is not None:
            color = colors[i % len(colors)]
            vis[mask > 0] = color
            # 标注房间 ID + 标签
            ys, xs = np.where(mask > 0)
            cy, cx = int(ys.mean()), int(xs.mean())
            label = r.get("room_label", r["room_id"])
            cv2.putText(vis, f'{r["room_id"]}: {label}', (cx-20, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 楼层信息标注 (左上角)
    if floors_history:
        n_floors = len(floors_history)
        title = f"Floors: {n_floors}"
        if explored_floor_id:
            title += f"  |  Explored: {explored_floor_id}"
        cv2.putText(vis, title, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_off = 40
        for fl in floors_history:
            zr = fl.z_range or {}
            n_objs = sum(len(rm.objects) for rm in fl.rooms)
            info = (f"{fl.floor_id}: {len(fl.rooms)}R, {n_objs} objs, "
                    f"y=[{zr.get('z_min',0):.1f},{zr.get('z_max',0):.1f}]")
            cv2.putText(vis, info, (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
            y_off += 18

    out_path = os.path.join(output_dir, f"{scene_name}_rooms_vis.png")
    cv2.imwrite(out_path, vis)
    print(f"[已保存] {out_path}")


def _generate_hierarchical_scene_graph(
    occ_grid: OccupancyGrid,
    rooms: List[Dict[str, Any]],
    floors_history: List,
    voronoi_graph,
    room_seg_result: Optional[Dict],
    visited_positions: List,
    output_dir: str,
    scene_name: str,
):
    """HOV-SG 风格分层语义地图可视化 (支持多楼层).

    生成一张大图, 包含:
      - 左侧: 占据栅格 + 房间着色 + 轨迹 + Voronoi 边 + 物体位置(按楼层着色)
      - 右侧: 分层图 (Scene → Floor → Room → Objects 树结构)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patches as mpatches

    fig = plt.figure(figsize=(24, 12), facecolor='white')

    # ===== 左侧: 空间地图 =====
    ax_map = fig.add_axes([0.02, 0.05, 0.48, 0.90])

    H, W = occ_grid.shape
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    vis[occ_grid.grid == 2] = [40, 40, 40]
    vis[occ_grid.explored] = [20, 20, 20]
    vis[occ_grid.grid == 1] = [30, 30, 30]

    room_colors_rgb = [
        (66/255, 133/255, 244/255), (219/255, 68/255, 55/255),
        (244/255, 180/255, 0/255), (15/255, 157/255, 88/255),
        (171/255, 71/255, 188/255), (0/255, 172/255, 193/255),
        (255/255, 112/255, 67/255), (158/255, 157/255, 36/255),
        (121/255, 85/255, 72/255), (96/255, 125/255, 139/255),
    ]
    room_colors_bgr = [(int(c[2]*255), int(c[1]*255), int(c[0]*255)) for c in room_colors_rgb]

    # 楼层颜色
    floor_colors_rgb = [
        (0.26, 0.52, 0.96),  # 蓝
        (0.96, 0.40, 0.26),  # 红橙
        (0.20, 0.70, 0.40),  # 绿
        (0.70, 0.33, 0.80),  # 紫
    ]

    for i, r in enumerate(rooms):
        mask = r.get("grid_mask")
        if mask is not None:
            color = room_colors_bgr[i % len(room_colors_bgr)]
            vis[mask > 0] = color

    # Voronoi 边
    if voronoi_graph is not None:
        for u, v in voronoi_graph.edges():
            u_data = voronoi_graph.nodes[u]
            v_data = voronoi_graph.nodes[v]
            r1, c1 = int(u_data.get("row", 0)), int(u_data.get("col", 0))
            r2, c2 = int(v_data.get("row", 0)), int(v_data.get("col", 0))
            cv2.line(vis, (c1, r1), (c2, r2), (80, 80, 80), 1)

    # 物体位置 — 按楼层着色
    floor_dot_colors = [
        (255, 255, 255),   # F0 白色
        (100, 255, 100),   # F1 绿色
        (100, 100, 255),   # F2 蓝色
        (255, 100, 255),   # F3 紫色
    ]
    for fi, fl in enumerate(floors_history):
        dot_color = floor_dot_colors[fi % len(floor_dot_colors)]
        for rm in fl.rooms:
            for obj in rm.objects:
                pos_2d = getattr(obj, 'pos_2d', None)
                if pos_2d is None:
                    continue
                if isinstance(pos_2d, dict):
                    wx, wz = pos_2d.get("x", 0), pos_2d.get("y", 0)
                else:
                    wx, wz = float(pos_2d[0]), float(pos_2d[1])
                gc = occ_grid.world_to_grid(np.array([wx, wz]))
                c, r_coord = int(gc[0]), int(gc[1])
                if 0 <= c < W and 0 <= r_coord < H:
                    cv2.circle(vis, (c, r_coord), 2, dot_color, -1)

    # 轨迹
    if visited_positions:
        for i in range(1, len(visited_positions)):
            p1 = visited_positions[i-1]
            p2 = visited_positions[i]
            gc1 = occ_grid.world_to_grid(np.array([p1[0], p1[2]]))
            gc2 = occ_grid.world_to_grid(np.array([p2[0], p2[2]]))
            cv2.line(vis, (int(gc1[0]), int(gc1[1])), (int(gc2[0]), int(gc2[1])),
                     (0, 200, 255), 1)

    # 房间标签
    for i, r in enumerate(rooms):
        mask = r.get("grid_mask")
        if mask is not None:
            ys, xs = np.where(mask > 0)
            if len(ys) > 0:
                cy, cx = int(ys.mean()), int(xs.mean())
                label = r.get("room_label", r["room_id"])
                text = f'{r["room_id"]}: {label}'
                cv2.putText(vis, text, (cx-30, cy+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    ax_map.imshow(vis_rgb, origin='upper')
    n_floors = len(floors_history)
    ax_map.set_title(f'Scene: {scene_name}  |  {n_floors} Floor(s)  |  '
                     f'Spatial Map + Voronoi + Objects', fontsize=12)
    ax_map.axis('off')

    # 楼层图例 (物体点颜色)
    floor_legend = []
    for fi, fl in enumerate(floors_history):
        dc = floor_dot_colors[fi % len(floor_dot_colors)]
        dc_rgb = (dc[0]/255, dc[1]/255, dc[2]/255)
        zr = fl.z_range or {}
        n_objs = sum(len(rm.objects) for rm in fl.rooms)
        floor_legend.append(mpatches.Patch(
            color=dc_rgb,
            label=f'{fl.floor_id}: y=[{zr.get("z_min",0):.1f},{zr.get("z_max",0):.1f}] ({n_objs} objs)'
        ))
    if floor_legend:
        ax_map.legend(handles=floor_legend, loc='upper left', fontsize=7,
                      framealpha=0.8, facecolor='black', labelcolor='white',
                      title='Floors', title_fontsize=8)

    # ===== 右侧: 分层图 (Scene → Floor → Room → Objects) =====
    ax_tree = fig.add_axes([0.52, 0.05, 0.46, 0.90])
    ax_tree.set_xlim(0, 10)
    ax_tree.set_ylim(0, 10)
    ax_tree.axis('off')
    ax_tree.set_title('Hierarchical Scene Graph (Multi-Floor)', fontsize=12)

    # 场景根节点
    scene_x, scene_y = 5.0, 9.5
    scene_box = FancyBboxPatch((scene_x - 0.9, scene_y - 0.2), 1.8, 0.4,
                                boxstyle="round,pad=0.05", facecolor='#333333',
                                edgecolor='white', linewidth=2)
    ax_tree.add_patch(scene_box)
    ax_tree.text(scene_x, scene_y, f'Scene ({n_floors}F)',
                 ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    # 每个楼层的布局
    # 动态计算布局: Floor 节点 → Room 节点 → Object 节点
    if n_floors == 1:
        floor_positions = [5.0]  # 居中
    else:
        floor_spacing = min(8.0 / max(n_floors, 1), 4.0)
        floor_start_x = scene_x - (n_floors - 1) * floor_spacing / 2
        floor_positions = [floor_start_x + i * floor_spacing for i in range(n_floors)]

    floor_y_pos = 8.2

    global_room_idx = 0  # 用于全局颜色分配

    for fi, fl in enumerate(floors_history):
        fx = floor_positions[fi]
        fc = floor_colors_rgb[fi % len(floor_colors_rgb)]
        zr = fl.z_range or {}
        fl_n_objs = sum(len(rm.objects) for rm in fl.rooms)

        # Scene → Floor 连线
        ax_tree.plot([scene_x, fx], [scene_y - 0.2, floor_y_pos + 0.2],
                    color='gray', linewidth=1.5, zorder=0)

        # Floor 节点
        floor_box = FancyBboxPatch((fx - 0.8, floor_y_pos - 0.2), 1.6, 0.4,
                                    boxstyle="round,pad=0.05", facecolor=fc,
                                    edgecolor='white', linewidth=2)
        ax_tree.add_patch(floor_box)
        y_label = f'y=[{zr.get("z_min",0):.1f},{zr.get("z_max",0):.1f}]'
        ax_tree.text(fx, floor_y_pos + 0.05, f'{fl.floor_id} ({fl_n_objs})',
                     ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        ax_tree.text(fx, floor_y_pos - 0.10, y_label,
                     ha='center', va='center', fontsize=6, color='white')

        # 房间节点
        fl_rooms = fl.rooms
        n_rooms = len(fl_rooms)
        if n_rooms == 0:
            continue

        room_y_pos = 6.5
        # 房间水平分布: 以 fx 为中心
        if n_floors == 1:
            room_x_span = 8.0
        else:
            room_x_span = min(8.0 / n_floors, 4.0)
        room_spacing = min(room_x_span / max(n_rooms, 1), 1.8)
        room_start_x = fx - (n_rooms - 1) * room_spacing / 2

        for ri, rm in enumerate(fl_rooms):
            rx = room_start_x + ri * room_spacing
            color = room_colors_rgb[global_room_idx % len(room_colors_rgb)]
            global_room_idx += 1

            # Floor → Room 连线
            ax_tree.plot([fx, rx], [floor_y_pos - 0.2, room_y_pos + 0.25],
                        color='gray', linewidth=1.2, zorder=0)

            room_label = rm.room_name.get("en", rm.room_id) if rm.room_name else rm.room_id
            room_box = FancyBboxPatch((rx - 0.6, room_y_pos - 0.20), 1.2, 0.4,
                                       boxstyle="round,pad=0.05", facecolor=color,
                                       edgecolor='white', linewidth=1.5, alpha=0.9)
            ax_tree.add_patch(room_box)
            ax_tree.text(rx, room_y_pos + 0.05, rm.room_id, ha='center', va='center',
                         fontsize=7, color='white', fontweight='bold')
            ax_tree.text(rx, room_y_pos - 0.08, room_label[:12], ha='center', va='center',
                         fontsize=5, color='white')

            # 物体节点 (top-5 唯一标签)
            label_counts: Dict[str, int] = {}
            for obj in rm.objects:
                lbl = getattr(obj, 'label', '?')
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

            top_labels = sorted(label_counts.items(), key=lambda x: -x[1])[:5]
            obj_y_start = 5.0
            obj_spacing = 0.40

            for j, (lbl, cnt) in enumerate(top_labels):
                oy = obj_y_start - j * obj_spacing
                ax_tree.plot([rx, rx], [room_y_pos - 0.20, oy + 0.1],
                            color=(*color, 0.3), linewidth=0.6, zorder=0)
                obj_box = FancyBboxPatch((rx - 0.50, oy - 0.10), 1.0, 0.20,
                                          boxstyle="round,pad=0.02",
                                          facecolor=(*color, 0.15),
                                          edgecolor=(*color, 0.6), linewidth=0.6)
                ax_tree.add_patch(obj_box)
                ax_tree.text(rx, oy, f'{lbl} ×{cnt}', ha='center', va='center',
                             fontsize=5, color='black')

            remaining = len(rm.objects) - sum(c for _, c in top_labels)
            if remaining > 0:
                oy = obj_y_start - len(top_labels) * obj_spacing
                ax_tree.text(rx, oy, f'... +{remaining} more', ha='center', va='center',
                             fontsize=4.5, color='gray', style='italic')

    # 图例 (房间颜色)
    legend_items = []
    color_idx = 0
    for fl in floors_history:
        for rm in fl.rooms:
            c = room_colors_rgb[color_idx % len(room_colors_rgb)]
            color_idx += 1
            room_label = rm.room_name.get("en", rm.room_id) if rm.room_name else rm.room_id
            legend_items.append(mpatches.Patch(
                color=c, label=f'{fl.floor_id}/{rm.room_id}: {room_label}'))
    if legend_items and len(legend_items) <= 15:
        ax_tree.legend(handles=legend_items, loc='lower right', fontsize=5,
                       framealpha=0.8)

    out_path = os.path.join(output_dir, f"{scene_name}_hierarchical_graph.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[已保存] {out_path}")


def _generate_3d_scene_graph_gif(
    occ_grid: OccupancyGrid,
    rooms: List[Dict[str, Any]],
    floors_history: List,
    visited_positions: List,
    output_dir: str,
    scene_name: str,
    n_frames: int = 60,
    fps: int = 10,
):
    """生成 HOV-SG 风格的 3D 分层场景图 GIF (PyVista 渲染, 支持多楼层).

    展示:
      - 多个楼层沿 Y 轴分层堆叠
      - 每个楼层: 地面按房间着色 + 物体球体 + Floor/Room 标签
      - Scene → Floor → Room → Object 层级连线
      - 摄像机绕 Z 轴旋转一圈
    """
    import pyvista as pv
    from PIL import Image as PILImage

    ROOM_COLORS = [
        (0.24, 0.48, 0.89), (0.89, 0.26, 0.20),
        (0.96, 0.65, 0.14), (0.18, 0.74, 0.43),
        (0.67, 0.28, 0.74), (0.36, 0.79, 0.89),
        (0.95, 0.45, 0.60), (0.55, 0.63, 0.32),
    ]
    FLOOR_COLORS = [
        (0.26, 0.52, 0.96), (0.96, 0.40, 0.26),
        (0.20, 0.70, 0.40), (0.70, 0.33, 0.80),
    ]

    if not floors_history:
        print("[3D GIF] 无楼层数据, 跳过")
        return

    plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
    plotter.set_background("white")

    # 计算场景中心 (XZ) — 从所有物体
    all_positions = []
    for fl in floors_history:
        for rm in fl.rooms:
            for obj in rm.objects:
                if obj.pos_3d and len(obj.pos_3d) >= 3:
                    all_positions.append([obj.pos_3d[0], obj.pos_3d[2]])
    if not all_positions:
        print("[3D GIF] 无物体位置数据, 跳过")
        return
    all_positions = np.array(all_positions)
    scene_center = all_positions.mean(axis=0)
    scene_radius = max(np.linalg.norm(all_positions - scene_center, axis=1).max(), 5.0) + 3.0

    n_floors = len(floors_history)
    floor_gap = 6.0  # 楼层间高度间距 (可视化用)

    # Scene 根节点 (最顶部)
    scene_node_y = n_floors * floor_gap + 2.0
    scene_sphere = pv.Sphere(radius=0.7, center=(scene_center[0], scene_node_y, scene_center[1]))
    plotter.add_mesh(scene_sphere, color=(0.2, 0.2, 0.2), smooth_shading=True)
    plotter.add_point_labels(
        np.array([[scene_center[0], scene_node_y + 1.0, scene_center[1]]]),
        [f"Scene ({n_floors}F)"], font_size=14, bold=True, text_color="black",
        shape=None, render_points_as_spheres=False, always_visible=True,
    )

    global_room_idx = 0
    for fi, fl in enumerate(floors_history):
        fc = FLOOR_COLORS[fi % len(FLOOR_COLORS)]
        base_y = fi * floor_gap  # 楼层基准高度
        floor_node_y = base_y + 4.5
        room_node_y = base_y + 2.5
        object_y = base_y + 3.8
        floor_ground_y = base_y + 0.0

        # Scene → Floor 连线
        line = pv.Line(
            (scene_center[0], scene_node_y, scene_center[1]),
            (scene_center[0], floor_node_y, scene_center[1]),
        )
        plotter.add_mesh(line, color="gray", line_width=2)

        # Floor 节点
        floor_sphere = pv.Sphere(
            radius=0.5, center=(scene_center[0], floor_node_y, scene_center[1]))
        plotter.add_mesh(floor_sphere, color=fc, smooth_shading=True)
        zr = fl.z_range or {}
        fl_n_objs = sum(len(rm.objects) for rm in fl.rooms)
        plotter.add_point_labels(
            np.array([[scene_center[0], floor_node_y + 0.7, scene_center[1]]]),
            [f"{fl.floor_id} ({fl_n_objs} objs)"],
            font_size=12, bold=True, text_color="black",
            shape=None, render_points_as_spheres=False, always_visible=True,
        )

        # 构建房间 → 物体映射
        for ri, rm in enumerate(fl.rooms):
            color = ROOM_COLORS[global_room_idx % len(ROOM_COLORS)]
            global_room_idx += 1

            # 计算房间中心 (从物体位置)
            room_obj_positions = []
            for obj in rm.objects:
                if obj.pos_3d and len(obj.pos_3d) >= 3:
                    room_obj_positions.append([obj.pos_3d[0], obj.pos_3d[2]])
            if not room_obj_positions:
                continue
            room_center = np.mean(room_obj_positions, axis=0)
            cx, cz = float(room_center[0]), float(room_center[1])

            # 房间地面盘
            area = len(rm.objects) * 0.5
            radius = max(np.sqrt(area / np.pi), 0.5)
            disc = pv.Disc(center=(cx, floor_ground_y, cz), normal=(0, 1, 0),
                           inner=0, outer=min(radius, 3.0), r_res=1, c_res=32)
            plotter.add_mesh(disc, color=color, opacity=0.5)

            # 房间节点
            room_sphere = pv.Sphere(radius=0.25, center=(cx, room_node_y, cz))
            plotter.add_mesh(room_sphere, color=color, smooth_shading=True)

            room_label = rm.room_name.get("en", rm.room_id) if rm.room_name else rm.room_id
            plotter.add_point_labels(
                np.array([[cx, room_node_y + 0.4, cz]]),
                [f"{rm.room_id}\n{room_label[:10]}"],
                font_size=8, text_color="black",
                shape=None, render_points_as_spheres=False, always_visible=True,
            )

            # Floor → Room 连线
            line = pv.Line(
                (scene_center[0], floor_node_y, scene_center[1]),
                (cx, room_node_y, cz),
            )
            plotter.add_mesh(line, color="gray", line_width=1.5)

            # 物体节点
            n_objs = len(rm.objects)
            for oi, obj in enumerate(rm.objects[:20]):
                if not obj.pos_3d or len(obj.pos_3d) < 3:
                    continue
                angle = 2 * np.pi * oi / min(n_objs, 20)
                spread = min(radius * 0.5, 1.5)
                ox = cx + spread * np.cos(angle)
                oz = cz + spread * np.sin(angle)

                obj_sphere = pv.Sphere(radius=0.10, center=(ox, object_y, oz))
                plotter.add_mesh(obj_sphere, color=color, opacity=0.7)

                obj_line = pv.Line((cx, room_node_y, cz), (ox, object_y, oz))
                plotter.add_mesh(obj_line, color=color, line_width=0.8, opacity=0.3)

    # 轨迹线 (在机器人所在楼层的地面)
    if len(visited_positions) > 1:
        # 机器人所在楼层
        agent_y_median = float(np.median([p[1] for p in visited_positions]))
        traj_floor_idx = 0
        for fi, fl in enumerate(floors_history):
            zr = fl.z_range or {}
            if zr.get("z_min", -999) <= agent_y_median <= zr.get("z_max", 999):
                traj_floor_idx = fi
                break
        traj_base_y = traj_floor_idx * floor_gap + 0.05
        traj_pts = np.array([[p[0], traj_base_y, p[2]] for p in visited_positions])
        traj_line = pv.Spline(traj_pts, n_points=min(len(traj_pts) * 3, 500))
        plotter.add_mesh(traj_line, color="orange", line_width=3, opacity=0.6)

    # 摄像机轨道渲染
    frames = []
    total_height = n_floors * floor_gap + 3.0
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        cam_dist = scene_radius * 2.5
        cam_x = scene_center[0] + cam_dist * np.cos(angle)
        cam_z = scene_center[1] + cam_dist * np.sin(angle)
        cam_y = total_height * 0.8

        plotter.camera.position = (cam_x, cam_y, cam_z)
        plotter.camera.focal_point = (scene_center[0], total_height * 0.3, scene_center[1])
        plotter.camera.up = (0, 1, 0)

        img = plotter.screenshot(return_img=True)
        frames.append(PILImage.fromarray(img))

    plotter.close()

    if frames:
        out_path = os.path.join(output_dir, f"{scene_name}_scene_graph_3d.gif")
        frames[0].save(
            out_path, save_all=True, append_images=frames[1:],
            duration=int(1000 / fps), loop=0,
        )
        print(f"[已保存] {out_path} ({len(frames)} frames, {n_frames / fps:.1f}s)")


def _generate_exploration_gif(output_dir: str, rgb_dir: str, fps: int = 4):
    """把探索过程的栅格快照合成 GIF 动画."""
    import glob

    grid_frames = sorted(glob.glob(os.path.join(rgb_dir, "grid_*.jpg")))
    if len(grid_frames) < 2:
        print(f"[GIF] 栅格帧不足 ({len(grid_frames)}), 跳过")
        return

    # 使用 cv2 + imageio 生成 GIF
    try:
        from PIL import Image as PILImage
    except ImportError:
        print("[GIF] PIL 不可用, 跳过 GIF 生成")
        return

    pil_frames = []
    for fp in grid_frames:
        img = cv2.imread(fp)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_frames.append(PILImage.fromarray(img_rgb))

    if len(pil_frames) < 2:
        return

    out_path = os.path.join(output_dir, "exploration_progress.gif")
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"[已保存] {out_path} ({len(pil_frames)} frames)")


# ---------------------------------------------------------------------------
# point-in-polygon 房间归属
# ---------------------------------------------------------------------------

def assign_room_id_by_polygon(
    pos_3d: List[float],
    room_polygons: List[Dict[str, Any]],
) -> Optional[str]:
    """根据 pos_3d 做 point-in-polygon 判定房间归属.

    Args:
        pos_3d: [x, y, z] 世界坐标
        room_polygons: 来自深度探索的房间多边形列表
            [{"room_id": "R0", "polygon": [{"x":..., "y":...}, ...]}, ...]

    Returns:
        匹配的 room_id, 若不在任何房间内则返回 None
    """
    from shapely.geometry import Point, Polygon

    if not pos_3d or len(pos_3d) < 3:
        return None

    query_point = Point(pos_3d[0], pos_3d[2])  # x-z 平面

    best_room = None
    min_dist = float('inf')

    for room in room_polygons:
        poly_pts = room.get("polygon", [])
        if not poly_pts:
            continue
        coords = [(p["x"], p["y"]) for p in poly_pts]
        poly = Polygon(coords)
        if poly.is_empty:
            continue

        if poly.contains(query_point):
            return room["room_id"]

        # 若不在任何房间内, 记录最近的
        dist = poly.exterior.distance(query_point)
        if dist < min_dist:
            min_dist = dist
            best_room = room["room_id"]

    # 如果距离最近边界 < 0.5m, 也归入该房间 (容差)
    if min_dist < 0.5 and best_room is not None:
        return best_room

    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="深度探索模式 (Stage 1) — 全覆盖建图, 无目标")
    parser.add_argument("--scene-dir", required=True, help="HM3D 场景目录")
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG,
                        help="scene_dataset_config.json")
    parser.add_argument("--output-dir", default="deep_exploration_output")
    parser.add_argument("--clip-mode", default="full", choices=["mask_only", "full"])
    parser.add_argument("--max-steps", type=int, default=200,
                        help="最大行走步数 (非路径点数)")
    parser.add_argument("--step-size", type=float, default=1.0,
                        help="路径采样步长 (米)")
    parser.add_argument("--n-views", type=int, default=4,
                        help="每步环视视角数 (4=90°)")
    parser.add_argument("--no-navmesh", action="store_true",
                        help="禁用navmesh, 纯深度+A*导航 (更难)")
    parser.add_argument("--coverage-target", type=float, default=0.95,
                        help="覆盖率目标 (0-1)")
    parser.add_argument("--frontier-min-area", type=float, default=0.3,
                        help="最小frontier面积 (m²)")
    parser.add_argument("--live-viz", action="store_true", default=True,
                        help="启用实时可视化仪表盘+MP4录制 (默认启用, 用 --no-live-viz 关闭)")
    parser.add_argument("--no-live-viz", dest="live_viz", action="store_false",
                        help="关闭实时可视化")
    args = parser.parse_args()

    result = build_complete_object_map(
        scene_dir=args.scene_dir,
        dataset_config=args.dataset_config,
        clip_mode=args.clip_mode,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        step_size=args.step_size,
        n_views=args.n_views,
        use_navmesh=not args.no_navmesh,
        coverage_target=args.coverage_target,
        frontier_min_area_m2=args.frontier_min_area,
        live_viz=args.live_viz,
    )

    print(f"\n深度探索完成! 统计: {json.dumps(result['stats'], indent=2)}")


if __name__ == "__main__":
    main()
