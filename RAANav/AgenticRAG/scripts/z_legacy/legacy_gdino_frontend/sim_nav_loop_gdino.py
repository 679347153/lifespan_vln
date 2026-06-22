"""闭环导航: Agent 在 Habitat 中沿路径行走、每步360°环视观测、增量更新地图、GMM引导导航.

核心循环:
  1. GMM/Frontier 决策 → 选择下一个宏观目标点
  2. 计算 shortest_path → 沿路径逐步行走 (每步 ~0.5m)
  3. 每个路径点: 360°环视 → GT语义传感器获取物体 → 增量合并地图
  4. 虚拟时钟推进
  5. 到达目标后重新查询 GMM → 选下一个目标
  6. 如果沿途发现目标物体 → 停止

与之前"传送版"的本质区别:
  - 旧: 直接传送到目标点, 只在目标点观测一次, 路径上的信息全部丢失
  - 新: 沿 shortest_path 逐步行走, 每步 360° 环视观测, 连续更新地图
  - 这意味着 agent 在去往高概率区域的**途中**就能发现大量物体, 地图构建更完整

用法:
  cd /home/adminer/agentRAG/AgenticRAG
  conda run -n agentrag python scripts/sim_nav_loop.py \\
      --scene-dir /home/adminer/agentRAG/experiment_data/hm3d/val/00824-Dd4bFSTQ8gi \\
      --target chair \\
      --max-steps 50
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

_PROJ_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT = _PROJ_ROOT.parents[1]
for _path in [str(_PROJ_ROOT), str(_REPO_ROOT)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import habitat_sim

from semantic_map import Floor, Room, Object
from semantic_map_Create.virtual_clock import VirtualClock, set_global_clock, clock_from_config
from semantic_map_Create.scene_extract import make_simulator
from semantic_map_Create.occupancy_grid import OccupancyGrid, CameraIntrinsics
from semantic_map_Create.astar_planner import GridAStarPlanner

# nav_core 模块 (从本文件提取的可复用组件)
from scripts.nav_core.habitat_agent import HabitatAgent
from scripts.nav_core.nav_strategy import NegativeGaussianField, frontier_exploration_step
from scripts.nav_core.perception import (
    STRUCTURAL_CATEGORIES,
    objects_from_observation,
    objects_from_detection,
    compute_clip_embeddings_for_detections,
    dedup_intra_frame,
    assign_room_ids as _assign_room_ids,
    build_floors_now,
    set_object_crop_dir,
)
from scripts.nav_core.visualization import (
    visualize_trajectory_on_occ_grid as _visualize_trajectory_on_occ_grid,
    visualize_trajectory as _visualize_trajectory,
)

CONFIG_PATH = Path("config/map.yaml")
DEFAULT_SCENE_DIR = "/home/adminer/agentRAG/experiment_data/hm3d/val/00824-Dd4bFSTQ8gi"
DEFAULT_DATASET_CONFIG = "/home/adminer/agentRAG/experiment_data/hm3d/hm3d_val_scene_dataset_config.json"
DEFAULT_OUTPUT_DIR = "RAG_Graph/scene_build"


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def _load_yaml(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    import yaml
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_detect_labels(cfg: Dict) -> List[str]:
    """从 stability_priors.yaml 加载检测标签集; 若缺失则使用硬编码兜底."""
    sp_cfg = cfg.get("stability_priors", {})
    config_path = sp_cfg.get("config_path")
    labels = None
    if config_path:
        import yaml
        sp_path = Path(config_path)
        if not sp_path.is_absolute():
            sp_path = Path("config") / sp_path.name
        if sp_path.exists():
            with open(sp_path, "r", encoding="utf-8") as f:
                sp_data = yaml.safe_load(f) or {}
            labels = sp_data.get("detect_labels")
    if labels:
        return [lbl.lower().strip() for lbl in labels if isinstance(lbl, str)]
    # 兜底: 硬编码基础列表
    return [
        "chair", "dining chair", "table", "dining table", "sofa", "couch",
        "bed", "desk", "shelf", "bookshelf", "cabinet", "kitchen cabinet",
        "dresser", "wardrobe", "tv", "monitor", "lamp", "ceiling lamp",
        "plant", "picture", "painting", "mirror", "clock", "vase",
        "pillow", "cushion", "rug", "carpet", "mat", "curtain", "toilet",
        "sink", "bathtub", "shower", "refrigerator", "oven", "microwave",
        "stool", "counter", "towel", "bottle", "cup", "bowl", "basket",
        "box", "bag", "shoe", "book", "toy", "clothes", "decoration",
        "statue", "rack", "fireplace", "washer", "dryer", "bin",
        "toaster", "coffee machine", "speaker", "broom", "bucket", "hanger",
    ]

def run_sim_nav_loop(
    scene_dir: str,
    dataset_config: str,
    target: str,
    output_dir: str,
    max_steps: int = 50,
    query_interval: int = 5,
    config_path: Path = CONFIG_PATH,
    step_size: float = 0.5,
    n_views: int = 4,
    use_gt_semantic: bool = False,
    live_viz: bool = False,
    use_navmesh: bool = True,
) -> Dict[str, Any]:
    """闭环导航主循环 — 沿路径逐步行走, 每步 360° 环视观测.

    架构: 两层循环
      外层: macro_step (选择目标 → 计算路径)
      内层: micro_step (沿路径每 0.5m 行走一步 → 360° 观测 → 更新地图)

    Args:
        scene_dir: HM3D 场景目录
        dataset_config: scene_dataset_config.json
        target: 导航查询目标 (如 "chair")
        output_dir: 输出目录
        max_steps: 最大 micro 步数 (行走步)
        query_interval: 每隔多少 micro 步触发一次 GMM 查询
        config_path: map.yaml 路径
        step_size: 路径采样间距 (米)
        n_views: 每步环视视角数 (4=每90°)
        use_gt_semantic: True=使用 GT 语义传感器, False=使用视觉模型 (GroundingDINO+MobileSAM)
        live_viz: True=开启实时可视化窗口
        use_navmesh: True=使用 GT navmesh (对照实验), False=纯深度构建栅格
    """
    cfg = _load_yaml(config_path)
    os.makedirs(output_dir, exist_ok=True)
    rgb_dir = os.path.join(output_dir, "rgb_frames")
    os.makedirs(rgb_dir, exist_ok=True)
    set_object_crop_dir(os.path.join(output_dir, "object_crops"))

    # 加载预建的占据栅格 + 语义地图
    map_cfg = cfg.get("map_config") or {}
    grid_dir = map_cfg.get("occupancy_grid_dir", "RAG_Graph/scene_build/occupancy_grid")
    prebuilt_map = map_cfg.get("map_merged_json", "")

    # 初始化虚拟时钟
    clock = clock_from_config(cfg) or VirtualClock(step_hours=1.0)
    set_global_clock(clock)

    print(f"{'='*60}")
    perception_mode = "GT semantic" if use_gt_semantic else "GroundingDINO + MobileSAM"
    print(f"  闭环导航 (步进式, {n_views}视角环视)")
    print(f"  场景: {os.path.basename(scene_dir)}")
    print(f"  目标: \"{target}\"")
    print(f"  感知模式: {perception_mode}")
    nav_mode_str = "GT navmesh" if use_navmesh else "depth-only (A*)"
    print(f"  导航后端: {nav_mode_str}")
    print(f"  最大步数: {max_steps}, 查询间隔: {query_interval}")
    print(f"  步长: {step_size}m, 环视: {n_views}×{360//n_views}°")
    print(f"  虚拟时钟: step={clock.step_hours}h")
    print(f"{'='*60}\n")

    # 加载视觉检测器 (非 GT 模式)
    detector = None
    if not use_gt_semantic:
        from semantic_map_Create.perception import get_detector
        detector = get_detector(device="cuda")

    # 构建检测文本提示: 目标 + 配置驱动的室内物体标签
    INDOOR_OBJECTS = _load_detect_labels(cfg)
    target_lower = target.lower()
    # 确保目标在首位
    detect_labels = [target_lower]
    for lbl in INDOOR_OBJECTS:
        if lbl != target_lower:
            detect_labels.append(lbl)
    detect_text_prompt = " . ".join(detect_labels)
    detect_labels_set = set(detect_labels)  # 白名单集合, 过滤 BERT 碎片
    print(f"[检测标签] {len(detect_labels)} 类 (含目标 '{target_lower}')")

    # 打开 simulator
    scene_name = os.path.basename(scene_dir)
    parts = scene_name.split("-", 1)
    scene_stem = parts[1] if len(parts) > 1 else scene_name
    basis_glb = os.path.join(scene_dir, f"{scene_stem}.basis.glb")
    sim = make_simulator(basis_glb, dataset_config, enable_semantic=True, enable_depth=True)
    viz = None

    try:
        agent = HabitatAgent(sim, use_navmesh=use_navmesh)

        # 加载预建语义地图作为 history
        if prebuilt_map and Path(prebuilt_map).exists():
            print(f"[加载预建地图] {prebuilt_map}")
            with open(prebuilt_map, "r", encoding="utf-8") as f:
                map_data = json.load(f)
            floors_history = [Floor.from_dict(fd) for fd in map_data]
            all_obj_ids = set()
            for fl in floors_history:
                for rm in fl.rooms:
                    for obj in rm.objects:
                        all_obj_ids.add(obj.obj_id)
            print(f"  → {sum(len(rm.objects) for fl in floors_history for rm in fl.rooms)} 已知物体")
        else:
            floors_history = []
            all_obj_ids = set()

        prebuilt_obj_ids = set(all_obj_ids)  # 记录预建地图中的物体, 用于区分新学习物体

        # NOTE: 旧版从 grid_dir 加载离线栅格(可能来自不同场景), 已移除。
        # 所有栅格操作统一使用 occ_grid (活跃栅格, 来自当前场景 navmesh/深度).

        # 初始化 agent 位置
        start_pos = agent.get_random_navigable_point()
        agent.set_position(start_pos)
        print(f"[起始位置] {[round(x, 2) for x in start_pos]}")

        # 跟踪状态
        visited_positions = [start_pos]
        prob_field_base = None
        neg_field = NegativeGaussianField()
        target_watch_list: List[Dict] = []
        step_log = []
        target_found = False
        target_found_step = -1
        total_observed_unique = set()
        micro_step = 0  # 全局微步计数器
        gmm_query_count = 0  # GMM 查询计数 (用于 Agent 通道调度)

        # --- §4.2 Agent 通道 (LLM 调度) ---
        agent_rscore = None
        agent_cfg = (cfg.get('agent_Rscore') or {})
        agent_nav_enabled = agent_cfg.get('nav_enabled', False)
        agent_fusion_k = float(agent_cfg.get('nav_fusion_k', 0.35))
        agent_fusion_w_min = float(agent_cfg.get('nav_fusion_w_min', 0.15))
        agent_query_interval = int(agent_cfg.get('nav_query_interval', 2))
        agent_last_result = None  # 缓存最近一次 Agent LLM 结果
        if agent_nav_enabled:
            try:
                from GMM_map_Create.agent_Rscore import RScoreAgent
                agent_rscore = RScoreAgent(config_path=config_path)
                print(f"[Agent 通道] 已启用 (model={agent_rscore.model}, "
                      f"k={agent_fusion_k}, w_min={agent_fusion_w_min}, "
                      f"LLM间隔={agent_query_interval})")
            except Exception as e:
                print(f"[Agent 通道] 初始化失败: {e}, 将禁用")
                agent_nav_enabled = False

        # GT 目标物体位置 (用于可视化评估, 不参与导航决策)
        gt_target_positions = agent.get_gt_target_positions(target)
        if gt_target_positions:
            print(f"[GT 目标] 找到 {len(gt_target_positions)} 个 \"{target}\" 实例 (仅用于可视化)")
        else:
            print(f"[GT 目标] 场景中未标注 \"{target}\"")

        # --- 深度传感器 + 占据栅格 ---
        cam_intrinsics = CameraIntrinsics(hfov_deg=90.0, height=480, width=640)
        if use_navmesh:
            # 从 navmesh 初始化全局栅格 (作为 persistent 层基线)
            occ_grid = OccupancyGrid.from_navmesh_fast(
                sim, resolution=0.05, agent_radius=0.18, num_samples=50000,
            )
            occ_grid.init_persistent_from_navmesh()
        else:
            # 纯深度模式: 从 agent 位置初始化空白栅格, 由深度帧逐步填充
            occ_grid = OccupancyGrid.from_agent_position(
                agent_pos=np.array(start_pos),
                resolution=0.05,
                agent_radius=0.18,
                initial_extent=15.0,
            )
        agent.set_occ_grid(occ_grid)
        astar_planner = GridAStarPlanner(
            occ_grid, obstacle_weight=5.0, obstacle_decay_cells=6,
            use_navmesh_grid=use_navmesh,
            unknown_cost=0.3 if not use_navmesh else 0.0,
        )
        print(f"[占据栅格] {occ_grid.shape}, free={np.sum(occ_grid.grid == 1)}, "
              f"occ={np.sum(occ_grid.grid == 2)}")

        # --- 实时可视化 ---
        viz = None
        if live_viz:
            from scripts.live_visualizer import LiveVisualizer
            viz_dir = os.path.join(output_dir, "live_viz")
            viz = LiveVisualizer(
                panel_size=360, wait_ms=1, save_dir=viz_dir, record_video=True,
            )
            print(f"[实时可视化] 已启用, 录制: {viz_dir}/exploration.mp4")

        # ---------------------------------------------------------------
        # 辅助: 在当前位置执行一次完整的 "观测 + 地图更新" 周期
        # ---------------------------------------------------------------
        def observe_and_update(step_idx: int, save_rgb: bool = True) -> Tuple[List[Dict], bool]:
            """360° 环视观测 → 深度→栅格更新 → 构建 floors_now → 增量合并地图.

            GT 模式: panoramic_observe → GT semantic → objects_from_observation
            视觉模式: panoramic_observe → RGB + depth → 检测器 → objects_from_detection
            (单次环视同时获取 RGB + depth, 避免双重环视)

            Returns:
                (visible_objects, target_seen_this_step)
            """
            nonlocal floors_history, all_obj_ids, total_observed_unique

            pos = agent.get_position()
            heading_start = agent.get_heading_deg()

            # 单次 360° 环视: 同时获取 RGB + depth (+ semantic if GT mode)
            all_obs = agent.panoramic_observe(n_views)

            # --- 深度 → 占据栅格更新 ---
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
            if use_gt_semantic:
                # GT 模式: 原有语义传感器方式
                visible = agent.get_visible_objects(panoramic=True, n_views=n_views)
            else:
                # 视觉模式: 对每个视角运行 GroundingDINO + MobileSAM + 深度反投影
                visible = []
                heading = heading_start
                _vi = 0  # 视角索引 (0=正前方)
                # CLIP 编码配置
                clip_cfg = cfg.get("clip_visual", {})
                remote_clip_enabled = str(
                    os.environ.get("REMOTE_VISION_RETURN_CLIP")
                    or os.environ.get("RAANAV_REMOTE_CLIP_EMBEDDING")
                    or ""
                ).strip().lower() in {"1", "true", "yes", "y", "on"}
                clip_enabled = bool(clip_cfg.get("enabled", True)) and str(
                    os.environ.get("RAANAV_DISABLE_CLIP", "")
                ).strip().lower() not in {"1", "true", "yes", "y", "on"} and not remote_clip_enabled
                clip_encode_mode = clip_cfg.get("mode", "mask_only")
                clip_masked_weight = float(clip_cfg.get("masked_weight", 0.75))
                clip_bbox_padding = int(clip_cfg.get("bbox_padding", 20))
                for obs in all_obs:
                    rgb = obs.get("color")
                    depth = obs.get("depth")
                    if rgb is None:
                        heading += 360.0 / n_views
                        continue
                    rgb_np = rgb[:, :, :3].copy()  # RGBA → RGB
                    if depth is not None:
                        dets = detector.detect_with_depth(
                            rgb_np, depth, detect_text_prompt,
                            cam_intrinsics, np.array(pos), heading,
                            max_depth=5.0,
                            box_threshold=0.40, text_threshold=0.35,
                        )
                    else:
                        dets = detector.detect(rgb_np, detect_text_prompt,
                                              box_threshold=0.40, text_threshold=0.35)

                    # CLIP 视觉编码: 为该视角下的检测结果计算 clip_embedding
                    dets_have_clip = any(bool(d.get("clip_embedding")) for d in dets)
                    if dets and clip_enabled and not dets_have_clip:
                        _ensure_clip_model()
                        compute_clip_embeddings_for_detections(
                            rgb_np, dets,
                            _clip_model_cache["model"],
                            _clip_model_cache["processor"],
                            _clip_model_cache["device"],
                            mode=clip_encode_mode,
                            masked_weight=clip_masked_weight,
                            bbox_padding=clip_bbox_padding,
                        )
                    elif dets:
                        for d in dets:
                            d.setdefault("clip_embedding", [])

                    _img_area = rgb_np.shape[0] * rgb_np.shape[1]
                    for d in dets:
                        lbl = d.get("label", "")
                        # 过滤 BERT 分词碎片 (如 ##helf, ##er) 和单字符标签
                        if lbl.startswith("##") or len(lbl) <= 1:
                            continue
                        # 白名单: 只保留 detect_labels 中存在的标签
                        if lbl not in detect_labels_set:
                            continue
                        # 过滤结构性类别 (wall, floor, ceiling 等)
                        if lbl in STRUCTURAL_CATEGORIES:
                            continue
                        # 面积过滤: 去掉过小(<0.1%图像)和过大(>80%图像)的框
                        _bb = d.get("bbox_xyxy")
                        if _bb is not None:
                            _bw = max(_bb[2] - _bb[0], 0)
                            _bh = max(_bb[3] - _bb[1], 0)
                            _ba = _bw * _bh
                            if _ba < _img_area * 0.001 or _ba > _img_area * 0.8:
                                continue
                        d["_view_index"] = _vi  # 标记视角索引
                        d["_rgb"] = rgb_np        # 保存来源视角的 RGB
                        visible.append(d)
                    heading += 360.0 / n_views
                    _vi += 1

            visible_labels = [v["label"] for v in visible]

            # 保存 RGB 帧 (正前方视角) + 占据栅格快照
            if save_rgb:
                obs = agent.observe()
                if "color" in obs:
                    rgb = obs["color"][:, :, :3]  # RGBA → RGB
                    rgb_path = os.path.join(rgb_dir, f"step_{step_idx:04d}.jpg")
                    cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                # 保存栅格快照 (每10步)
                if step_idx % 10 == 0:
                    grid_img = occ_grid.to_image(navigable=True, show_layers=True)
                    # 标注 agent 位置
                    gc = occ_grid.world_to_grid(np.array([pos[0], pos[2]]))
                    cv2.circle(grid_img, (int(gc[0]), int(gc[1])), 4, (0, 255, 0), -1)
                    grid_path = os.path.join(rgb_dir, f"grid_{step_idx:04d}.jpg")
                    cv2.imwrite(grid_path, grid_img)

            # 检查是否看到目标 — 需要有有效的3D位置且在合理距离内 (防误检)
            _target_dets = [v for v in visible if v["label"] == target.lower()]
            target_seen = False
            for _td in _target_dets:
                _td_pos = _td.get("pos_3d")
                if _td_pos is not None:
                    # 有3D位置: 检查距离合理性 (< 5m 为可信检测)
                    _td_dist = math.sqrt(
                        (float(_td_pos[0]) - pos[0])**2 +
                        (float(_td_pos[2]) - pos[2])**2)
                    if _td_dist < 5.0:
                        target_seen = True
                        break
                elif _td.get("bbox_xyxy") is not None:
                    # 无3D但有bbox: bbox面积>图像2%视为可信
                    _bb = _td["bbox_xyxy"]
                    _bw = max(_bb[2] - _bb[0], 0)
                    _bh = max(_bb[3] - _bb[1], 0)
                    if _bw * _bh > 0.02 * 480 * 640:
                        target_seen = True
                        break

            # 构建 floors_now 并增量合并
            if visible:
                room_id = "R0"
                if use_gt_semantic:
                    for v in visible:
                        obj_ref = v.get("object_ref")
                        if obj_ref and hasattr(obj_ref, "region") and obj_ref.region is not None:
                            region = obj_ref.region
                            if hasattr(region, "id"):
                                room_id = f"R{region.id}"
                                break

                if use_gt_semantic:
                    obs_objects, all_obj_ids = objects_from_observation(
                        visible, pos, room_id, clock, all_obj_ids, n_views=n_views,
                    )
                else:
                    obs_objects, all_obj_ids = objects_from_detection(
                        visible, pos, room_id, clock, all_obj_ids,
                        step=step_idx,
                    )

                # 帧内去重: 同 label + pos_3d 距离过近 → 合并
                dedup_dist = float(cfg.get("clip_visual", {}).get("dedup_dist_threshold", 0.5))
                obs_objects = dedup_intra_frame(obs_objects, dist_threshold=dedup_dist)

                # 根据 pos_3d 最近邻分配 room_id (替代硬编码 R0)
                if not use_gt_semantic and floors_history:
                    _assign_room_ids(obs_objects, floors_history, fallback_room_id="R0")

                for o in obs_objects:
                    total_observed_unique.add(o.label)

                # 使用 floors_history 中已有的 floor_id，避免 F0≠F1 导致创建重复楼层
                # 多楼层: 根据机器人 Y 坐标选择正确的楼层
                hist_floor_id = "F0"
                if floors_history:
                    agent_y = float(agent.get_position()[1])
                    for _fl in floors_history:
                        zr = _fl.z_range or {}
                        if zr.get("z_min", -999) <= agent_y <= zr.get("z_max", 999):
                            hist_floor_id = _fl.floor_id
                            break
                    else:
                        hist_floor_id = floors_history[0].floor_id
                floors_now = build_floors_now(obs_objects, room_id=room_id, floor_id=hist_floor_id)

                if floors_history:
                    from semantic_map_Update.map_Update import run_merge
                    try:
                        floors_merged, warns = run_merge(
                            floors_now, floors_history, cfg,
                            shape_check=False,
                            allow_new_floors=True,
                            allow_new_rooms=True,
                        )
                        floors_history = floors_merged
                    except Exception as e:
                        print(f"    [合并错误] {e}")
                else:
                    floors_history = floors_now

            clock.step()
            return visible, target_seen

        # ---------------------------------------------------------------
        # 辅助: 执行一次 GMM 查询 (使用动态 CLIP 索引)
        # ---------------------------------------------------------------
        # CLIP 模型缓存 (首次查询时加载, 后续复用)
        _clip_model_cache: Dict[str, Any] = {}

        def _ensure_clip_model():
            """按需加载 CLIP 模型, 后续复用."""
            if _clip_model_cache:
                return
            import torch as _torch
            from transformers import CLIPProcessor, CLIPModel
            clip_cfg = cfg.get("clip_visual", {})
            _model_name = (
                os.environ.get("RAANAV_CLIP_MODEL_PATH")
                or os.environ.get("RAANAV_CLIP_MODEL")
                or clip_cfg.get("model_path")
                or clip_cfg.get("model_name")
                or "openai/clip-vit-base-patch32"
            )
            _local_only = str(
                os.environ.get(
                    "RAANAV_CLIP_LOCAL_FILES_ONLY",
                    "1" if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1" else "0",
                )
            ).strip().lower() in {"1", "true", "yes", "y", "on"}
            _dev = "cuda" if _torch.cuda.is_available() else "cpu"
            print(f"  [CLIP] 加载模型 {_model_name} → {_dev} ...")
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            _m = CLIPModel.from_pretrained(_model_name, local_files_only=_local_only)
            _p = CLIPProcessor.from_pretrained(_model_name, local_files_only=_local_only)
            _m.to(_dev); _m.eval()
            _clip_model_cache["model"] = _m
            _clip_model_cache["processor"] = _p
            _clip_model_cache["device"] = _dev

        def _clip_search_in_memory(query_text: str, min_score: float = 0.85) -> List[Tuple[str, float]]:
            """对 floors_history 内所有物体做 CLIP 文本相似度搜索.

            不读磁盘索引, 直接用内存中物体的 label 编码 → 查询余弦排序.
            仅搜索与 agent 同楼层的物体 (|Δy| < 1.5m).
            """
            if str(os.environ.get("RAANAV_DISABLE_CLIP", "")).strip().lower() in {"1", "true", "yes", "y", "on"}:
                agent_y = float(agent.get_position()[1])
                floor_y_threshold = 1.5
                query = query_text.strip().lower()
                fallback_hits: List[Tuple[str, float]] = []
                for fl in floors_history:
                    for rm in fl.rooms:
                        for o in rm.objects:
                            if not (o.obj_id and o.label):
                                continue
                            if o.pos_3d and abs(float(o.pos_3d[1]) - agent_y) > floor_y_threshold:
                                continue
                            label = str(o.label).strip().lower()
                            if label == query:
                                fallback_hits.append((o.obj_id, 1.0))
                return fallback_hits

            import torch as _torch

            _ensure_clip_model()
            model = _clip_model_cache["model"]
            proc = _clip_model_cache["processor"]
            dev = _clip_model_cache["device"]

            # 楼层过滤: 仅搜索与 agent 同高度的物体
            agent_y = float(agent.get_position()[1])
            FLOOR_Y_THRESHOLD = 1.5  # 同楼层判定阈值 (m)

            # 收集所有物体 (同楼层)
            all_objs: List[Tuple[str, str]] = []  # (obj_id, label)
            _skipped_floor = 0
            for fl in floors_history:
                for rm in fl.rooms:
                    for o in rm.objects:
                        if o.obj_id and o.label:
                            # 楼层过滤: 跳过不同楼层的物体
                            if o.pos_3d and abs(float(o.pos_3d[1]) - agent_y) > FLOOR_Y_THRESHOLD:
                                _skipped_floor += 1
                                continue
                            all_objs.append((o.obj_id, o.label))
            if _skipped_floor > 0:
                print(f"  [楼层过滤] 跳过 {_skipped_floor} 个异层物体 (agent_y={agent_y:.1f})")
            if not all_objs:
                return []

            # 去重 label → 编码一次
            unique_labels = list(set(lbl for _, lbl in all_objs))
            label_vecs = {}
            for i in range(0, len(unique_labels), 32):
                batch = unique_labels[i:i+32]
                inputs = proc(text=batch, return_tensors="pt", padding=True, truncation=True).to(dev)
                with _torch.no_grad():
                    feats = model.get_text_features(**inputs)
                    feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                for j, lbl in enumerate(batch):
                    label_vecs[lbl] = feats[j].cpu().numpy()

            # 编码查询
            q_inputs = proc(text=[query_text], return_tensors="pt", padding=True, truncation=True).to(dev)
            with _torch.no_grad():
                q_feat = model.get_text_features(**q_inputs)
                q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            q_vec = q_feat[0].cpu().numpy()

            # 余弦相似度
            results = []
            for oid, lbl in all_objs:
                sim = float(np.dot(label_vecs[lbl], q_vec))
                if sim >= min_score:
                    results.append((oid, sim))
            results.sort(key=lambda x: -x[1])
            return results

        def _score_from_memory(
            target_text: str, clip_hits: List[Tuple[str, float]],
        ) -> Dict[str, Any]:
            """在 floors_history 上做 GMM 评分 (不读磁盘地图)."""
            from GMM_map_Create.GMM_map_calcualte import (
                Calculate_obj_Score, Calculate_Robj_Score,
            )
            from scripts.query_e2e import _get_pos_xy

            # 建立 obj_id → Object 索引 (同楼层)
            agent_y = float(agent.get_position()[1])
            FLOOR_Y_THRESHOLD = 1.5
            obj_index: Dict[str, Any] = {}
            for fl in floors_history:
                for rm in fl.rooms:
                    for o in rm.objects:
                        if o.pos_3d and abs(float(o.pos_3d[1]) - agent_y) > FLOOR_Y_THRESHOLD:
                            continue
                        obj_index[o.obj_id] = o

            target_ids = [oid for oid, _ in clip_hits if oid in obj_index]

            targets_self = []
            for tid in target_ids:
                obj = obj_index[tid]
                self_score = Calculate_obj_Score(
                    N=obj.N or 1,
                    stability=obj.stability or 0.5,
                    cfd=obj.cfd,
                    config_path=config_path,
                )
                targets_self.append({
                    "obj_id": tid,
                    "label": obj.label,
                    "self_score": self_score,
                    "pos_2d": obj.pos_2d if hasattr(obj, "pos_2d") else None,
                })

            related_scores: Dict[str, float] = {}
            related_info: Dict[str, Dict] = {}
            for tid in target_ids:
                t_obj = obj_index[tid]
                R_map = t_obj.R_objs if hasattr(t_obj, "R_objs") else {}
                if not isinstance(R_map, dict):
                    continue
                N_target = int(t_obj.N or 1)
                for rel_oid, rel_data in R_map.items():
                    r_obj = obj_index.get(rel_oid)
                    if not r_obj:
                        continue
                    Nr = int((rel_data or {}).get("Nr", 0) or 0)
                    Rcfd = float((rel_data or {}).get("Rcfd", 0.0) or 0.0)
                    Nr_over_N = Nr / max(1, N_target)
                    score = Calculate_Robj_Score(
                        total_N=N_target,
                        Nr_over_N=Nr_over_N,
                        Rscore=0.0,
                        Rcfd=Rcfd,
                        stability=r_obj.stability or 0.5,
                        exist_prob=r_obj.exist_prob if hasattr(r_obj, "exist_prob") else 1.0,
                        config_path=config_path,
                    )
                    if score > related_scores.get(rel_oid, 0.0):
                        related_scores[rel_oid] = score
                        related_info[rel_oid] = {
                            "obj_id": rel_oid,
                            "label": r_obj.label,
                            "score": score,
                            "pos_2d": r_obj.pos_2d if hasattr(r_obj, "pos_2d") else None,
                        }

            gmm_scores: Dict[str, float] = {}
            gmm_positions: Dict[str, List[float]] = {}
            for t in targets_self:
                gmm_scores[t["obj_id"]] = t["self_score"]
                xy = _get_pos_xy(t.get("pos_2d"))
                if xy:
                    gmm_positions[t["obj_id"]] = list(xy)
            for oid, info in related_info.items():
                if info["score"] > 0:
                    gmm_scores[oid] = info["score"]
                    xy = _get_pos_xy(info.get("pos_2d"))
                    if xy:
                        gmm_positions[oid] = list(xy)

            return {
                "targets_self": targets_self,
                "gmm_scores": gmm_scores,
                "gmm_positions": gmm_positions,
            }

        def run_gmm_query(step_idx: int):
            """运行 GMM 查询 (动态 CLIP 索引), 更新 prob_field_base 和 target_watch_list."""
            nonlocal prob_field_base, target_watch_list, gmm_query_count, agent_last_result

            if not floors_history:
                return

            gmm_query_count += 1
            print(f"\n  [GMM 查询] target=\"{target}\" (step {step_idx}), "
                  f"动态CLIP索引, 查询#{gmm_query_count}")
            try:
                # 保存 map_live.json (用于调试/回放)
                tmp_map = os.path.join(output_dir, "map_live.json")
                map_data = [fl.to_dict() for fl in floors_history]
                with open(tmp_map, "w", encoding="utf-8") as f:
                    json.dump(map_data, f, ensure_ascii=False, indent=2)

                # 动态 CLIP 搜索 (从 floors_history 内存)
                clip_hits = _clip_search_in_memory(target, min_score=0.85)
                n_map_objs = sum(len(rm.objects) for fl in floors_history for rm in fl.rooms)
                print(f"  [CLIP] {len(clip_hits)} 命中 / {n_map_objs} 地图物体")
                if clip_hits:
                    for oid, sc in clip_hits[:5]:
                        print(f"    {oid}: {sc:.4f}")

                    # 在 floors_history 上评分
                    score_data = _score_from_memory(target, clip_hits)

                    # --- §4.2 Agent 通道融合 ---
                    if agent_nav_enabled and agent_rscore and score_data["gmm_scores"]:
                        import math as _math
                        # 动态融合权重: w_agent(n) = max(w_min, exp(-k * n))
                        n_obs = gmm_query_count
                        w_agent = max(agent_fusion_w_min,
                                      _math.exp(-agent_fusion_k * n_obs))
                        w_rag = 1.0 - w_agent

                        # 按间隔调用 LLM (昂贵), 否则复用缓存
                        if gmm_query_count % agent_query_interval == 1 or agent_last_result is None:
                            try:
                                map_data_for_agent = [fl.to_dict() for fl in floors_history]
                                clip_ids = [oid for oid, _ in clip_hits]
                                agent_result = agent_rscore.score_for_navigation(
                                    target, map_data_for_agent, clip_ids,
                                )
                                agent_last_result = agent_result
                                room_priors = agent_result.get("room_priors", {})
                                if room_priors:
                                    top_rooms = sorted(room_priors.items(),
                                                       key=lambda x: -x[1])[:3]
                                    print(f"  [Agent] 房间先验: "
                                          f"{', '.join(f'{r}={p:.2f}' for r, p in top_rooms)}")
                                n_scored = len(agent_result.get("obj_scores", {}))
                                print(f"  [Agent] LLM调用完成, {n_scored} 物体评分, "
                                      f"w_rag={w_rag:.3f}, w_agent={w_agent:.3f}")
                            except Exception as e:
                                print(f"  [Agent] LLM 调用失败: {e}")
                                agent_last_result = None

                        # 融合: S_fused(i) = w_rag * S_rag(i) + w_agent * S_agent(i)
                        if agent_last_result and agent_last_result.get("obj_scores"):
                            agent_obj_scores = agent_last_result["obj_scores"]
                            # Agent 分数需归一化到与 RAG 分数同量级
                            rag_max = max(score_data["gmm_scores"].values()) if score_data["gmm_scores"] else 1.0
                            for oid in score_data["gmm_scores"]:
                                s_rag = score_data["gmm_scores"][oid]
                                s_agent = agent_obj_scores.get(oid, 0.0) * rag_max
                                score_data["gmm_scores"][oid] = (
                                    w_rag * s_rag + w_agent * s_agent
                                )
                            # Agent 可能推荐不在 RAG 结果中的物体 (间接关联)
                            from scripts.query_e2e import _get_pos_xy as _gxy
                            for oid, s_a in agent_obj_scores.items():
                                if oid not in score_data["gmm_scores"] and s_a > 0.3:
                                    # 从 floors_history 查找位置
                                    for fl in floors_history:
                                        for rm in fl.rooms:
                                            for o in rm.objects:
                                                if o.obj_id == oid:
                                                    xy = _gxy(o.pos_2d if hasattr(o, "pos_2d") else None)
                                                    if xy:
                                                        score_data["gmm_scores"][oid] = w_agent * s_a * rag_max
                                                        score_data["gmm_positions"][oid] = list(xy)
                            print(f"  [融合] 融合后 {len(score_data['gmm_scores'])} 物体")

                    if score_data["gmm_scores"]:
                        from scripts.query_e2e import build_probability_field
                        # 使用活动栅格数据而非从磁盘加载旧栅格
                        live_grid_meta = {
                            "resolution": occ_grid.resolution,
                            "origin_x": float(occ_grid._origin[0]),
                            "origin_z": float(occ_grid._origin[1]),
                        }
                        prob_field, _, _ = build_probability_field(
                            score_data["gmm_scores"],
                            score_data["gmm_positions"],
                            grid_dir,
                            sigma_base=1.5,
                            grid_array=occ_grid.grid,
                            grid_meta_override=live_grid_meta,
                        )

                        # --- §4.2 room_priors 空间加成 ---
                        # 将 LLM 的房间先验概率作为乘性加成施加到概率场:
                        # 属于高概率房间的物体对应的高斯核获得额外 boost
                        if (agent_last_result and agent_last_result.get("room_priors")
                                and score_data.get("gmm_positions")):
                            room_priors = agent_last_result["room_priors"]
                            # 建 obj_id → room_id 索引
                            _obj_room = {}
                            for fl in floors_history:
                                for rm in fl.rooms:
                                    for o in rm.objects:
                                        _obj_room[o.obj_id] = rm.room_id
                            # 对每个物体, 若其所在房间有 room_prior, 则加成
                            boosted = 0
                            for oid in list(score_data["gmm_scores"].keys()):
                                rid = _obj_room.get(oid, "")
                                rp = room_priors.get(rid, 0.0)
                                if rp > 0.1:
                                    # 加性 boost: score += rp * rag_max * 0.5
                                    boost = rp * (rag_max if 'rag_max' in dir() else 0.5) * 0.5
                                    score_data["gmm_scores"][oid] += boost
                                    boosted += 1
                            if boosted > 0:
                                # 重建概率场 (含 room_prior boost)
                                prob_field, _, _ = build_probability_field(
                                    score_data["gmm_scores"],
                                    score_data["gmm_positions"],
                                    grid_dir,
                                    sigma_base=1.5,
                                    grid_array=occ_grid.grid,
                                    grid_meta_override=live_grid_meta,
                                )
                                print(f"  [room_priors] {boosted} 物体获得房间先验加成")

                        prob_field_base = prob_field.copy()
                        print(f"  [概率场] max={prob_field.max():.6f}, "
                              f"n_gaussians={len(score_data['gmm_scores'])}")

                    # 构建监视列表
                    from scripts.query_e2e import _get_pos_xy
                    _stab_map = {}
                    for fl in floors_history:
                        for rm in fl.rooms:
                            for o in rm.objects:
                                _stab_map[o.obj_id] = o.stability
                    target_watch_list = []
                    for t in score_data.get("targets_self", []):
                        xy = _get_pos_xy(t.get("pos_2d"))
                        if xy and t.get("self_score", 0) > 0:
                            target_watch_list.append({
                                "obj_id": t["obj_id"],
                                "label": t["label"],
                                "wx": xy[0], "wz": xy[1],
                                "score": t["self_score"],
                                "stability": _stab_map.get(t["obj_id"], 0.7),
                            })
                    if target_watch_list:
                        print(f"  [监视列表] {len(target_watch_list)} 个目标物体")
                else:
                    print(f"  [CLIP] 无匹配 (阈值 0.85)")

                # 保存查询热图 (可选)
                q_dir = os.path.join(output_dir, f"query_step{step_idx}")
                os.makedirs(q_dir, exist_ok=True)
                if prob_field_base is not None:
                    from scripts.query_e2e import visualize_and_extract_topk
                    try:
                        _gm = {"resolution": occ_grid.resolution,
                                "origin_x": float(occ_grid._origin[0]),
                                "origin_z": float(occ_grid._origin[1]),
                                "n_gaussians": len(score_data.get("gmm_scores", {})),
                                "n_scored_objects": len(score_data.get("gmm_scores", {}))}
                        visualize_and_extract_topk(
                            prob_field_base, occ_grid.grid, _gm,
                            score_data.get("targets_self", []),
                            target, q_dir, top_k=3,
                        )
                    except Exception:
                        pass  # 热力图保存失败不影响导航
            except Exception as e:
                import traceback
                print(f"  [查询错误] {e}")
                traceback.print_exc()

        # ---------------------------------------------------------------
        # 主循环: 两层 — 外层选目标, 内层沿路径行走
        # ---------------------------------------------------------------
        print(f"\n[Phase 0] 起点处首次观测...")
        visible, target_seen = observe_and_update(micro_step)
        print(f"  起始环视观测到 {len(visible)} 个物体, {len(total_observed_unique)} 唯一标签")
        if target_seen:
            target_found = True
            target_found_step = micro_step + 1
            print(f"  ★ 起点即发现目标 \"{target}\"!")

        # Phase 0.5: 强制执行一次 GMM 查询, 确保导航 step 1 就有概率场引导
        if not target_found and floors_history:
            print(f"\n[Phase 0.5] 强制 GMM 查询 (确保 step 1 有概率场)...")
            run_gmm_query(0)
            if prob_field_base is not None:
                print(f"  ✓ 概率场已就绪, max={prob_field_base.max():.6f}")
            else:
                print(f"  ⚠ 概率场仍为空 (无 CLIP 匹配或物体为空)")

        # Phase 0.6: 房间分割 + Voronoi 导航图
        room_seg_result = None
        voronoi_nav_graph = None
        _room_seg_last_step = -1  # 上次房间分割的步数

        # --- Voronoi 导航器: 优先从预建文件加载, 否则现场构建 ---
        from semantic_map_Create.voronoi_graph import VoronoiNavigator
        from scripts.nav_core.voronoi_navigator import StuckDetector, recover_from_stuck
        voronoi_nav = VoronoiNavigator(occ_grid)
        stuck_detector = StuckDetector(window=5, min_displacement=0.5)

        # 尝试加载深度探索阶段保存的 Voronoi 导航图
        _voronoi_loaded = False
        if prebuilt_map:
            voronoi_json = os.path.join(os.path.dirname(prebuilt_map), "voronoi_nav_graph.json")
            if Path(voronoi_json).exists():
                try:
                    voronoi_nav = VoronoiNavigator.load(voronoi_json, occ_grid=occ_grid)
                    voronoi_nav_graph = voronoi_nav.graph
                    _voronoi_loaded = True
                except Exception as e:
                    print(f"  [Voronoi] 加载失败: {e}, 将现场构建")

        def _rebuild_room_and_voronoi(step: int):
            nonlocal room_seg_result, voronoi_nav_graph, _room_seg_last_step
            if np.sum(occ_grid.grid == 1) < 100:
                return  # free 区域太少, 跳过
            from semantic_map_Create.room_segmentation import (
                segment_rooms_from_occ_grid, assign_objects_to_rooms,
            )
            from semantic_map_Create.voronoi_graph import build_voronoi_graph
            room_seg_result = segment_rooms_from_occ_grid(occ_grid)
            if room_seg_result["n_rooms"] > 0:
                voronoi_nav_graph = build_voronoi_graph(
                    occ_grid, room_labels=room_seg_result["room_labels"],
                )
                voronoi_nav.set_graph(voronoi_nav_graph)
                # 为已知物体重新分配 room_id
                if floors_history:
                    all_objs = []
                    for fl in floors_history:
                        for rm in fl.rooms:
                            all_objs.extend(rm.objects)
                    assign_objects_to_rooms(
                        all_objs, room_seg_result["room_labels"], occ_grid,
                    )
            _room_seg_last_step = step
            print(f"  [房间分割] {room_seg_result['n_rooms']} 个房间, "
                  f"Voronoi 节点={voronoi_nav_graph.number_of_nodes() if voronoi_nav_graph else 0}")

        if not _voronoi_loaded and (use_navmesh or np.sum(occ_grid.grid == 1) > 500):
            try:
                _rebuild_room_and_voronoi(0)
            except Exception as e:
                print(f"  [房间分割] 初始化失败: {e}")

        viz_quit = False

        while micro_step < max_steps and not target_found and not viz_quit:
            # --- 选择下一个宏观目标 ---
            if micro_step % query_interval == 0:
                run_gmm_query(micro_step)

            # 检查 prob_field_base 与 occ_grid 形状一致性 (扩展后需重建)
            if (prob_field_base is not None
                    and prob_field_base.shape != occ_grid.grid.shape):
                print(f"  [形状修正] prob_field {prob_field_base.shape} "
                      f"!= occ_grid {occ_grid.grid.shape}, 强制重建")
                run_gmm_query(micro_step)

            # 使用活动栅格数据 (occ_grid 会动态扩展, 避免使用启动时加载的旧栅格)
            live_meta = {
                "resolution": occ_grid.resolution,
                "origin_x": float(occ_grid._origin[0]),
                "origin_z": float(occ_grid._origin[1]),
            }
            nav_target, nav_mode = frontier_exploration_step(
                agent, prob_field_base, neg_field, live_meta,
                occ_grid.grid, visited_positions, micro_step,
                occ_grid=occ_grid,
            )
            print(f"\n{'='*50}")
            print(f"  [宏目标] 模式={nav_mode}, 负核={neg_field.n_active}, 步={micro_step}/{max_steps}")
            print(f"  [目标点] {[round(x, 2) for x in nav_target]}")

            # --- 计算路径: Voronoi (coarse) → A* → navmesh (回退) ---
            pos_now = agent.get_position()
            start_3d = [float(pos_now[0]), float(pos_now[1]), float(pos_now[2])]
            waypoints = None
            path_source = ""

            # 1. Voronoi 路径 (走房间中心/门口, 适合跨房间导航)
            if voronoi_nav.has_graph():
                vpath = voronoi_nav.plan_path_3d(start_3d, nav_target, step_size=step_size)
                if vpath and len(vpath) > 0:
                    waypoints = vpath
                    path_source = "voronoi"

            # 2. A* on occupancy grid
            if waypoints is None:
                astar_planner.invalidate_cost_map()
                try:
                    waypoints_3d = astar_planner.plan_3d(
                        start_3d, nav_target, step_size=step_size,
                    )
                except Exception as e:
                    print(f"  [A* 异常] {e}")
                    waypoints_3d = None
                if waypoints_3d is not None:
                    waypoints = waypoints_3d[1:] if len(waypoints_3d) > 1 else waypoints_3d
                    path_source = "A*"

            # 3. navmesh 回退
            if waypoints is None:
                waypoints = agent.get_path_waypoints(nav_target, step_size=step_size)
                if waypoints is not None:
                    path_source = "navmesh"

            if waypoints is None:
                print(f"  [路径] 所有规划器均不可达, 随机传送")
                rp = agent.get_random_navigable_point()
                agent.navigate_to(rp)
                visited_positions.append(rp)
                micro_step += 1
                continue

            print(f"  [路径] {path_source}: {len(waypoints)} 个路径点 ({len(waypoints)*step_size:.1f}m)")

            # --- 沿路径逐步行走 ---
            for wi, wp in enumerate(waypoints):
                if micro_step >= max_steps or target_found:
                    break

                # 注: 移除了 check_collision_ahead 预检查 (v0.8.1)
                # 原因: 使用 agent 当前朝向而非 waypoint 方向, 且基于
                # short_term 层过于激进, 导致走廊中所有后续 waypoint 被跳过。
                # move_to_waypoint() 已有碰撞恢复 (±15°/±30°/±45° nudge)。

                agent.move_to_waypoint(wp)
                pos = agent.get_position()
                visited_positions.append([float(pos[0]), float(pos[1]), float(pos[2])])

                # 卡住检测
                is_stuck = stuck_detector.update([float(pos[0]), float(pos[1]), float(pos[2])])
                if is_stuck:
                    recovery_target = recover_from_stuck(
                        agent, occ_grid, visited_positions,
                        stuck_detector.stuck_count, use_navmesh=use_navmesh,
                    )
                    if recovery_target is not None:
                        agent.move_to_waypoint(recovery_target)
                        stuck_detector.reset()
                    break  # 跳出当前路径, 重新选择宏目标

                # 360° 环视 + 地图更新
                visible, target_seen = observe_and_update(micro_step, save_rgb=True)
                visible_labels = list(set(v["label"] for v in visible))

                # 打印简要信息 (每5步或发现目标时)
                if micro_step % 5 == 0 or target_seen:
                    print(f"  step {micro_step+1}: [{pos[0]:.1f},{pos[2]:.1f}] "
                          f"obs={len(visible)} labels={len(total_observed_unique)} "
                          f"wp={wi+1}/{len(waypoints)}")

                if target_seen and not target_found:
                    target_found = True
                    target_found_step = micro_step + 1
                    print(f"\n  ★★★ 沿途发现目标 \"{target}\"! (micro_step {target_found_step}) ★★★")

                # 假阴性检测
                if target_watch_list and visible:
                    visible_label_set = set(v["label"] for v in visible)
                    injected = neg_field.check_and_inject(
                        agent_pos=pos,
                        visible_labels=visible_label_set,
                        watch_list=target_watch_list,
                        current_step=micro_step,
                        sigma_base=1.5,
                    )
                    if injected:
                        print(f"    [负核] 注入 {len(injected)} 个")
                    neg_field.cleanup(micro_step)

                # 中途 GMM 重查询 (每 query_interval 步)
                if (micro_step + 1) % query_interval == 0:
                    run_gmm_query(micro_step + 1)

                # 记录
                step_log.append({
                    "step": micro_step + 1,
                    "position": [round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3)],
                    "n_visible": len(visible),
                    "visible_labels": visible_labels,
                    "target_visible": target_seen,
                    "nav_target": [round(x, 3) for x in nav_target],
                    "nav_mode": nav_mode,
                    "waypoint": f"{wi+1}/{len(waypoints)}",
                    "neg_kernels_active": neg_field.n_active,
                    "n_unique_labels": len(total_observed_unique),
                    "virtual_time": clock.now_iso(),
                })

                # --- 实时可视化更新 ---
                if viz is not None:
                    # 获取正前方 RGB 帧
                    _obs = agent.observe()
                    _rgb = _obs.get("color")
                    _n_map_objs = sum(
                        len(rm.objects) for fl in floors_history for rm in fl.rooms
                    ) if floors_history else 0
                    # 只传正前方视角 (view_index=0) 的检测给可视化, 避免其他视角的框画在正前方图上
                    _front_dets = [d for d in visible if d.get("_view_index", 0) == 0] if not use_gt_semantic else None
                    viz_continue = viz.update(
                        step=micro_step + 1,
                        rgb=_rgb,
                        detections=_front_dets,
                        occ_grid=occ_grid,
                        agent_pos=np.array(pos),
                        nav_target=nav_target,
                        prob_field=prob_field_base,
                        grid_meta={
                            "resolution": occ_grid.resolution,
                            "origin_x": float(occ_grid._origin[0]),
                            "origin_z": float(occ_grid._origin[1]),
                        },
                        neg_kernels=neg_field.n_active,
                        nav_mode=nav_mode,
                        target_name=target,
                        target_found=target_found,
                        n_objects=_n_map_objs,
                        n_unique_labels=len(total_observed_unique),
                        visited_positions=visited_positions,
                        gt_target_positions=gt_target_positions,
                    )
                    if not viz_continue:
                        print("[LiveViz] 用户退出")
                        viz_quit = True
                        micro_step += 1
                        break

                micro_step += 1

            # 到达宏目标后, 多视点调查模式 (GMM引导时)
            if not target_found and micro_step < max_steps and nav_mode == "gmm":
                print(f"  [到达] 宏目标点, 启动多视点调查...")
                # 先在当前位置环视
                visible, target_seen = observe_and_update(micro_step)
                if target_seen and not target_found:
                    target_found = True
                    target_found_step = micro_step + 1
                    print(f"  ★★★ 到达目标点后发现 \"{target}\"! ★★★")

                # 生成概率峰值附近的调查视点
                if not target_found and prob_field_base is not None:
                    from scripts.nav_core.nav_strategy import generate_investigation_viewpoints
                    inv_meta = {
                        "resolution": occ_grid.resolution,
                        "origin_x": float(occ_grid._origin[0]),
                        "origin_z": float(occ_grid._origin[1]),
                    }
                    inv_viewpoints = generate_investigation_viewpoints(
                        prob_field_base, occ_grid.grid, inv_meta,
                        agent_y=float(agent.get_position()[1]),
                        n_viewpoints=3, standoff_m=1.5,
                    )
                    if inv_viewpoints:
                        print(f"    [调查] {len(inv_viewpoints)} 个调查视点")
                    for iv_i, iv_wp in enumerate(inv_viewpoints):
                        if micro_step >= max_steps or target_found:
                            break
                        agent.move_to_waypoint(iv_wp)
                        pos = agent.get_position()
                        visited_positions.append([float(pos[0]), float(pos[1]), float(pos[2])])
                        visible, target_seen = observe_and_update(micro_step, save_rgb=True)
                        print(f"    [调查 {iv_i+1}/{len(inv_viewpoints)}] "
                              f"[{pos[0]:.1f},{pos[2]:.1f}] obs={len(visible)}")
                        if target_seen and not target_found:
                            target_found = True
                            target_found_step = micro_step + 1
                            print(f"  ★★★ 调查视点发现 \"{target}\"! ★★★")
                        micro_step += 1
            elif not target_found and micro_step < max_steps:
                # 非 GMM 模式 (frontier/random): 仍做单次环视
                print(f"  [到达] 宏目标点, 环视确认...")
                visible, target_seen = observe_and_update(micro_step)
                if target_seen and not target_found:
                    target_found = True
                    target_found_step = micro_step + 1
                    print(f"  ★★★ 到达目标点后发现 \"{target}\"! ★★★")

    finally:
        if viz is not None:
            viz.close()
        sim.close()

    # 输出总结
    result = {
        "scene": scene_name,
        "target": target,
        "max_steps": max_steps,
        "actual_steps": len(step_log),
        "step_size_m": step_size,
        "n_views": n_views,
        "target_found": target_found,
        "target_found_step": target_found_step,
        "unique_labels_observed": sorted(total_observed_unique),
        "n_unique_labels": len(total_observed_unique),
        "total_map_objects": sum(
            len(rm.objects) for fl in floors_history for rm in fl.rooms
        ) if floors_history else 0,
        "total_distance_m": round(sum(
            HabitatAgent._dist3(visited_positions[i], visited_positions[i+1])
            for i in range(len(visited_positions)-1)
        ), 2),
        "step_log": step_log,
    }

    result_path = os.path.join(output_dir, "nav_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 保存最终合并地图
    if floors_history:
        final_map_path = os.path.join(output_dir, "map_final.json")
        with open(final_map_path, "w", encoding="utf-8") as f:
            json.dump([fl.to_dict() for fl in floors_history], f, ensure_ascii=False, indent=2)

    # 保存局部 RAG 地图 (仅任务导航期间新发现的物体)
    new_obj_ids = all_obj_ids - prebuilt_obj_ids
    if new_obj_ids and floors_history:
        local_objs = []
        for fl in floors_history:
            for rm in fl.rooms:
                for obj in rm.objects:
                    if obj.obj_id in new_obj_ids:
                        local_objs.append(obj.to_dict())
        local_rag_path = os.path.join(output_dir, "map_local_rag.json")
        with open(local_rag_path, "w", encoding="utf-8") as f:
            json.dump({"n_new_objects": len(local_objs), "objects": local_objs},
                      f, ensure_ascii=False, indent=2)
        print(f"[局部RAG] 新学习 {len(local_objs)} 个物体 → {local_rag_path}")

    # 保存占据栅格
    occ_grid_dir = os.path.join(output_dir, "occupancy")
    occ_grid.save(occ_grid_dir, prefix="occupancy")

    # 保存房间分割 + Voronoi 可视化
    if room_seg_result is not None and room_seg_result["n_rooms"] > 0:
        from semantic_map_Create.room_segmentation import visualize_room_segmentation
        visualize_room_segmentation(
            room_seg_result["room_labels"],
            room_seg_result["walls_skeleton"],
            room_seg_result["room_centers"],
            occ_grid,
            save_path=os.path.join(output_dir, "room_segmentation.png"),
        )
    if voronoi_nav_graph is not None and voronoi_nav_graph.number_of_nodes() > 0:
        from semantic_map_Create.voronoi_graph import visualize_voronoi_graph
        visualize_voronoi_graph(
            voronoi_nav_graph, occ_grid,
            room_labels=room_seg_result["room_labels"] if room_seg_result else None,
            save_path=os.path.join(output_dir, "voronoi_graph.png"),
        )

    # 生成轨迹可视化 (优先使用深度构建的栅格)
    _visualize_trajectory_on_occ_grid(
        occ_grid, visited_positions, step_log, target, output_dir,
    )

    # 生成导航过程 GIF (从 live_viz 帧或栅格帧)
    try:
        import glob
        from PIL import Image as PILImage
        gif_frames_dir = os.path.join(output_dir, "live_viz")
        if not os.path.isdir(gif_frames_dir):
            gif_frames_dir = rgb_dir  # fallback 到 RGB 帧目录
        # 优先用 live_viz 帧, 否则用栅格帧
        viz_files = sorted(glob.glob(os.path.join(gif_frames_dir, "viz_*.jpg")))
        if not viz_files:
            viz_files = sorted(glob.glob(os.path.join(rgb_dir, "grid_*.jpg")))
        if len(viz_files) >= 2:
            pil_frames = []
            for fp in viz_files:
                img = cv2.imread(fp)
                if img is not None:
                    pil_frames.append(PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            if len(pil_frames) >= 2:
                gif_path = os.path.join(output_dir, "navigation_process.gif")
                pil_frames[0].save(
                    gif_path, save_all=True, append_images=pil_frames[1:],
                    duration=250, loop=0,
                )
                print(f"[已保存] {gif_path} ({len(pil_frames)} frames)")
    except Exception as e:
        print(f"[GIF] 生成失败: {e}")

    print(f"\n{'='*60}")
    print(f"  导航完成!")
    print(f"  步数: {len(step_log)}/{max_steps} (步长 {step_size}m)")
    print(f"  总行走距离: {result['total_distance_m']}m")
    print(f"  目标发现: {'是 (第 ' + str(target_found_step) + ' 步)' if target_found else '否'}")
    print(f"  观测到唯一标签: {len(total_observed_unique)}")
    print(f"  最终地图物体数: {result['total_map_objects']}")
    print(f"  RGB帧: {rgb_dir}")
    print(f"  结果: {result_path}")
    print(f"{'='*60}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AgenticRAG 闭环导航 (步进式)")
    parser.add_argument("--scene-dir", default=DEFAULT_SCENE_DIR)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--target", required=True, help="导航目标 (如 chair)")
    parser.add_argument("--output-dir", default="RAG_Graph/scene_build/nav_sim")
    parser.add_argument("--max-steps", type=int, default=50, help="最大 micro 步数")
    parser.add_argument("--query-interval", type=int, default=5, help="GMM 查询间隔 (步)")
    parser.add_argument("--step-size", type=float, default=0.5, help="路径采样步长 (米)")
    parser.add_argument("--n-views", type=int, default=4, help="每步环视视角数 (4=90°)")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--use-gt-semantic", action="store_true",
                        help="使用 GT 语义传感器 (默认使用视觉模型)")
    parser.add_argument("--live-viz", action="store_true",
                        help="开启实时可视化窗口 (OpenCV)")
    parser.add_argument("--no-navmesh", action="store_true",
                        help="禁用 GT navmesh, 使用纯深度构建栅格 + A*导航")
    parser.add_argument("--perception-backend", choices=["local", "remote"], default=None,
                        help="Perception backend: local or remote.")
    parser.add_argument("--remote-vision-base-url", default=None,
                        help="Local URL after a manual SSH tunnel, e.g. http://127.0.0.1:50220")
    parser.add_argument("--remote-vision-use-ssh-tunnel", action="store_true",
                        help="Start the SSH tunnel to the remote vision server automatically.")
    parser.add_argument("--remote-vision-ssh-host", default=None)
    parser.add_argument("--remote-vision-ssh-port", type=int, default=None)
    parser.add_argument("--remote-vision-ssh-user", default=None)
    parser.add_argument("--remote-vision-ssh-password", default=None)
    parser.add_argument("--remote-vision-local-port", type=int, default=None)
    parser.add_argument("--remote-vision-remote-port", type=int, default=None)
    parser.add_argument("--remote-clip", action="store_true",
                        help="Ask the remote vision server to compute CLIP image embeddings.")
    parser.add_argument("--remote-clip-model-path", default=None,
                        help="Server-side Hugging Face CLIP directory, e.g. /home/.../clip-vit-large-patch14")
    parser.add_argument("--remote-clip-model-name", default=None,
                        help="Server-side CLIP model id or local model name.")
    parser.add_argument("--remote-clip-online", action="store_true",
                        help="Allow the remote CLIP loader to use online Hugging Face lookups.")
    parser.add_argument("--disable-clip", action="store_true",
                        help="Skip local CLIP embeddings for detections. Useful for remote vision smoke tests.")
    parser.add_argument("--clip-model-path", default=None,
                        help="Local Hugging Face CLIP directory, e.g. /data/trans/clip-vit-large-patch14")
    parser.add_argument("--clip-model-name", default=None,
                        help="HF model id or local model name for CLIP.")
    parser.add_argument("--clip-online", action="store_true",
                        help="Allow CLIP to download from HF/mirror instead of local-only loading.")
    args = parser.parse_args()

    if args.perception_backend:
        os.environ["RAANAV_PERCEPTION_BACKEND"] = args.perception_backend
    if args.remote_vision_base_url:
        os.environ["REMOTE_VISION_BASE_URL"] = args.remote_vision_base_url
    if args.remote_vision_use_ssh_tunnel:
        os.environ["REMOTE_VISION_USE_SSH_TUNNEL"] = "1"
        os.environ.setdefault("RAANAV_PERCEPTION_BACKEND", "remote")
    if args.remote_vision_ssh_host:
        os.environ["REMOTE_VISION_SSH_HOST"] = args.remote_vision_ssh_host
    if args.remote_vision_ssh_port is not None:
        os.environ["REMOTE_VISION_SSH_PORT"] = str(args.remote_vision_ssh_port)
    if args.remote_vision_ssh_user:
        os.environ["REMOTE_VISION_SSH_USER"] = args.remote_vision_ssh_user
    if args.remote_vision_ssh_password:
        os.environ["REMOTE_VISION_SSH_PASSWORD"] = args.remote_vision_ssh_password
    if args.remote_vision_local_port is not None:
        os.environ["REMOTE_VISION_LOCAL_PORT"] = str(args.remote_vision_local_port)
    if args.remote_vision_remote_port is not None:
        os.environ["REMOTE_VISION_REMOTE_PORT"] = str(args.remote_vision_remote_port)

    remote_backend_requested = (
        args.perception_backend == "remote"
        or bool(args.remote_vision_base_url)
        or bool(args.remote_vision_use_ssh_tunnel)
        or os.environ.get("RAANAV_PERCEPTION_BACKEND") == "remote"
        or bool(os.environ.get("REMOTE_VISION_BASE_URL"))
        or str(os.environ.get("REMOTE_VISION_USE_SSH_TUNNEL", "")).strip().lower()
        in {"1", "true", "yes", "y", "on"}
    )
    local_clip_requested = (
        bool(args.clip_model_path)
        or bool(args.clip_model_name)
        or bool(args.clip_online)
        or bool(os.environ.get("RAANAV_CLIP_MODEL_PATH"))
        or bool(os.environ.get("RAANAV_CLIP_MODEL"))
    )
    if args.remote_clip:
        os.environ["REMOTE_VISION_RETURN_CLIP"] = "1"
        os.environ["RAANAV_REMOTE_CLIP_EMBEDDING"] = "1"
        if not local_clip_requested:
            os.environ.setdefault("RAANAV_DISABLE_CLIP", "1")
    elif remote_backend_requested and not local_clip_requested:
        os.environ.setdefault("REMOTE_VISION_RETURN_CLIP", "1")
        os.environ.setdefault("RAANAV_REMOTE_CLIP_EMBEDDING", "1")
        os.environ.setdefault("RAANAV_DISABLE_CLIP", "1")
        print("[Remote CLIP] remote perception detected; using server-side CLIP and disabling workstation CLIP.")
    if args.remote_clip_model_path:
        os.environ["REMOTE_VISION_CLIP_MODEL_PATH"] = args.remote_clip_model_path
    if args.remote_clip_model_name:
        os.environ["REMOTE_VISION_CLIP_MODEL"] = args.remote_clip_model_name
    if args.remote_clip_online:
        os.environ["REMOTE_VISION_CLIP_LOCAL_FILES_ONLY"] = "0"
    if args.disable_clip:
        os.environ["RAANAV_DISABLE_CLIP"] = "1"
    if args.clip_model_path:
        os.environ["RAANAV_CLIP_MODEL_PATH"] = args.clip_model_path
    if args.clip_model_name:
        os.environ["RAANAV_CLIP_MODEL"] = args.clip_model_name
    if args.clip_online:
        os.environ["RAANAV_CLIP_LOCAL_FILES_ONLY"] = "0"

    run_sim_nav_loop(
        scene_dir=args.scene_dir,
        dataset_config=args.dataset_config,
        target=args.target,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        query_interval=args.query_interval,
        config_path=Path(args.config),
        step_size=args.step_size,
        n_views=args.n_views,
        use_gt_semantic=args.use_gt_semantic,
        live_viz=args.live_viz,
        use_navmesh=not args.no_navmesh,
    )


if __name__ == "__main__":
    main()
