"""从 HM3D Habitat 仿真场景中提取完整物体信息.

功能:
  1. 加载 Habitat 仿真场景，读取 semantic scene graph
  2. 提取每个物体的: category, 3D AABB/OBB, 中心点, 所属房间
  3. 将 3D 包围盒投影为 2D region (多边形)
  4. 解析 semantic.txt 获取额外标注信息
  5. 按房间分组，为每个物体生成 Object 节点

输出: 符合 AgenticRAG semantic_map 数据模型的 Floor/Room/Object 层级结构
"""
from __future__ import annotations

import csv
import io
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 将项目根目录加入 path
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from semantic_map import Floor, Object, Room

# ---------------------------------------------------------------------------
# 过滤掉的结构性类别 (不作为可寻物目标)
# ---------------------------------------------------------------------------
STRUCTURAL_CATEGORIES = frozenset({
    "wall", "floor", "ceiling", "void", "unknown", "misc",
    "door", "window", "column", "beam", "railing", "stair",
    "stairs", "banister", "balustrade",
})

# 物体稳定性先验：从 config/map.yaml 的 stability_priors 加载
# 若配置文件不存在或解析失败, 回退到内置硬编码字典
def _load_stability_priors() -> Tuple[Dict[str, float], float]:
    """从 config/map.yaml 加载 stability_priors, 回退到硬编码默认值."""
    _hardcoded = {
        "refrigerator": 0.95, "oven": 0.95, "bathtub": 0.95, "toilet": 0.95,
        "sink": 0.90, "fireplace": 0.95, "washer": 0.90, "dryer": 0.90,
        "couch": 0.85, "sofa": 0.85, "bed": 0.90, "wardrobe": 0.90, "closet": 0.90,
        "cabinet": 0.85, "bookshelf": 0.85, "shelf": 0.80, "dresser": 0.85,
        "desk": 0.80, "table": 0.80, "dining table": 0.80, "counter": 0.85,
        "chair": 0.65, "sofa chair": 0.70, "armchair": 0.70, "stool": 0.55,
        "tv": 0.70, "television": 0.70, "monitor": 0.65, "microwave": 0.70,
        "lamp": 0.55, "plant": 0.60, "potted plant": 0.60, "rug": 0.70,
        "curtain": 0.65, "mirror": 0.70, "picture": 0.70, "painting": 0.70,
        "clock": 0.65, "decoration": 0.55, "vase": 0.55, "podium": 0.75,
        "pillow": 0.30, "cushion": 0.30, "blanket": 0.25, "towel": 0.25,
        "book": 0.35, "bottle": 0.30, "cup": 0.25, "mug": 0.25,
        "remote": 0.20, "phone": 0.20, "toy": 0.30, "bag": 0.25,
        "shoe": 0.25, "clothes": 0.20, "basket": 0.40, "box": 0.40,
    }
    _default_stab = 0.50
    try:
        import yaml
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "map.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            sp = cfg.get("stability_priors")
            if isinstance(sp, dict):
                # 支持外部文件引用: stability_priors.config_path
                ext_path = sp.get("config_path")
                if ext_path and isinstance(ext_path, str):
                    ext_abs = os.path.normpath(os.path.join(os.path.dirname(cfg_path), ext_path))
                    if os.path.exists(ext_abs):
                        with open(ext_abs, "r", encoding="utf-8") as ef:
                            ext_data = yaml.safe_load(ef) or {}
                        sp = ext_data.get("stability_priors", ext_data)
                default_val = float(sp.pop("default", _default_stab))
                priors = {str(k).lower(): float(v) for k, v in sp.items()
                          if isinstance(v, (int, float))}
                # 合并: 配置覆盖硬编码
                merged = {**_hardcoded, **priors}
                return merged, default_val
    except Exception:
        pass
    return _hardcoded, _default_stab

STABILITY_PRIOR, DEFAULT_STABILITY = _load_stability_priors()


def _vec3_to_list(v) -> List[float]:
    """将 habitat_sim 的 Vector3 转为 Python list."""
    return [round(float(v[0]), 5), round(float(v[1]), 5), round(float(v[2]), 5)]


def _aabb_to_2d_region(aabb_min: List[float], aabb_max: List[float]) -> List[Dict[str, float]]:
    """将 3D AABB 投影为 2D 矩形多边形 (xz 平面).

    HM3D 坐标系: x-right, y-up, z-back
    投影到 xz 平面 -> region 的 x 对应 world-x, y 对应 world-z
    """
    x_min, z_min = aabb_min[0], aabb_min[2]
    x_max, z_max = aabb_max[0], aabb_max[2]
    return [
        {"x": round(x_min, 5), "y": round(z_min, 5)},
        {"x": round(x_max, 5), "y": round(z_min, 5)},
        {"x": round(x_max, 5), "y": round(z_max, 5)},
        {"x": round(x_min, 5), "y": round(z_max, 5)},
        {"x": round(x_min, 5), "y": round(z_min, 5)},  # 闭合
    ]


def _obb_to_2d_region(
    center: List[float], half_extents: List[float], rotation: Optional[List[List[float]]]
) -> List[Dict[str, float]]:
    """将 3D OBB 投影为 2D 旋转矩形多边形 (xz 平面).

    优先使用 OBB 因为 HM3D 的 AABB 有时过于宽松。
    """
    cx, cz = center[0], center[2]
    hx, hz = half_extents[0], half_extents[2]

    if rotation is not None:
        # 提取 xz 平面的旋转分量
        R = np.array(rotation)
        # OBB 的 4 个角点 (在局部坐标)
        local_corners = np.array([
            [-hx, -hz], [hx, -hz], [hx, hz], [-hx, hz]
        ])
        # 用旋转矩阵的 xz 分量旋转
        R2d = np.array([[R[0, 0], R[0, 2]], [R[2, 0], R[2, 2]]])
        rotated = local_corners @ R2d.T
        corners = rotated + np.array([cx, cz])
    else:
        # 无旋转，退化为 AABB
        corners = np.array([
            [cx - hx, cz - hz], [cx + hx, cz - hz],
            [cx + hx, cz + hz], [cx - hx, cz + hz],
        ])

    region = [{"x": round(float(c[0]), 5), "y": round(float(c[1]), 5)} for c in corners]
    region.append(region[0].copy())  # 闭合
    return region


def parse_semantic_txt(semantic_txt_path: str) -> Dict[int, Dict[str, Any]]:
    """解析 HM3D semantic.txt 标注文件.

    返回 {semantic_id: {"category": str, "region_id": int, "color_hex": str}}
    """
    entries = {}
    with open(semantic_txt_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if line_no == 0 or not line:
                continue
            reader = csv.reader(io.StringIO(line))
            for row in reader:
                if len(row) < 4:
                    continue
                obj_id = int(row[0])
                color_hex = row[1].strip()
                category = row[2].strip().strip('"')
                region_id = int(row[3])
                entries[obj_id] = {
                    "category": category,
                    "region_id": region_id,
                    "color_hex": color_hex,
                }
    return entries


def _get_stability(category: str) -> float:
    """根据类别名返回稳定性先验."""
    cat_lower = category.lower().strip()
    if cat_lower in STABILITY_PRIOR:
        return STABILITY_PRIOR[cat_lower]
    # 模糊匹配
    for key, val in STABILITY_PRIOR.items():
        if key in cat_lower or cat_lower in key:
            return val
    return DEFAULT_STABILITY


def make_simulator(
    scene_glb: str,
    dataset_config: str,
    resolution: Tuple[int, int] = (480, 640),
    enable_semantic: bool = False,
    enable_depth: bool = False,
    hfov: float = 90.0,
):
    """创建最小化 habitat_sim Simulator, 用于读取语义场景图.

    Args:
        scene_glb: .basis.glb 场景文件路径
        dataset_config: scene_dataset_config.json 路径
        resolution: (height, width)
        enable_semantic: 是否启用语义传感器 (闭环导航需要)
        enable_depth: 是否启用深度传感器 (occupancy grid 构建需要)
        hfov: 水平视场角 (度), 默认 90°
    """
    import habitat_sim

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = os.path.abspath(dataset_config)
    sim_cfg.scene_id = scene_glb
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = 0
    sim_cfg.load_semantic_mesh = True
    sim_cfg.use_semantic_textures = True  # HM3D语义纹理模式

    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "color"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = list(resolution)
    sensor.hfov = hfov

    sensors = [sensor]

    if enable_semantic:
        sem_sensor = habitat_sim.CameraSensorSpec()
        sem_sensor.uuid = "semantic"
        sem_sensor.sensor_type = habitat_sim.SensorType.SEMANTIC
        sem_sensor.resolution = list(resolution)
        sem_sensor.hfov = hfov
        sensors.append(sem_sensor)

    if enable_depth:
        depth_sensor = habitat_sim.CameraSensorSpec()
        depth_sensor.uuid = "depth"
        depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor.resolution = list(resolution)
        depth_sensor.hfov = hfov
        sensors.append(depth_sensor)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensors

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    return habitat_sim.Simulator(cfg)


def extract_objects_from_scene(
    sim,
    semantic_txt_path: str,
    exclude_structural: bool = True,
    min_volume: float = 1e-5,
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[str, Any]]:
    """从 habitat simulator 中提取所有物体信息.

    Args:
        sim: habitat_sim.Simulator 实例
        semantic_txt_path: semantic.txt 文件路径
        exclude_structural: 是否过滤结构性类别 (wall/floor/ceiling等)
        min_volume: 最小体积过滤 (排除无效极小物体)

    Returns:
        (room_objects, scene_meta)
        room_objects: {region_id: [obj_info_dict, ...]}
        scene_meta: 场景级元数据
    """
    txt_entries = parse_semantic_txt(semantic_txt_path)
    sem_scene = sim.semantic_scene

    # 场景 AABB (有时 habitat 返回 0，后续从物体推算)
    scene_aabb = sem_scene.aabb
    scene_meta = {
        "aabb_min": _vec3_to_list(scene_aabb.min),
        "aabb_max": _vec3_to_list(scene_aabb.max),
        "aabb_center": _vec3_to_list(scene_aabb.center()),
        "aabb_size": _vec3_to_list(scene_aabb.size()),
    }

    room_objects: Dict[int, List[Dict[str, Any]]] = {}
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for obj in sem_scene.objects:
        if obj is None:
            continue
        sid = obj.semantic_id
        txt_info = txt_entries.get(sid, {})
        category = txt_info.get(
            "category",
            obj.category.name() if obj.category else "unknown",
        )
        region_id = txt_info.get("region_id", -1)

        # 过滤结构性物体
        if exclude_structural and category.lower().strip() in STRUCTURAL_CATEGORIES:
            continue

        # 3D AABB
        aabb = obj.aabb
        aabb_min = _vec3_to_list(aabb.min)
        aabb_max = _vec3_to_list(aabb.max)
        aabb_center = _vec3_to_list(aabb.center())
        aabb_size = _vec3_to_list(aabb.size())

        # 体积过滤
        volume = abs(aabb_size[0] * aabb_size[1] * aabb_size[2])
        if volume < min_volume:
            continue

        # OBB
        obb = obj.obb
        obb_center = _vec3_to_list(obb.center)
        obb_half_extents = _vec3_to_list(obb.half_extents)
        try:
            obb_rotation = [
                [round(float(obb.rotation[i][j]), 6) for j in range(3)]
                for i in range(3)
            ]
        except Exception:
            obb_rotation = None

        # 2D region: 优先使用 OBB (更精确), 回退到 AABB
        # 同时检查 AABB 是否过大 (HM3D 常见问题)
        aabb_xz_size = max(abs(aabb_size[0]), abs(aabb_size[2]))
        obb_xz_size = max(abs(obb_half_extents[0]), abs(obb_half_extents[2])) * 2

        # 如果 AABB 比 OBB 大 3 倍以上，说明 AABB 可能不准确
        use_obb = obb_xz_size > 0.01 and (aabb_xz_size > obb_xz_size * 3 or aabb_xz_size > 5.0)
        if use_obb:
            region = _obb_to_2d_region(obb_center, obb_half_extents, obb_rotation)
            effective_center = obb_center
        else:
            region = _aabb_to_2d_region(aabb_min, aabb_max)
            effective_center = aabb_center

        # 2D 中心 (xz)
        pos_2d = {"x": round(effective_center[0], 5), "y": round(effective_center[2], 5)}

        obj_info = {
            "semantic_id": sid,
            "category": category,
            "region_id": region_id,
            "aabb_min": aabb_min,
            "aabb_max": aabb_max,
            "aabb_center": aabb_center,
            "aabb_size": aabb_size,
            "obb_center": obb_center,
            "obb_half_extents": obb_half_extents,
            "obb_rotation": obb_rotation,
            "region_2d": region,
            "pos_2d": pos_2d,
            "pos_3d": list(effective_center),
            "stability": _get_stability(category),
            "last_update_time": now_str,
            "use_obb": use_obb,
        }
        room_objects.setdefault(region_id, []).append(obj_info)

    # 如果 scene_aabb 全零，从物体推算
    all_objs = [o for objs in room_objects.values() for o in objs]
    if all_objs and all(v == 0.0 for v in scene_meta["aabb_size"]):
        all_mins = np.array([o["aabb_min"] for o in all_objs])
        all_maxs = np.array([o["aabb_max"] for o in all_objs])
        computed_min = all_mins.min(axis=0).tolist()
        computed_max = all_maxs.max(axis=0).tolist()
        computed_center = ((np.array(computed_min) + np.array(computed_max)) / 2).tolist()
        computed_size = (np.array(computed_max) - np.array(computed_min)).tolist()
        scene_meta = {
            "aabb_min": [round(v, 4) for v in computed_min],
            "aabb_max": [round(v, 4) for v in computed_max],
            "aabb_center": [round(v, 4) for v in computed_center],
            "aabb_size": [round(v, 4) for v in computed_size],
        }

    return room_objects, scene_meta


def _compute_room_region(objects: List[Dict]) -> List[Dict[str, float]]:
    """从房间内所有物体的 AABB 估算房间 2D 区域."""
    if not objects:
        return []
    all_x = []
    all_z = []
    for o in objects:
        all_x.extend([o["aabb_min"][0], o["aabb_max"][0]])
        all_z.extend([o["aabb_min"][2], o["aabb_max"][2]])
    x_min, x_max = min(all_x), max(all_x)
    z_min, z_max = min(all_z), max(all_z)
    # 稍微膨胀 0.3m
    pad = 0.3
    return [
        {"x": round(x_min - pad, 4), "y": round(z_min - pad, 4)},
        {"x": round(x_max + pad, 4), "y": round(z_min - pad, 4)},
        {"x": round(x_max + pad, 4), "y": round(z_max + pad, 4)},
        {"x": round(x_min - pad, 4), "y": round(z_max + pad, 4)},
        {"x": round(x_min - pad, 4), "y": round(z_min - pad, 4)},
    ]


def _compute_z_range(objects: List[Dict]) -> Dict[str, float]:
    """从物体集合估算层高范围."""
    if not objects:
        return {"z_min": 0.0, "z_max": 4.0}
    all_y_min = min(o["aabb_min"][1] for o in objects)
    all_y_max = max(o["aabb_max"][1] for o in objects)
    return {"z_min": round(all_y_min, 4), "z_max": round(all_y_max, 4)}


def build_semantic_map(
    room_objects: Dict[int, List[Dict[str, Any]]],
    scene_meta: Dict[str, Any],
    floor_id: str = "F1",
    scan_version: str = "1",
) -> List[Floor]:
    """将提取的物体信息构建为 Floor/Room/Object 层级结构.

    Args:
        room_objects: {region_id: [obj_info, ...]}
        scene_meta: 场景元数据
        floor_id: 楼层 ID
        scan_version: 扫描版本号 (用于 imgs/description 字典键)

    Returns:
        floors: Floor 列表
    """
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rooms = []
    all_objects_flat = []

    for region_id in sorted(room_objects.keys()):
        obj_list = room_objects[region_id]
        room_id = f"R{region_id}"

        objects = []
        for idx, obj_info in enumerate(obj_list):
            obj_id = f"{obj_info['category'].replace(' ', '_')}_{idx}_{room_id}"
            obj = Object(
                obj_id=obj_id,
                label=obj_info["category"],
                region=obj_info["region_2d"],
                stability=obj_info["stability"],
                clip_embedding=[],  # 初始为空，后续由 CLIP 编码填充
                cfd=1.0,  # 仿真场景：视觉置信度 = 1.0
                room_id=room_id,
                R_objs={},
                imgs={},
                N=1,
                description={scan_version: [f"semantic_id={obj_info['semantic_id']}"]},
                last_update_time=obj_info["last_update_time"],
                cooccur_stats={},
                exist_prob=1.0,
            )
            # 扩展属性: 保留 3D 空间信息 (GMM 等后续模块可能需要)
            obj.pos_3d = obj_info["pos_3d"]
            obj.pos_2d = obj_info["pos_2d"]
            obj.bbox_3d = {
                "min": obj_info["aabb_min"],
                "max": obj_info["aabb_max"],
                "center": obj_info["aabb_center"],
                "size": obj_info["aabb_size"],
            }
            obj.obb = {
                "center": obj_info["obb_center"],
                "half_extents": obj_info["obb_half_extents"],
                "rotation": obj_info["obb_rotation"],
            }
            objects.append(obj)
            all_objects_flat.append(obj_info)

        room_region = _compute_room_region(obj_list)
        room = Room(
            room_id=room_id,
            room_name={scan_version: f"region_{region_id}"},
            objects=objects,
            region=room_region,
            floor_id=floor_id,
            door_positions=[],
            N=1,
            imgs={},
            description={scan_version: f"auto-extracted room with {len(objects)} objects"},
        )
        rooms.append(room)

    z_range = _compute_z_range(all_objects_flat) if all_objects_flat else {"z_min": 0.0, "z_max": 4.0}

    floor = Floor(
        floor_id=floor_id,
        rooms=rooms,
        z_range=z_range,
        description=f"auto-extracted floor from HM3D scene",
    )

    return [floor]


def compute_R_objs_by_proximity(
    floors: List[Floor],
    inflation_rate: float = 1.2,
    overlap_threshold: float = 0.05,
) -> None:
    """基于空间邻近性计算物体间的 R_objs 关系 (原地修改).

    对每个房间内的物体，膨胀其 region 后判断与其他物体的重叠。
    """
    from semantic_map_Create.utility import inflate_region_from_center, judge_overlaps

    for floor in floors:
        for room in floor.rooms:
            for obj in room.objects:
                for other in room.objects:
                    if other.obj_id == obj.obj_id:
                        continue
                    inflated = inflate_region_from_center(obj.region, inflation_rate)
                    if judge_overlaps(inflated, overlap_threshold, other.region):
                        obj.add_R_objs(other.obj_id, Nt=1, Rcfd=1.0, Nr_inc=1)


# ---------------------------------------------------------------------------
# 便捷入口函数
# ---------------------------------------------------------------------------
def extract_scene(
    scene_dir: str,
    dataset_config: str,
    exclude_structural: bool = True,
    compute_relations: bool = True,
    inflation_rate: float = 1.2,
    overlap_threshold: float = 0.05,
) -> Tuple[List[Floor], Dict[str, Any]]:
    """一键从 HM3D 场景目录提取完整语义地图.

    Args:
        scene_dir: 场景目录 (包含 .basis.glb 和 .semantic.txt)
        dataset_config: scene_dataset_config.json 路径
        exclude_structural: 过滤结构性类别
        compute_relations: 是否计算邻近关系 R_objs
        inflation_rate: 关系计算时的区域膨胀率
        overlap_threshold: 重叠判断阈值

    Returns:
        (floors, scene_meta)
    """
    import habitat_sim  # noqa: F811 (延迟导入)

    # 从目录名解析场景 ID
    scene_name = os.path.basename(scene_dir)
    parts = scene_name.split("-", 1)
    scene_stem = parts[1] if len(parts) > 1 else scene_name

    basis_glb = os.path.join(scene_dir, f"{scene_stem}.basis.glb")
    semantic_txt = os.path.join(scene_dir, f"{scene_stem}.semantic.txt")

    if not os.path.isfile(basis_glb):
        raise FileNotFoundError(f"场景文件不存在: {basis_glb}")
    if not os.path.isfile(semantic_txt):
        raise FileNotFoundError(f"语义标注文件不存在: {semantic_txt}")

    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

    sim = make_simulator(basis_glb, dataset_config)
    try:
        room_objects, scene_meta = extract_objects_from_scene(
            sim, semantic_txt, exclude_structural=exclude_structural,
        )
    finally:
        sim.close()

    floors = build_semantic_map(room_objects, scene_meta)

    if compute_relations:
        compute_R_objs_by_proximity(floors, inflation_rate, overlap_threshold)

    # 统计摘要
    total_objs = sum(len(o) for o in room_objects.values())
    total_rooms = len(room_objects)
    print(f"[scene_extract] 提取完成: {total_rooms} rooms, {total_objs} objects (excl_structural={exclude_structural})")

    return floors, scene_meta
