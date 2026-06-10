"""完整建图流水线: 从 HM3D 场景 → 语义地图 JSON + 占据栅格.

用法:
  python -m semantic_map_Create.map_Create --scene-dir <场景目录> [--output-dir <输出目录>]

示例:
  cd /home/adminer/agentRAG/AgenticRAG
  conda run -n agentrag python -m semantic_map_Create.map_Create \
      --scene-dir /home/adminer/agentRAG/experiment_data/hm3d/val/00824-Dd4bFSTQ8gi \
      --output-dir /home/adminer/agentRAG/AgenticRAG/RAG_Graph/scene_build
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from semantic_map import Floor, Object, Room
from semantic_map_Create.scene_extract import extract_scene
from semantic_map_Create.occupancy_grid import OccupancyGrid


# ---------------------------------------------------------------------------
# 默认路径
# ---------------------------------------------------------------------------
DEFAULT_DATASET_CONFIG = "/home/adminer/agentRAG/experiment_data/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
DEFAULT_SCENE_DIR = "/home/adminer/agentRAG/experiment_data/hm3d/val/00824-Dd4bFSTQ8gi"
DEFAULT_OUTPUT_DIR = "/home/adminer/agentRAG/AgenticRAG/RAG_Graph/scene_build"

# CLIP 模型
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


def create_semantic_map(
    scene_dir: str,
    dataset_config: str,
    output_dir: str,
    grid_resolution: float = 0.05,
    navmesh_samples: int = 50000,
    exclude_structural: bool = True,
    compute_relations: bool = True,
    inflation_rate: float = 1.2,
    overlap_threshold: float = 0.05,
    fill_clip: bool = True,
    clip_model: str = DEFAULT_CLIP_MODEL,
) -> Dict[str, Any]:
    """从 HM3D 场景一键创建完整语义地图 + 占据栅格.

    流水线:
      1. 从 Habitat 仿真器提取物体信息 → Floor/Room/Object 层级
      2. 计算物体间空间邻近关系 (R_objs)
      3. 从 navmesh 构建 2D 占据栅格地图
      4. 在栅格上标注物体位置
      5. 序列化保存为 JSON + npy 文件

    Args:
        scene_dir: HM3D 场景目录 (含 .basis.glb, .semantic.txt)
        dataset_config: scene_dataset_config.json 路径
        output_dir: 输出目录
        grid_resolution: 栅格分辨率 (m/px)
        navmesh_samples: navmesh 采样点数量
        exclude_structural: 是否过滤结构性物体
        compute_relations: 是否计算邻近关系
        inflation_rate: 关系计算膨胀率
        overlap_threshold: 重叠判断阈值

    Returns:
        result: 包含输出路径和统计信息的字典
    """
    os.makedirs(output_dir, exist_ok=True)
    scene_name = os.path.basename(scene_dir)
    result = {"scene_name": scene_name, "output_dir": output_dir}

    print(f"{'='*60}")
    print(f"  AgenticRAG 建图流水线")
    print(f"  场景: {scene_name}")
    print(f"  输出: {output_dir}")
    print(f"{'='*60}")

    # ---------------------------------------------------------------
    # Step 1: 提取物体信息
    # ---------------------------------------------------------------
    print("\n[Step 1/4] 从仿真场景提取物体信息...")
    floors, scene_meta = extract_scene(
        scene_dir=scene_dir,
        dataset_config=dataset_config,
        exclude_structural=exclude_structural,
        compute_relations=compute_relations,
        inflation_rate=inflation_rate,
        overlap_threshold=overlap_threshold,
    )

    # 统计
    total_objects = sum(
        len(room.objects) for floor in floors for room in floor.rooms
    )
    total_rooms = sum(len(floor.rooms) for floor in floors)
    result["total_objects"] = total_objects
    result["total_rooms"] = total_rooms
    result["total_floors"] = len(floors)

    print(f"  → {len(floors)} floor(s), {total_rooms} room(s), {total_objects} object(s)")

    # ---------------------------------------------------------------
    # Step 2: 保存语义地图 JSON
    # ---------------------------------------------------------------
    print("\n[Step 2/4] 序列化语义地图...")

    # 2a: 标准 Floor 格式 (与 map_Update 兼容)
    map_data = [floor.to_dict() for floor in floors]

    # 扩展: 将 3D 空间信息写入 Object 字典 (to_dict 不包含动态属性)
    for floor_dict, floor_obj in zip(map_data, floors):
        for room_dict, room_obj in zip(floor_dict["rooms"], floor_obj.rooms):
            for obj_dict, obj_instance in zip(room_dict["objects"], room_obj.objects):
                if hasattr(obj_instance, "pos_3d"):
                    obj_dict["pos_3d"] = obj_instance.pos_3d
                if hasattr(obj_instance, "pos_2d"):
                    obj_dict["pos_2d"] = obj_instance.pos_2d
                if hasattr(obj_instance, "bbox_3d"):
                    obj_dict["bbox_3d"] = obj_instance.bbox_3d
                if hasattr(obj_instance, "obb"):
                    obj_dict["obb"] = obj_instance.obb

    map_path = os.path.join(output_dir, f"{scene_name}_semantic_map.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(map_data, f, ensure_ascii=False, indent=2)
    result["semantic_map_path"] = map_path
    print(f"  → 语义地图: {map_path}")

    # 2b: 场景元数据
    meta_path = os.path.join(output_dir, f"{scene_name}_scene_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(scene_meta, f, ensure_ascii=False, indent=2)
    result["scene_meta_path"] = meta_path

    # ---------------------------------------------------------------
    # Step 2.5: 计算 CLIP 文本嵌入并写回 JSON + 构建 CLIP 索引
    # ---------------------------------------------------------------
    if fill_clip:
        print("\n[Step 2.5] 计算 CLIP 文本嵌入...")
        # 设置 HuggingFace 镜像 (国内环境)
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from CLIP_RAG.build_clip_index import build_embeddings

        labels = []
        label_to_indices = {}  # label -> list of (floor_idx, room_idx, obj_idx)
        for fi, floor_dict in enumerate(map_data):
            for ri, room_dict in enumerate(floor_dict["rooms"]):
                for oi, obj_dict in enumerate(room_dict["objects"]):
                    lbl = obj_dict.get("label", "unknown")
                    if lbl not in label_to_indices:
                        label_to_indices[lbl] = []
                        labels.append(lbl)
                    label_to_indices[lbl].append((fi, ri, oi))

        # 只编码不重复的标签 → 同标签共享向量
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vecs = build_embeddings(labels, clip_model, device=device, batch_size=32)

        # 写入 JSON dict 和 Object 实例
        n_filled = 0
        for li, lbl in enumerate(labels):
            emb_list = vecs[li].round(6).tolist()
            for fi, ri, oi in label_to_indices[lbl]:
                map_data[fi]["rooms"][ri]["objects"][oi]["clip_embedding"] = emb_list
                # 同步更新内存中的 Object 实例
                floors[fi].rooms[ri].objects[oi].clip_embedding = emb_list
                n_filled += 1

        # 重新保存 JSON (含 clip_embedding)
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(map_data, f, ensure_ascii=False, indent=2)

        # 构建 CLIP 索引目录
        clip_index_dir = os.path.join(output_dir, "clip_index")
        os.makedirs(clip_index_dir, exist_ok=True)

        all_ids = []
        all_texts = []
        all_vecs_list = []
        for floor_dict in map_data:
            for room_dict in floor_dict["rooms"]:
                for obj_dict in room_dict["objects"]:
                    all_ids.append(obj_dict["obj_id"])
                    all_texts.append(obj_dict.get("label", "unknown"))
                    all_vecs_list.append(obj_dict["clip_embedding"])

        import hashlib, datetime as dt
        emb_mat = np.array(all_vecs_list, dtype="float32")
        np.save(os.path.join(clip_index_dir, "embeddings.npy"), emb_mat)
        with open(os.path.join(clip_index_dir, "ids.json"), "w", encoding="utf-8") as f:
            json.dump(all_ids, f, ensure_ascii=False, indent=2)
        with open(os.path.join(clip_index_dir, "objects.jsonl"), "w", encoding="utf-8") as f:
            for oid, txt in zip(all_ids, all_texts):
                f.write(json.dumps({"obj_id": oid, "description": txt}, ensure_ascii=False) + "\n")
        clip_meta = {
            "model": clip_model,
            "count": len(all_ids),
            "dim": int(emb_mat.shape[1]),
            "field_mode": "label",
            "source_map": map_path,
        }
        with open(os.path.join(clip_index_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(clip_meta, f, ensure_ascii=False, indent=2)

        result["clip_index_dir"] = clip_index_dir
        result["clip_unique_labels"] = len(labels)
        result["clip_dim"] = int(emb_mat.shape[1])
        print(f"  → {n_filled} 物体填充 CLIP 嵌入 ({len(labels)} 唯一标签, dim={emb_mat.shape[1]})")
        print(f"  → CLIP 索引: {clip_index_dir}/")
    else:
        print("\n[跳过 CLIP 嵌入]")

    # ---------------------------------------------------------------
    # Step 3: 构建占据栅格
    # ---------------------------------------------------------------
    print("\n[Step 3/4] 构建占据栅格地图...")

    # 需要重新打开 simulator (因为 step 1 已关闭)
    import habitat_sim
    parts = scene_name.split("-", 1)
    scene_stem = parts[1] if len(parts) > 1 else scene_name
    basis_glb = os.path.join(scene_dir, f"{scene_stem}.basis.glb")

    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

    from semantic_map_Create.scene_extract import make_simulator
    sim = make_simulator(basis_glb, dataset_config)
    try:
        occ_grid = OccupancyGrid.from_navmesh_fast(
            sim,
            resolution=grid_resolution,
            num_samples=navmesh_samples,
        )
    finally:
        sim.close()

    grid_dir = os.path.join(output_dir, "occupancy_grid")
    grid_paths = occ_grid.save(grid_dir, prefix="occupancy")
    result["occupancy_grid_dir"] = grid_dir
    result["occupancy_grid_paths"] = grid_paths
    print(f"  → 栅格地图: {grid_dir}/")
    print(f"  → 分辨率: {grid_resolution} m/px, 大小: {occ_grid.shape}")

    # ---------------------------------------------------------------
    # Step 4: 在栅格上标注物体位置
    # ---------------------------------------------------------------
    print("\n[Step 4/4] 生成物体标注可视化...")
    obj_positions = []
    obj_labels = []
    for floor in floors:
        for room in floor.rooms:
            for obj in room.objects:
                if hasattr(obj, "pos_2d"):
                    obj_positions.append(obj.pos_2d)
                    obj_labels.append(obj.label)

    if obj_positions:
        import cv2
        annotated_img = occ_grid.plot_objects_on_grid(obj_positions, obj_labels)
        annotated_path = os.path.join(output_dir, f"{scene_name}_objects_on_grid.png")
        cv2.imwrite(annotated_path, annotated_img)
        result["annotated_grid_path"] = annotated_path
        print(f"  → 物体标注图: {annotated_path}")

    # ---------------------------------------------------------------
    # 完成
    # ---------------------------------------------------------------
    summary_path = os.path.join(output_dir, f"{scene_name}_build_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  建图完成!")
    print(f"  - 语义地图: {result.get('semantic_map_path', 'N/A')}")
    print(f"  - 占据栅格: {result.get('occupancy_grid_dir', 'N/A')}")
    print(f"  - 物体标注图: {result.get('annotated_grid_path', 'N/A')}")
    print(f"  - 统计: {total_objects} 物体, {total_rooms} 房间")
    print(f"{'='*60}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="AgenticRAG 建图流水线: HM3D 场景 → 语义地图 + 占据栅格"
    )
    parser.add_argument(
        "--scene-dir", type=str, default=DEFAULT_SCENE_DIR,
        help=f"HM3D 场景目录 (默认: {DEFAULT_SCENE_DIR})",
    )
    parser.add_argument(
        "--dataset-config", type=str, default=DEFAULT_DATASET_CONFIG,
        help="scene_dataset_config.json 路径",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录 (默认: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--grid-resolution", type=float, default=0.05,
        help="栅格分辨率 (m/px), 默认 0.05",
    )
    parser.add_argument(
        "--navmesh-samples", type=int, default=50000,
        help="Navmesh 采样点数, 默认 50000",
    )
    parser.add_argument(
        "--include-structural", action="store_true",
        help="包含结构性物体 (wall/floor/ceiling)",
    )
    parser.add_argument(
        "--no-relations", action="store_true",
        help="跳过物体邻近关系计算",
    )
    parser.add_argument(
        "--no-clip", action="store_true",
        help="跳过 CLIP 嵌入计算",
    )
    parser.add_argument(
        "--clip-model", type=str, default=DEFAULT_CLIP_MODEL,
        help=f"CLIP 模型名 (默认: {DEFAULT_CLIP_MODEL})",
    )
    args = parser.parse_args()

    create_semantic_map(
        scene_dir=args.scene_dir,
        dataset_config=args.dataset_config,
        output_dir=args.output_dir,
        grid_resolution=args.grid_resolution,
        navmesh_samples=args.navmesh_samples,
        exclude_structural=not args.include_structural,
        compute_relations=not args.no_relations,
        fill_clip=not args.no_clip,
        clip_model=args.clip_model,
    )


if __name__ == "__main__":
    main()