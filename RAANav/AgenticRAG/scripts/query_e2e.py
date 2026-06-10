"""端到端查询测试: 自然语言目标 → CLIP匹配 → GMM评分 → 概率热图.

用法:
  cd /home/adminer/agentRAG/AgenticRAG
  conda run -n agentrag python scripts/query_e2e.py --target chair
  conda run -n agentrag python scripts/query_e2e.py --target "potted plant" --top-k 5

流水线:
  1. CLIP 语义检索: 从索引中找出与 target 语义相近的物体实例
  2. GMM 评分: 综合 Rscore / R_objs / stability / cfd 算出每个相关物体分数
  3. 高斯混合概率场: 将 GMM 分数转为 2D 高斯分布叠加到占据栅格上
  4. 可视化: 输出热图 + Top-K 候选导航点

输出文件 (写入 --output-dir):
  query_<target>_heatmap.png   — 概率热图叠加栅格
  query_<target>_result.json   — 结构化结果 (候选点、分数)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
CONFIG_PATH = Path("config/map.yaml")


def _load_yaml(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    import yaml
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _try_cuda() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _get_pos_xy(pos) -> Optional[Tuple[float, float]]:
    """从 pos_2d 中提取 (world_x, world_z). 
    支持 dict {'x':..,'y':..} 或 list [x,z] 格式."""
    if pos is None:
        return None
    if isinstance(pos, dict):
        x = pos.get("x")
        y = pos.get("y")  # y 在 2D 投影中实际代表 world-z
        if x is not None and y is not None:
            return (float(x), float(y))
    elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
        return (float(pos[0]), float(pos[1]))
    return None


# ---------------------------------------------------------------------------
# Step 1: CLIP 语义检索
# ---------------------------------------------------------------------------
def clip_search_objects(
    query: str, cfg: Dict[str, Any], min_score: Optional[float] = None
) -> List[Tuple[str, float]]:
    """返回 [(obj_id, cosine_score)] 按分数降序."""
    from CLIP_RAG.query_clip_index import load_index, encode_query, search

    map_cfg = cfg.get("map_config") or {}
    index_dir = map_cfg.get("CLIP_RAG_map_dir")
    if not index_dir or not Path(index_dir).exists():
        raise FileNotFoundError(f"CLIP 索引目录不存在: {index_dir}")

    ids, vecs, texts, meta = load_index(Path(index_dir))
    device = _try_cuda()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    qv = encode_query([query], meta["model"], device=device)

    clip_cfg = cfg.get("CLIP_RAG") or {}
    thr = min_score if min_score is not None else float(clip_cfg.get("consine_similarity_filmin", 0.75))
    results = search(vecs, ids, qv, top_k=None, min_score=thr)
    return [(oid, sc) for _rank, oid, sc, _idx in results]


# ---------------------------------------------------------------------------
# Step 2: 加载地图并构建 GMM 特征集
# ---------------------------------------------------------------------------
def load_map_and_score(
    target: str, cfg: Dict[str, Any], clip_hits: List[Tuple[str, float]]
) -> Dict[str, Any]:
    """基于 CLIP 匹配 + 地图关系信息计算 GMM 评分.

    简化版: 不依赖 agent_Rscore 输出 (仿真场景无 LLM 预测).
    直接用物体地图中的 R_objs 空间共现信息打分.
    """
    from GMM_map_Create.GMM_map_calcualte import (
        Calculate_obj_Score,
        Calculate_Robj_Score,
    )

    map_cfg = cfg.get("map_config") or {}
    map_json = map_cfg.get("map_merged_json", "")
    if not map_json or not Path(map_json).exists():
        raise FileNotFoundError(f"地图 JSON 不存在: {map_json}")

    with open(map_json, "r", encoding="utf-8") as f:
        floors = json.load(f)

    # 索引所有物体
    obj_index: Dict[str, Dict[str, Any]] = {}
    for fl in floors:
        for room in fl.get("rooms", []):
            for obj in room.get("objects", []):
                oid = obj.get("obj_id")
                if oid:
                    obj_index[oid] = obj

    target_ids = [oid for oid, _sc in clip_hits if oid in obj_index]

    # 目标物体自身得分
    targets_self = []
    for tid in target_ids:
        obj = obj_index[tid]
        self_score = Calculate_obj_Score(
            N=obj.get("N", 1),
            stability=obj.get("stability", 0.5),
            cfd=obj.get("cfd"),
            config_path=CONFIG_PATH,
        )
        targets_self.append({
            "obj_id": tid,
            "label": obj.get("label"),
            "self_score": self_score,
            "pos_2d": obj.get("pos_2d"),
            "region": obj.get("region"),
        })

    # 相关物体得分
    related_scores: Dict[str, float] = {}
    related_info: Dict[str, Dict[str, Any]] = {}
    for tid in target_ids:
        t_obj = obj_index[tid]
        R_map = t_obj.get("R_objs") or {}
        if not isinstance(R_map, dict):
            continue
        N_target = int(t_obj.get("N", 1) or 1)
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
                Rscore=0.0,  # 无 agent 预测，仅靠共现关系
                Rcfd=Rcfd,
                stability=r_obj.get("stability", 0.5),
                exist_prob=r_obj.get("exist_prob", 1.0),
                config_path=CONFIG_PATH,
            )
            # 取最高分
            if score > related_scores.get(rel_oid, 0.0):
                related_scores[rel_oid] = score
                related_info[rel_oid] = {
                    "obj_id": rel_oid,
                    "label": r_obj.get("label"),
                    "score": score,
                    "pos_2d": r_obj.get("pos_2d"),
                    "region": r_obj.get("region"),
                }

    # 汇总所有 GMM 参与者: 目标 + 相关
    gmm_scores: Dict[str, float] = {}
    gmm_positions: Dict[str, List[float]] = {}  # obj_id -> [x, z]

    for t in targets_self:
        oid = t["obj_id"]
        gmm_scores[oid] = t["self_score"]
        xy = _get_pos_xy(t.get("pos_2d"))
        if xy:
            gmm_positions[oid] = list(xy)

    for oid, info in related_info.items():
        if info["score"] > 0:
            gmm_scores[oid] = info["score"]
            xy = _get_pos_xy(info.get("pos_2d"))
            if xy:
                gmm_positions[oid] = list(xy)

    return {
        "target": target,
        "target_ids": target_ids,
        "targets_self": targets_self,
        "related": list(related_info.values()),
        "gmm_scores": gmm_scores,
        "gmm_positions": gmm_positions,
    }


# ---------------------------------------------------------------------------
# Step 3: 生成 GMM 概率热图
# ---------------------------------------------------------------------------
def build_probability_field(
    gmm_scores: Dict[str, float],
    gmm_positions: Dict[str, List[float]],
    grid_dir: str,
    sigma_base: float = 1.0,
    score_amplify: float = 3.0,
    grid_array: Optional[np.ndarray] = None,
    grid_meta_override: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """在占据栅格上构建 2D 高斯混合概率场.

    Args:
        gmm_scores: {obj_id: score}
        gmm_positions: {obj_id: [world_x, world_z]}
        grid_dir: occupancy_grid 保存目录 (当 grid_array/grid_meta_override 为 None 时使用)
        sigma_base: 高斯核基础标准差 (m)
        score_amplify: 分数放大因子 (使低分信号可见)
        grid_array: 可选, 直接传入占据栅格数组 (避免从磁盘加载过时数据)
        grid_meta_override: 可选, 直接传入 {resolution, origin_x, origin_z}

    Returns:
        prob_field: [H, W] 概率场 (未归一化, 不受 free_mask 约束 — 物体本身在 OCCUPIED cell 上)
        grid: [H, W] 占据栅格 (0=unknown, 1=free, 2=occupied)
        grid_meta: 栅格元数据
    """
    # 加载栅格: 优先使用直接传入的活动栅格
    if grid_array is not None and grid_meta_override is not None:
        grid = grid_array
        resolution = float(grid_meta_override["resolution"])
        origin_x = float(grid_meta_override["origin_x"])
        origin_z = float(grid_meta_override["origin_z"])
    else:
        grid = np.load(os.path.join(grid_dir, "occupancy_grid.npy"))
        meta = dict(np.load(os.path.join(grid_dir, "occupancy_meta.npz")))
        resolution = float(meta["resolution"].item() if hasattr(meta["resolution"], "item") else meta["resolution"])
        origin = meta["origin"]
        origin_x = float(origin[0])
        origin_z = float(origin[1])
    H, W = grid.shape

    def world_to_grid(wx, wz):
        col = int(round((wx - origin_x) / resolution))
        row = int(round((wz - origin_z) / resolution))
        return row, col

    prob = np.zeros((H, W), dtype=np.float64)
    sigma_px = sigma_base / resolution  # 转像素

    n_placed = 0
    for oid, score in gmm_scores.items():
        pos = gmm_positions.get(oid)
        if pos is None or len(pos) < 2:
            continue
        r, c = world_to_grid(pos[0], pos[1])
        if r < 0 or r >= H or c < 0 or c >= W:
            continue

        amp = score * score_amplify
        # 计算有效窗口 (3σ 截断)
        win = int(math.ceil(3 * sigma_px))
        r0, r1 = max(0, r - win), min(H, r + win + 1)
        c0, c1 = max(0, c - win), min(W, c + win + 1)

        yy, xx = np.mgrid[r0:r1, c0:c1]
        g = amp * np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * sigma_px ** 2))
        prob[r0:r1, c0:c1] += g
        n_placed += 1

    # GMM 概率场不再受 free_mask 约束:
    # 物体本身就在 OCCUPIED cell 上 (桌子/椅子/冰箱等),
    # 概率场仅作为引导信号, 下游 frontier_exploration_step 中
    # score[~passable_mask] = -np.inf 保证导航目标只选可通行 cell.

    # 注意: 不再做全局归一化 (之前 prob /= total 导致 prob_max 太小,
    # frontier_exploration_step 的 flatness_ratio 判定几乎总判定为 flat)
    # 归一化仅在可视化时按需进行

    return prob, grid, {
        "resolution": resolution,
        "origin_x": origin_x,
        "origin_z": origin_z,
        "n_gaussians": n_placed,
        "n_scored_objects": len(gmm_scores),
    }


# ---------------------------------------------------------------------------
# Step 4: 可视化 + Top-K 候选点
# ---------------------------------------------------------------------------
def visualize_and_extract_topk(
    prob: np.ndarray,
    grid: np.ndarray,
    grid_meta: Dict,
    targets_self: List[Dict],
    target: str,
    output_dir: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """绘制热图并提取 Top-K 导航候选点."""
    H, W = prob.shape
    resolution = grid_meta["resolution"]
    origin_x = grid_meta["origin_x"]
    origin_z = grid_meta["origin_z"]

    def grid_to_world(r, c):
        wx = origin_x + c * resolution
        wz = origin_z + r * resolution
        return [round(wx, 3), round(wz, 3)]

    # --- 提取 Top-K 局部最大值 ---
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(prob, size=int(0.5 / resolution))  # 0.5m邻域
    peaks = (prob == local_max) & (prob > 0)
    peak_coords = np.argwhere(peaks)  # [N, 2] (row, col)
    peak_values = prob[peaks]

    # 按概率排序
    order = np.argsort(-peak_values)
    candidates = []
    for i in order[:top_k]:
        r, c = peak_coords[i]
        candidates.append({
            "rank": len(candidates) + 1,
            "grid_pos": [int(r), int(c)],
            "world_pos": grid_to_world(r, c),
            "probability": round(float(peak_values[i]), 6),
        })

    # --- 热图可视化 ---
    # 基础栅格 (灰度)
    base = np.zeros((H, W, 3), dtype=np.uint8)
    base[grid == 1] = [200, 200, 200]  # free = 浅灰
    base[grid == 2] = [40, 40, 40]     # occupied = 深灰
    # unknown 保持黑色

    # 概率叠加 (jet colormap)
    if prob.max() > 0:
        prob_norm = (prob / prob.max() * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(prob_norm, cv2.COLORMAP_JET)
        # 仅在有概率的区域叠加
        mask = prob_norm > 2
        alpha = 0.7
        for ch in range(3):
            base[:, :, ch] = np.where(
                mask,
                (alpha * heatmap[:, :, ch] + (1 - alpha) * base[:, :, ch]).astype(np.uint8),
                base[:, :, ch],
            )

    # 标注目标物体 (绿色圆)
    for t in targets_self:
        pos = _get_pos_xy(t.get("pos_2d"))
        if pos:
            wx, wz = pos
            c = int(round((wx - origin_x) / resolution))
            r = int(round((wz - origin_z) / resolution))
            if 0 <= r < H and 0 <= c < W:
                cv2.circle(base, (c, r), 6, (0, 255, 0), 2)
                cv2.putText(base, t.get("label", ""), (c + 8, r - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    # 标注 Top-K 候选点 (蓝色十字)
    for cand in candidates:
        r, c = cand["grid_pos"]
        cv2.drawMarker(base, (c, r), (255, 100, 0), cv2.MARKER_CROSS, 10, 2)
        cv2.putText(base, f"#{cand['rank']}", (c + 6, r - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 100, 0), 1)

    # 标题
    cv2.putText(base, f'Query: "{target}"  |  Top-{len(candidates)} candidates',
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    os.makedirs(output_dir, exist_ok=True)
    safe_target = target.replace(" ", "_").replace("/", "_")
    img_path = os.path.join(output_dir, f"query_{safe_target}_heatmap.png")
    cv2.imwrite(img_path, base)

    result = {
        "target": target,
        "candidates": candidates,
        "n_target_instances": len(targets_self),
        "n_gaussians": grid_meta["n_gaussians"],
        "n_scored_objects": grid_meta["n_scored_objects"],
        "heatmap_path": img_path,
    }
    json_path = os.path.join(output_dir, f"query_{safe_target}_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_query(
    target: str,
    config_path: Path = CONFIG_PATH,
    output_dir: str = "RAG_Graph/scene_build/queries",
    min_clip_score: Optional[float] = None,
    top_k: int = 5,
    sigma: float = 1.0,
) -> Dict[str, Any]:
    """完整查询管线."""
    cfg = _load_yaml(config_path)

    print(f"[1/4] CLIP 语义检索: \"{target}\" ...")
    clip_hits = clip_search_objects(target, cfg, min_score=min_clip_score)
    print(f"  → 匹配 {len(clip_hits)} 个物体实例")
    if clip_hits:
        for oid, sc in clip_hits[:5]:
            print(f"      {oid}: {sc:.4f}")
        if len(clip_hits) > 5:
            print(f"      ... ({len(clip_hits) - 5} more)")

    if not clip_hits:
        print("  [WARN] 未找到匹配物体，降低阈值重试...")
        clip_hits = clip_search_objects(target, cfg, min_score=0.5)
        if not clip_hits:
            print("  [ERROR] CLIP 完全无匹配")
            return {"target": target, "error": "no_clip_match"}

    print(f"\n[2/4] GMM 评分...")
    score_data = load_map_and_score(target, cfg, clip_hits)
    print(f"  → 目标实例: {len(score_data['target_ids'])}")
    print(f"  → 有分数物体: {len(score_data['gmm_scores'])}")
    if score_data["targets_self"]:
        for t in score_data["targets_self"][:3]:
            print(f"      {t['obj_id']}: self_score={t['self_score']:.4f}")

    grid_dir = (cfg.get("map_config") or {}).get("occupancy_grid_dir", "RAG_Graph/scene_build/occupancy_grid")
    print(f"\n[3/4] 构建概率场 (σ={sigma}m)...")
    prob, grid, grid_meta = build_probability_field(
        score_data["gmm_scores"],
        score_data["gmm_positions"],
        grid_dir,
        sigma_base=sigma,
    )
    print(f"  → 放置 {grid_meta['n_gaussians']} 个高斯核")
    print(f"  → 概率场最大值: {prob.max():.6f}")

    print(f"\n[4/4] 可视化 + Top-{top_k} 候选点...")
    result = visualize_and_extract_topk(
        prob, grid, grid_meta,
        score_data["targets_self"],
        target, output_dir, top_k=top_k,
    )

    print(f"\n{'='*50}")
    print(f"  查询完成: \"{target}\"")
    print(f"  热图: {result['heatmap_path']}")
    if result.get("candidates"):
        print(f"  Top-{len(result['candidates'])} 候选导航点:")
        for c in result["candidates"]:
            print(f"    #{c['rank']}: world={c['world_pos']} prob={c['probability']:.6f}")
    print(f"{'='*50}")
    return result


def main():
    parser = argparse.ArgumentParser(description="AgenticRAG 端到端查询测试")
    parser.add_argument("--target", type=str, required=True, help="查询目标 (如 chair, plant)")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--output-dir", type=str, default="RAG_Graph/scene_build/queries")
    parser.add_argument("--min-clip-score", type=float, default=None, help="CLIP 最低相似度 (默认用配置)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=1.0, help="高斯核标准差 (m)")
    args = parser.parse_args()

    run_query(
        target=args.target,
        config_path=Path(args.config),
        output_dir=args.output_dir,
        min_clip_score=args.min_clip_score,
        top_k=args.top_k,
        sigma=args.sigma,
    )


if __name__ == "__main__":
    main()
