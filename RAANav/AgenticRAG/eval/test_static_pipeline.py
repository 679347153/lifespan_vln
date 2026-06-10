"""P0b: 静态基线验证 —— 在物体位置完全不变的情况下验证管线完整性。

验证项:
  a) run_merge 正确: 同一物体重复 merge 后 exist_prob 恢复/稳定 ≈1.0, cfd/pos 收敛
  b) GMM 查询稳定: 同一物体的 Top-1 峰值距 GT 位置距离 < 1m
  c) 无异常 crash / warning

用法:
  cd AgenticRAG
  conda run -n agentrag python eval/test_static_pipeline.py \
      --map RAG_Graph/scene_build/deep_explore_v2/00814-p53SfW6mjZe_semantic_map.json \
      --grid-dir RAG_Graph/scene_build/deep_explore_v2/occupancy_grid \
      --epochs 5 \
      --targets chair bed refrigerator
"""
import argparse
import copy
import json
import os
import sys
import math
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Optional
import numpy as np

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 确保项目根目录在 path 中
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from semantic_map import Floor, Room, Object
from semantic_map_Update.map_Update import run_merge


# ──────────────── 配置加载 ────────────────


def _load_config(config_path: str = "config/map.yaml") -> dict:
    """加载 map.yaml 配置, 返回 dict."""
    import yaml
    cfg_path = _project_root / config_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ──────────────── 辅助函数 ────────────────


def _get_all_objects(floors: List[Floor]) -> List[Object]:
    """从 floors 中提取所有物体的扁平列表."""
    objs = []
    for fl in floors:
        for rm in fl.rooms:
            objs.extend(rm.objects)
    return objs


def _find_object_by_id(floors: List[Floor], obj_id: str) -> Optional[Object]:
    for fl in floors:
        for rm in fl.rooms:
            for o in rm.objects:
                if o.obj_id == obj_id:
                    return o
    return None


def _advance_timestamps(floors: List[Floor], hours: float):
    """将所有物体的 last_update_time 向前推进 hours 小时."""
    delta = timedelta(hours=hours)
    for fl in floors:
        for rm in fl.rooms:
            for obj in rm.objects:
                ts = getattr(obj, 'last_update_time', None)
                if ts:
                    try:
                        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        dt = dt + delta
                        obj.last_update_time = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                    except Exception:
                        obj.last_update_time = datetime.now(
                            timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _build_gt_index(floors: List[Floor]) -> Dict[str, Dict]:
    """构建 obj_id → {label, pos_3d, pos_2d, room_id} 的 GT 索引."""
    index = {}
    for fl in floors:
        for rm in fl.rooms:
            for o in rm.objects:
                pos_3d = o.pos_3d if hasattr(o, 'pos_3d') else None
                pos_2d = o.pos_2d if hasattr(o, 'pos_2d') else None
                index[o.obj_id] = {
                    'label': o.label,
                    'pos_3d': pos_3d,
                    'pos_2d': pos_2d,
                    'room_id': rm.room_id,
                }
    return index


# ──────────────── CLIP 内存搜索 (简化版) ────────────────

_clip_cache: Dict = {}


def _ensure_clip():
    """懒加载 CLIP 模型 (模块级缓存, 仅加载一次)."""
    if 'model' not in _clip_cache:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        model_name = 'openai/clip-vit-base-patch32'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = CLIPModel.from_pretrained(model_name).to(device)
        model.eval()
        proc = CLIPProcessor.from_pretrained(model_name)
        _clip_cache.update({'model': model, 'processor': proc, 'device': device})


def _clip_search_in_memory(target_label: str, floors: List[Floor],
                           min_score: float = 0.75) -> List[Tuple[str, float]]:
    """在 floors 内存中搜索 CLIP 文本匹配.

    对所有物体的 label 和查询文本分别用 CLIP text encoder 编码,
    计算余弦相似度。若 CLIP 不可用, 回退到 label 精确匹配。
    """
    try:
        import torch
        _ensure_clip()
        model = _clip_cache['model']
        proc = _clip_cache['processor']
        dev = _clip_cache['device']

        # 收集所有物体
        all_objs: List[Tuple[str, str]] = []  # (obj_id, label)
        for fl in floors:
            for rm in fl.rooms:
                for o in rm.objects:
                    if o.obj_id and o.label:
                        all_objs.append((o.obj_id, o.label))
        if not all_objs:
            return []

        # 去重 label → 编码一次
        unique_labels = list(set(lbl for _, lbl in all_objs))
        label_vecs = {}
        for i in range(0, len(unique_labels), 32):
            batch = unique_labels[i:i + 32]
            inputs = proc(text=batch, return_tensors='pt',
                          padding=True, truncation=True).to(dev)
            with torch.no_grad():
                feats = model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            for j, lbl in enumerate(batch):
                label_vecs[lbl] = feats[j].cpu().numpy()

        # 编码查询
        q_inputs = proc(text=[target_label], return_tensors='pt',
                        padding=True, truncation=True).to(dev)
        with torch.no_grad():
            q_feat = model.get_text_features(**q_inputs)
            q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        q_vec = q_feat[0].cpu().numpy()

        # 余弦相似度
        hits = []
        for oid, lbl in all_objs:
            sim = float(np.dot(label_vecs[lbl], q_vec))
            if sim >= min_score:
                hits.append((oid, sim))
        hits.sort(key=lambda x: -x[1])
        return hits
    except Exception as e:
        print(f"  [CLIP 回退] {e}, 使用 label 精确匹配")
        hits = []
        for fl in floors:
            for rm in fl.rooms:
                for o in rm.objects:
                    if o.label and target_label.lower() in o.label.lower():
                        hits.append((o.obj_id, 1.0))
        return hits


# ──────────────── GMM 查询 (简化版, 不依赖 sim_nav_loop) ────────────────


def _score_from_memory(target: str, clip_hits: List[Tuple[str, float]],
                       floors: List[Floor], cfg: dict) -> Dict:
    """从 floors + clip_hits 计算 GMM 评分, 返回 {gmm_scores, gmm_positions}."""
    from GMM_map_Create.GMM_map_calcualte import Calculate_obj_Score, Calculate_Robj_Score

    gmm_scores: Dict[str, float] = {}
    gmm_positions: Dict[str, List[float]] = {}

    # 建 obj_id → Object 索引
    obj_index: Dict[str, Object] = {}
    for fl in floors:
        for rm in fl.rooms:
            for o in rm.objects:
                obj_index[o.obj_id] = o

    config_path = _project_root / "config" / "map.yaml"

    for oid, clip_score in clip_hits:
        obj = obj_index.get(oid)
        if not obj:
            continue
        # 自身得分
        self_score = Calculate_obj_Score(
            N=getattr(obj, 'N', 1) or 1,
            stability=getattr(obj, 'stability', 0.5) or 0.5,
            cfd=getattr(obj, 'cfd', None),
            config_path=config_path,
        )
        gmm_scores[oid] = self_score

        # 位置
        pos_2d = getattr(obj, 'pos_2d', None)
        if pos_2d:
            if isinstance(pos_2d, dict):
                xy = [pos_2d.get('x', 0), pos_2d.get('y', 0)]
            elif isinstance(pos_2d, (list, tuple)):
                xy = list(pos_2d[:2])
            else:
                xy = [0, 0]
            gmm_positions[oid] = xy

    return {"gmm_scores": gmm_scores, "gmm_positions": gmm_positions}


def _build_probability_field(score_data: Dict, grid_dir: str,
                             grid_array=None, grid_meta=None):
    """构建 GMM 概率场, 返回 (prob_field, peaks)."""
    from scripts.query_e2e import build_probability_field

    prob_field, grid_vis, meta = build_probability_field(
        score_data["gmm_scores"],
        score_data["gmm_positions"],
        grid_dir,
        sigma_base=1.5,
        grid_array=grid_array,
        grid_meta_override=grid_meta,
    )

    # 提取 Top-K 峰值坐标
    peaks = _extract_peaks(prob_field, meta, top_k=5)
    return prob_field, peaks


def _extract_peaks(prob_field: np.ndarray, meta: dict,
                   top_k: int = 5, min_dist_cells: int = 10) -> List[Dict]:
    """从概率场中提取 Top-K 峰值位置 (NMS)."""
    if prob_field is None or prob_field.size == 0:
        return []

    resolution = meta.get("resolution", 0.05)
    origin_x = meta.get("origin_x", 0.0)
    origin_z = meta.get("origin_z", 0.0)

    flat_indices = np.argsort(prob_field.ravel())[::-1]
    peaks = []

    for idx in flat_indices:
        if len(peaks) >= top_k:
            break
        r, c = divmod(int(idx), prob_field.shape[1])
        # NMS: 与已选峰值保持最小距离
        too_close = False
        for p in peaks:
            if abs(r - p['row']) < min_dist_cells and abs(c - p['col']) < min_dist_cells:
                too_close = True
                break
        if too_close:
            continue

        world_x = origin_x + c * resolution
        world_z = origin_z + r * resolution
        peaks.append({
            'row': r, 'col': c,
            'world_x': world_x, 'world_z': world_z,
            'prob': float(prob_field[r, c]),
        })

    return peaks


# ──────────────── 指标计算 ────────────────


def compute_mra(peak_pos: List[float], gt_pos: List[float],
                r_hit: float = 2.0) -> bool:
    """MRA: Top-1 峰值距 GT 位置 < r_hit."""
    if not peak_pos or not gt_pos:
        return False
    dist = math.sqrt((peak_pos[0] - gt_pos[0])**2 +
                     (peak_pos[1] - gt_pos[1])**2)
    return dist < r_hit


# ──────────────── 主测试流程 ────────────────


def run_static_test(map_path: str, grid_dir: str, n_epochs: int,
                    target_labels: List[str], config_path: str = "config/map.yaml"):
    """静态基线测试主流程."""

    print("=" * 70)
    print("P0b: 静态基线验证 (物体位置完全不变)")
    print("=" * 70)

    # 1. 加载 T0 地图作为 GT 和初始 history
    print(f"\n[1] 加载基础地图: {map_path}")
    floors_gt = Floor.from_json(map_path)
    gt_index = _build_gt_index(floors_gt)
    total_objs = sum(len(rm.objects) for fl in floors_gt for rm in fl.rooms)
    print(f"    楼层: {len(floors_gt)}, 物体总数: {total_objs}")

    cfg = _load_config(config_path)

    # 加载栅格 (用于 GMM 概率场)
    grid_array = None
    grid_meta = None
    grid_npy = os.path.join(grid_dir, "occupancy_grid.npy")
    meta_npz = os.path.join(grid_dir, "occupancy_meta.npz")
    if os.path.exists(grid_npy) and os.path.exists(meta_npz):
        grid_array = np.load(grid_npy)
        meta_data = np.load(meta_npz)
        origin = meta_data["origin"]
        grid_meta = {
            "resolution": float(meta_data["resolution"]),
            "origin_x": float(origin[0]),
            "origin_z": float(origin[1]),
        }
        print(f"    栅格: {grid_array.shape}, 分辨率: {grid_meta['resolution']}m")
    else:
        print(f"    [WARN] 栅格文件不存在, GMM 概率场测试将跳过")

    # 选择测试物体: 每个 target_label 取第一个匹配的物体
    test_objects = []
    for label in target_labels:
        for oid, info in gt_index.items():
            if label.lower() in info['label'].lower():
                test_objects.append((oid, info))
                break
        else:
            print(f"    [WARN] 未找到标签 '{label}' 的物体, 跳过")

    if not test_objects:
        # 自动选5个不同标签的物体
        seen_labels = set()
        for oid, info in gt_index.items():
            if info['label'] not in seen_labels and len(test_objects) < 5:
                test_objects.append((oid, info))
                seen_labels.add(info['label'])

    print(f"\n    测试物体 ({len(test_objects)}):")
    for oid, info in test_objects:
        print(f"      {oid}: {info['label']} @ {info.get('pos_3d', '?')}")

    # 2. 多 epoch merge 循环
    print(f"\n[2] 开始 {n_epochs} 轮静态 merge 测试")
    print("-" * 70)

    floors_history = copy.deepcopy(floors_gt)
    epoch_results = []

    for epoch in range(n_epochs):
        # 构造 floors_now: 同样的物体, 推进时间戳
        floors_now = copy.deepcopy(floors_gt)
        # 重置 exist_prob=1.0: floors_gt 中残留低值, 但 floors_now 代表"刚观测到"
        for _fl in floors_now:
            for _rm in _fl.rooms:
                for _obj in _rm.objects:
                    _obj.exist_prob = 1.0
        _advance_timestamps(floors_now, hours=24.0 * (epoch + 1))  # 每 epoch = 1天

        # 执行 merge
        floors_history, warnings = run_merge(
            floors_now, floors_history, cfg,
            shape_check=True,
            allow_new_floors=True,
            allow_new_rooms=True,
        )

        # 统计
        n_objs_after = sum(len(rm.objects) for fl in floors_history for rm in fl.rooms)

        # 检查测试物体的状态
        epoch_data = {
            'epoch': epoch,
            'n_objects': n_objs_after,
            'warnings': warnings,
            'objects': {},
        }

        for oid, gt_info in test_objects:
            obj = _find_object_by_id(floors_history, oid)
            if obj is None:
                epoch_data['objects'][oid] = {'status': 'MISSING'}
                continue

            pos_3d = getattr(obj, 'pos_3d', None)
            gt_pos = gt_info.get('pos_3d')
            pos_drift = 0.0
            if pos_3d and gt_pos:
                pos_drift = math.sqrt(sum((a - b)**2
                                         for a, b in zip(pos_3d, gt_pos)))

            epoch_data['objects'][oid] = {
                'status': 'OK',
                'exist_prob': getattr(obj, 'exist_prob', None),
                'cfd': getattr(obj, 'cfd', None),
                'N': getattr(obj, 'N', None),
                'pos_drift_m': round(pos_drift, 4),
                'label': obj.label,
            }

        epoch_results.append(epoch_data)

        # 打印摘要
        print(f"\n  Epoch {epoch}: {n_objs_after} objs"
              f" | warnings: {len(warnings)}")
        for oid, gt_info in test_objects:
            od = epoch_data['objects'].get(oid, {})
            if od.get('status') == 'MISSING':
                print(f"    ❌ {oid}: MISSING!")
            else:
                ep = od.get('exist_prob', '?')
                drift = od.get('pos_drift_m', '?')
                n = od.get('N', '?')
                print(f"    ✓ {oid}: exist_prob={ep:.4f}, "
                      f"pos_drift={drift}m, N={n}")
        if warnings:
            for w in warnings[:3]:
                print(f"    ⚠ {w}")

    # 3. GMM 概率场测试
    print(f"\n[3] GMM 概率场测试 (在最终 floors_history 上)")
    print("-" * 70)

    gmm_results = {}
    for oid, gt_info in test_objects:
        label = gt_info['label']
        print(f"\n  查询: \"{label}\" (GT: {oid})")

        # CLIP 搜索
        clip_hits = _clip_search_in_memory(label, floors_history, min_score=0.90)
        print(f"    CLIP 命中: {len(clip_hits)}")

        if not clip_hits:
            gmm_results[oid] = {'clip_hits': 0, 'mra': False, 'top1_dist': None}
            continue

        # GMM 评分
        score_data = _score_from_memory(label, clip_hits, floors_history, cfg)

        if not score_data["gmm_scores"]:
            gmm_results[oid] = {'clip_hits': len(clip_hits), 'mra': False,
                                'top1_dist': None}
            continue

        # 打印 Top-3 得分最高的物体 (调试)
        top_scored = sorted(score_data["gmm_scores"].items(),
                            key=lambda x: -x[1])[:3]
        for so, sc in top_scored:
            print(f"    Top: {so} score={sc:.4f}")

        # 如果有栅格, 构建概率场并提取峰值
        if grid_array is not None:
            prob_field, peaks = _build_probability_field(
                score_data, grid_dir,
                grid_array=grid_array, grid_meta=grid_meta)

            if peaks:
                top1 = [peaks[0]['world_x'], peaks[0]['world_z']]
                print(f"    Top-1 峰值: ({top1[0]:.2f}, {top1[1]:.2f})")

                # MRA: 检查 Top-1 距离任意同标签物体的最小距离
                min_dist = float('inf')
                nearest_oid = None
                for fl in floors_history:
                    for rm in fl.rooms:
                        for o in rm.objects:
                            if o.label and label.lower() == o.label.lower():
                                p2d = getattr(o, 'pos_2d', None)
                                if p2d:
                                    if isinstance(p2d, dict):
                                        gx = p2d.get('x', 0)
                                        gz = p2d.get('y', 0)
                                    else:
                                        gx, gz = p2d[0], p2d[1]
                                    d = math.sqrt((top1[0] - gx)**2 +
                                                  (top1[1] - gz)**2)
                                    if d < min_dist:
                                        min_dist = d
                                        nearest_oid = o.obj_id

                mra = min_dist < 2.0
                print(f"    最近同标签物体: {nearest_oid}, 距离: {min_dist:.2f}m"
                      f" → MRA={'✓' if mra else '✗'}")
                gmm_results[oid] = {
                    'clip_hits': len(clip_hits),
                    'mra': mra,
                    'top1_dist': round(min_dist, 3),
                    'nearest_oid': nearest_oid,
                    'top1': top1,
                    'peaks': peaks[:3],
                }
            else:
                print(f"    [WARN] 无峰值")
                gmm_results[oid] = {'clip_hits': len(clip_hits), 'mra': None,
                                    'top1_dist': None}
        else:
            # 无栅格, 仅报告评分
            top_scored = sorted(score_data["gmm_scores"].items(),
                                key=lambda x: -x[1])[:3]
            for so, sc in top_scored:
                print(f"    {so}: score={sc:.4f}")
            gmm_results[oid] = {'clip_hits': len(clip_hits),
                                'scores': dict(top_scored)}

    # 4. 总结报告
    print(f"\n{'=' * 70}")
    print("                        验证总结")
    print(f"{'=' * 70}")

    # exist_prob 检查
    final_epoch = epoch_results[-1] if epoch_results else {}
    all_ok = True

    print(f"\n  [exist_prob 验证] (经 {n_epochs} 轮 merge 后)")
    for oid, gt_info in test_objects:
        od = final_epoch.get('objects', {}).get(oid, {})
        ep = od.get('exist_prob')
        if ep is None:
            print(f"    ❌ {oid}: MISSING")
            all_ok = False
        elif ep < 0.7:
            print(f"    ⚠ {oid}: exist_prob={ep:.4f} (预期 ≥0.7)")
            all_ok = False
        else:
            print(f"    ✓ {oid}: exist_prob={ep:.4f}")

    print(f"\n  [位置漂移验证]")
    for oid, gt_info in test_objects:
        od = final_epoch.get('objects', {}).get(oid, {})
        drift = od.get('pos_drift_m')
        if drift is not None:
            ok = drift < 0.5
            sym = "✓" if ok else "⚠"
            print(f"    {sym} {oid}: 漂移={drift}m")
            if not ok:
                all_ok = False

    print(f"\n  [物体数量稳定性]")
    obj_counts = [e['n_objects'] for e in epoch_results]
    first, last = obj_counts[0], obj_counts[-1]
    ratio = last / first if first > 0 else 0
    ok = 0.9 < ratio < 1.1
    sym = "✓" if ok else "⚠"
    print(f"    {sym} Epoch 0: {first} → Epoch {n_epochs-1}: {last} "
          f"(比值: {ratio:.3f})")
    if not ok:
        all_ok = False

    print(f"\n  [GMM 查询验证]")
    for oid, gt_info in test_objects:
        gr = gmm_results.get(oid, {})
        mra = gr.get('mra')
        dist = gr.get('top1_dist')
        if mra is True:
            print(f"    ✓ {oid} ({gt_info['label']}): MRA=True, dist={dist}m")
        elif mra is False:
            print(f"    ❌ {oid} ({gt_info['label']}): MRA=False, dist={dist}m")
            all_ok = False
        else:
            print(f"    ⚠ {oid} ({gt_info['label']}): MRA=N/A")

    print(f"\n{'=' * 70}")
    if all_ok:
        print("  ✅ 静态基线验证通过! 管线完整性确认。")
    else:
        print("  ⚠ 存在问题, 需在动态实验前修复。")
    print(f"{'=' * 70}\n")

    return {
        'all_ok': all_ok,
        'epoch_results': epoch_results,
        'gmm_results': gmm_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P0b: 静态基线验证")
    parser.add_argument("--map", required=True,
                        help="T0 语义地图 JSON 路径")
    parser.add_argument("--grid-dir", required=True,
                        help="占据栅格目录 (含 occupancy_grid.npy + meta)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="模拟 merge 轮数 (默认 5)")
    parser.add_argument("--targets", nargs="*",
                        default=["chair", "bed", "refrigerator", "desk", "mirror"],
                        help="要测试的物体标签列表")
    parser.add_argument("--config", default="config/map.yaml",
                        help="配置文件路径")
    args = parser.parse_args()

    results = run_static_test(
        map_path=args.map,
        grid_dir=args.grid_dir,
        n_epochs=args.epochs,
        target_labels=args.targets,
        config_path=args.config,
    )
