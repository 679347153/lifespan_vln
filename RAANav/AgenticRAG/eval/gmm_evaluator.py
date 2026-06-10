"""P4: GMM 评估器 —— 封装 CLIP 搜索 + GMM 评分 + 概率场构建, 用于批量评估.

将 test_static_pipeline.py 中的 CLIP 搜索和 GMM 评分逻辑抽取为可复用模块,
支持在任意 floors_history 上执行查询并返回结构化结果.

用法:
    from eval.gmm_evaluator import GMMEvaluator
    evaluator = GMMEvaluator(config_path, grid_dir)
    result = evaluator.query(target_label="chair", floors=floors_history)
    # result: {peaks, scores, clip_hits, top1_pos, ...}
"""
from __future__ import annotations
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from semantic_map import Floor, Object


# ────────────── CLIP 模型缓存 ──────────────

_clip_cache: Dict = {}


def _ensure_clip():
    """懒加载 CLIP 模型 (模块级缓存)."""
    if 'model' not in _clip_cache:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        model_name = 'openai/clip-vit-base-patch32'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = CLIPModel.from_pretrained(model_name).to(device)
        model.eval()
        proc = CLIPProcessor.from_pretrained(model_name)
        _clip_cache.update({'model': model, 'processor': proc, 'device': device})


def clip_search_in_memory(target_label: str, floors: List[Floor],
                          min_score: float = 0.90) -> List[Tuple[str, float]]:
    """在 floors 内存中用 CLIP text-to-text 余弦相似度搜索物体.

    Args:
        target_label: 查询文本 (物体标签)
        floors: Floor 对象列表
        min_score: 最低相似度阈值

    Returns:
        [(obj_id, similarity_score), ...] 降序排列
    """
    try:
        import torch
        _ensure_clip()
        model = _clip_cache['model']
        proc = _clip_cache['processor']
        dev = _clip_cache['device']

        all_objs: List[Tuple[str, str]] = []
        for fl in floors:
            for rm in fl.rooms:
                for o in rm.objects:
                    if o.obj_id and o.label:
                        all_objs.append((o.obj_id, o.label))
        if not all_objs:
            return []

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

        q_inputs = proc(text=[target_label], return_tensors='pt',
                        padding=True, truncation=True).to(dev)
        with torch.no_grad():
            q_feat = model.get_text_features(**q_inputs)
            q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        q_vec = q_feat[0].cpu().numpy()

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


# ────────────── GMM 评分 ──────────────


def score_from_memory(clip_hits: List[Tuple[str, float]],
                      floors: List[Floor],
                      config_path: str = "config/map.yaml") -> Dict:
    """从 floors + clip_hits 计算 GMM 评分.

    Returns:
        {gmm_scores: {obj_id: float}, gmm_positions: {obj_id: [x, z]}}
    """
    from GMM_map_Create.GMM_map_calcualte import (
        Calculate_obj_Score, Calculate_Robj_Score,
    )

    cfg_path = _project_root / config_path

    obj_index: Dict[str, Object] = {}
    for fl in floors:
        for rm in fl.rooms:
            for o in rm.objects:
                obj_index[o.obj_id] = o

    gmm_scores: Dict[str, float] = {}
    gmm_positions: Dict[str, List[float]] = {}

    for oid, clip_score in clip_hits:
        obj = obj_index.get(oid)
        if not obj:
            continue

        self_score = Calculate_obj_Score(
            N=getattr(obj, 'N', 1) or 1,
            stability=getattr(obj, 'stability', 0.5) or 0.5,
            cfd=getattr(obj, 'cfd', None),
            config_path=cfg_path,
        )
        gmm_scores[oid] = self_score

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


# ────────────── 概率场 + 峰值提取 ──────────────


def build_probability_field(score_data: Dict, grid_dir: str,
                            grid_array: np.ndarray = None,
                            grid_meta: dict = None) -> Tuple[np.ndarray, List[Dict]]:
    """构建 GMM 概率场并提取 Top-K 峰值.

    Returns:
        (prob_field, peaks)
        peaks: [{world_x, world_z, prob, row, col}, ...]
    """
    from scripts.query_e2e import build_probability_field as _build_pf

    prob_field, grid, meta = _build_pf(
        score_data["gmm_scores"],
        score_data["gmm_positions"],
        grid_dir,
        sigma_base=1.5,
        grid_array=grid_array,
        grid_meta_override=grid_meta,
    )

    peaks = _extract_peaks(prob_field, meta, top_k=5)
    return prob_field, peaks


def _extract_peaks(prob_field: np.ndarray, meta: dict,
                   top_k: int = 5, min_dist_cells: int = 10) -> List[Dict]:
    """从概率场中提取 Top-K 峰值 (NMS)."""
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


# ────────────── 主评估类 ──────────────


class GMMEvaluator:
    """封装 CLIP 搜索 + GMM 评分 + 概率场的评估器."""

    def __init__(self, config_path: str = "config/map.yaml",
                 grid_dir: str = None,
                 clip_min_score: float = 0.90):
        """
        Args:
            config_path: map.yaml 配置路径
            grid_dir: 占据栅格目录 (含 occupancy_grid.npy + meta)
            clip_min_score: CLIP 搜索最低分
        """
        self.config_path = config_path
        self.grid_dir = grid_dir
        self.clip_min_score = clip_min_score

        # 预加载栅格
        self.grid_array = None
        self.grid_meta = None
        if grid_dir:
            grid_npy = os.path.join(grid_dir, "occupancy_grid.npy")
            meta_npz = os.path.join(grid_dir, "occupancy_meta.npz")
            if os.path.exists(grid_npy) and os.path.exists(meta_npz):
                self.grid_array = np.load(grid_npy)
                meta_data = np.load(meta_npz)
                origin = meta_data["origin"]
                self.grid_meta = {
                    "resolution": float(meta_data["resolution"]),
                    "origin_x": float(origin[0]),
                    "origin_z": float(origin[1]),
                }

    def query(self, target_label: str,
              floors: List[Floor]) -> Dict:
        """对指定标签在 floors 上执行完整 GMM 查询.

        Returns:
            {
                'label': str,
                'clip_hits': int,
                'gmm_scores': {obj_id: score},
                'peaks': [{'world_x', 'world_z', 'prob'}, ...],
                'top1_pos': [x, z] or None,
            }
        """
        # 1. CLIP 搜索
        clip_hits = clip_search_in_memory(
            target_label, floors, min_score=self.clip_min_score)

        if not clip_hits:
            return {
                'label': target_label,
                'clip_hits': 0,
                'gmm_scores': {},
                'peaks': [],
                'top1_pos': None,
            }

        # 2. GMM 评分
        score_data = score_from_memory(
            clip_hits, floors, self.config_path)

        # 3. 概率场 + 峰值
        peaks = []
        if self.grid_array is not None and score_data["gmm_scores"]:
            _, peaks = build_probability_field(
                score_data, self.grid_dir,
                grid_array=self.grid_array,
                grid_meta=self.grid_meta)

        top1_pos = None
        if peaks:
            top1_pos = [peaks[0]['world_x'], peaks[0]['world_z']]

        return {
            'label': target_label,
            'clip_hits': len(clip_hits),
            'gmm_scores': score_data["gmm_scores"],
            'peaks': peaks,
            'top1_pos': top1_pos,
        }

    def batch_query(self, target_labels: List[str],
                    floors: List[Floor]) -> List[Dict]:
        """批量查询多个标签."""
        return [self.query(label, floors) for label in target_labels]

    def evaluate_against_gt(self, target_label: str,
                            floors: List[Floor],
                            gt_positions: List[List[float]],
                            r_hit: float = 2.0) -> Dict:
        """查询并与 GT 对比, 返回带 MRA/GHR 指标的结果.

        Args:
            target_label: 查询标签
            floors: 当前 floors_history
            gt_positions: 同标签所有 GT 位置 [[x, z], ...]
            r_hit: 命中半径

        Returns:
            包含查询结果 + MRA + GHR@3 + GHR@5
        """
        from eval.metrics import compute_mra_min_dist, compute_ghr_at_k

        result = self.query(target_label, floors)

        # MRA
        mra = False
        min_dist = float('inf')
        if result['top1_pos']:
            mra, min_dist = compute_mra_min_dist(
                result['top1_pos'], gt_positions, r_hit)

        # GHR@K
        peak_coords = [[p['world_x'], p['world_z']] for p in result['peaks']]
        ghr3 = compute_ghr_at_k(peak_coords, gt_positions, k=3, r_hit=r_hit)
        ghr5 = compute_ghr_at_k(peak_coords, gt_positions, k=5, r_hit=r_hit)

        result.update({
            'mra': mra,
            'min_dist': min_dist,
            'ghr3': ghr3,
            'ghr5': ghr5,
            'gt_positions': gt_positions,
        })
        return result
