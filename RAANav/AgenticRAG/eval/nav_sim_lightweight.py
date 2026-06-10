"""P5: 轻量导航模拟器 —— 模拟 SSS (Semantic Search Steps) 和负向高斯场效果。

不依赖 habitat-sim, 纯粹在概率场上模拟导航决策:
1. 对 GMM 概率场提取 Top-K 候选点 (扩展到 max_candidates)
2. 按得分降序 "访问" 每个候选点
3. 在候选点 r_hit 范围内检查是否有同标签 GT 物体
4. 记录 SSS = 第几个候选点找到目标
5. 可选: 注入负向高斯场 (模拟 false negative → 概率抑制 → 转移到下一候选)

用法:
    from eval.nav_sim_lightweight import LightweightNavSim
    sim = LightweightNavSim(evaluator, neg_field_enabled=True)
    result = sim.search("chair", floors, [[1.0, 2.0]])
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from semantic_map import Floor
from eval.gmm_evaluator import (
    GMMEvaluator,
    clip_search_in_memory,
    score_from_memory,
    _extract_peaks,
)


# ────────────── 负向高斯注入 (离线版) ──────────────

def _inject_negative_kernel(prob_field: np.ndarray, wx: float, wz: float,
                            grid_meta: dict, sigma_base: float = 1.5,
                            sigma_neg_ratio: float = 2.0,
                            neg_alpha: float = 0.3) -> np.ndarray:
    """在概率场指定世界坐标处叠加一个负向高斯核, 并 clamp ≥ 0."""
    resolution = grid_meta.get('resolution', 0.05)
    origin_x = grid_meta.get('origin_x', 0.0)
    origin_z = grid_meta.get('origin_z', 0.0)

    col = int(round((wx - origin_x) / resolution))
    row = int(round((wz - origin_z) / resolution))
    H, W = prob_field.shape

    sigma_m = sigma_base * sigma_neg_ratio
    sigma_px = sigma_m / resolution
    win = int(math.ceil(3 * sigma_px))

    r0, r1 = max(0, row - win), min(H, row + win + 1)
    c0, c1 = max(0, col - win), min(W, col + win + 1)

    yy, xx = np.mgrid[r0:r1, c0:c1]
    g = neg_alpha * np.exp(
        -((yy - row) ** 2 + (xx - col) ** 2) / (2 * sigma_px ** 2))

    out = prob_field.copy()
    out[r0:r1, c0:c1] -= g
    np.clip(out, 0, None, out=out)
    return out


# ────────────── 主类 ──────────────

class LightweightNavSim:
    """轻量导航模拟: 在 GMM 概率场上模拟候选点逐步访问."""

    def __init__(self, evaluator: GMMEvaluator, *,
                 neg_field_enabled: bool = True,
                 max_candidates: int = 10,
                 r_hit: float = 2.0,
                 sigma_base: float = 1.5,
                 sigma_neg_ratio: float = 2.0,
                 neg_alpha: float = 0.3):
        """
        Args:
            evaluator: GMMEvaluator 实例 (包含 grid_array/grid_meta)
            neg_field_enabled: 是否启用负向高斯场模拟
            max_candidates: 最大候选点搜索预算 (SSS 上限)
            r_hit: 命中半径 (m)
            sigma_base: 概率场高斯核基础 σ (m)
            sigma_neg_ratio: 负向核 σ = sigma_base * ratio
            neg_alpha: 负向核幅度
        """
        self.evaluator = evaluator
        self.neg_enabled = neg_field_enabled
        self.max_candidates = max_candidates
        self.r_hit = r_hit
        self.sigma_base = sigma_base
        self.sigma_neg_ratio = sigma_neg_ratio
        self.neg_alpha = neg_alpha

    # ── 核心搜索 ──

    def search(self, target_label: str, floors: List[Floor],
               gt_positions: List[List[float]]) -> Dict:
        """模拟对单个目标标签的搜索过程.

        Args:
            target_label: 查询标签
            floors: 当前 floors_history (merge 后)
            gt_positions: 同标签所有 GT 位置 [[x, z], ...]

        Returns:
            {
                'found': bool,
                'sss': int,
                'visited_points': list,
                'final_dist': float,
                'mra': bool,
                'label': str,
            }
        """
        # 1. CLIP 搜索 + GMM 评分 → 概率场 + 初始峰值
        prob_field, initial_peaks = self._build_prob_and_peaks(
            target_label, floors)

        if not initial_peaks:
            return self._empty_result(target_label)

        # 2. 逐候选点访问
        peaks = list(initial_peaks)     # 当前候选列表 (可变)
        visited: List[List[float]] = []
        sss = 0
        found = False
        found_dist = float('inf')

        for step in range(self.max_candidates):
            if step >= len(peaks):
                break
            pk = peaks[step]
            px, pz = pk['world_x'], pk['world_z']
            sss += 1
            visited.append([px, pz])

            # 检查是否命中任意 GT
            min_d = self._min_dist_to_gt(px, pz, gt_positions)
            if min_d < self.r_hit:
                found = True
                found_dist = min_d
                break

            # miss → 可选注入负向高斯 → 从修正场重新提取剩余候选
            if self.neg_enabled and prob_field is not None:
                prob_field = _inject_negative_kernel(
                    prob_field, px, pz, self.evaluator.grid_meta,
                    sigma_base=self.sigma_base,
                    sigma_neg_ratio=self.sigma_neg_ratio,
                    neg_alpha=self.neg_alpha)
                remaining = _extract_peaks(
                    prob_field, self.evaluator.grid_meta,
                    top_k=self.max_candidates - sss)
                # 替换尚未访问的候选
                peaks = peaks[:step + 1] + remaining

        # 3. MRA: Top-1 是否命中
        mra = self._check_mra(initial_peaks[0], gt_positions)

        return {
            'found': found,
            'sss': sss,
            'visited_points': visited,
            'final_dist': found_dist,
            'mra': mra,
            'label': target_label,
        }

    # ── 批量搜索 ──

    def batch_search(self, target_labels: List[str],
                     floors: List[Floor],
                     gt_by_label: Dict[str, List[List[float]]]) -> List[Dict]:
        """批量搜索多个标签."""
        results = []
        for label in target_labels:
            gt_positions = gt_by_label.get(label, [])
            results.append(self.search(label, floors, gt_positions))
        return results

    # ── 内部辅助 ──

    def _build_prob_and_peaks(self, target_label: str,
                              floors: List[Floor]
                              ) -> Tuple[Optional[np.ndarray], List[Dict]]:
        """构建概率场并提取扩展候选点."""
        if self.evaluator.grid_array is None:
            # 无栅格 → 退化为仅用 query() 的 peaks
            qr = self.evaluator.query(target_label, floors)
            return None, qr['peaks']

        clip_hits = clip_search_in_memory(
            target_label, floors, min_score=self.evaluator.clip_min_score)
        if not clip_hits:
            return None, []

        score_data = score_from_memory(clip_hits, floors,
                                       self.evaluator.config_path)
        if not score_data['gmm_scores']:
            return None, []

        from scripts.query_e2e import build_probability_field as _build_pf
        prob_field, _, meta = _build_pf(
            score_data['gmm_scores'],
            score_data['gmm_positions'],
            self.evaluator.grid_dir,
            sigma_base=self.sigma_base,
            grid_array=self.evaluator.grid_array,
            grid_meta_override=self.evaluator.grid_meta,
        )
        peaks = _extract_peaks(prob_field, meta,
                               top_k=self.max_candidates)
        return prob_field, peaks

    @staticmethod
    def _min_dist_to_gt(px: float, pz: float,
                        gt_positions: List[List[float]]) -> float:
        """候选点到最近 GT 的距离."""
        md = float('inf')
        for gt in gt_positions:
            if gt and len(gt) >= 2:
                d = math.sqrt((px - gt[0]) ** 2 + (pz - gt[1]) ** 2)
                md = min(md, d)
        return md

    def _check_mra(self, top1_peak: Dict,
                   gt_positions: List[List[float]]) -> bool:
        """Top-1 峰值是否在 r_hit 内命中 GT."""
        d = self._min_dist_to_gt(
            top1_peak['world_x'], top1_peak['world_z'], gt_positions)
        return d < self.r_hit

    def _empty_result(self, label: str) -> Dict:
        return {
            'found': False,
            'sss': self.max_candidates,
            'visited_points': [],
            'final_dist': float('inf'),
            'mra': False,
            'label': label,
        }
