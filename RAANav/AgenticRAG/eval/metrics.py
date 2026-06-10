"""P3: 评价指标计算模块.

指标体系:
  维度 A (记忆质量):   MRA, GHR@K
  维度 B (动态适应):   D-SR, AS, SSS
  维度 C (长期学习):   MRR
  传统参考:           SR, FR-Acc

用法:
    from eval.metrics import (
        compute_mra, compute_ghr_at_k, compute_d_sr,
        compute_as, compute_mrr, compute_all_metrics,
    )
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple


# ────────────── 维度 A: 记忆质量 ──────────────


def compute_mra(peak_pos: List[float], gt_pos: List[float],
                r_hit: float = 2.0) -> bool:
    """Memory Recall Accuracy: Top-1 峰值距 GT 位置 < r_hit.

    Args:
        peak_pos: GMM 概率场 Top-1 峰值 [x, z]
        gt_pos: GT 位置 [x, z]
        r_hit: 命中半径 (m)

    Returns:
        是否命中
    """
    if not peak_pos or not gt_pos or len(peak_pos) < 2 or len(gt_pos) < 2:
        return False
    dist = math.sqrt((peak_pos[0] - gt_pos[0]) ** 2 +
                     (peak_pos[1] - gt_pos[1]) ** 2)
    return dist < r_hit


def compute_mra_min_dist(peak_pos: List[float],
                         all_gt_pos: List[List[float]],
                         r_hit: float = 2.0) -> Tuple[bool, float]:
    """MRA 变体: 检查 Top-1 距任意同标签 GT 实例的最近距离.

    Returns:
        (is_hit, min_dist)
    """
    if not peak_pos or not all_gt_pos:
        return False, float('inf')
    min_dist = float('inf')
    for gt in all_gt_pos:
        if gt and len(gt) >= 2:
            d = math.sqrt((peak_pos[0] - gt[0]) ** 2 +
                          (peak_pos[1] - gt[1]) ** 2)
            min_dist = min(min_dist, d)
    return min_dist < r_hit, min_dist


def compute_ghr_at_k(peaks: List[List[float]],
                     all_gt_pos: List[List[float]],
                     k: int = 3,
                     r_hit: float = 2.0) -> bool:
    """GMM Hit Rate @ K: Top-K 峰值中是否有至少一个命中 GT.

    Args:
        peaks: Top-K 峰值坐标列表 [[x, z], ...]
        all_gt_pos: 所有同标签 GT 位置列表
        k: 考虑的峰值数量
        r_hit: 命中半径 (m)

    Returns:
        是否有至少一个峰值命中
    """
    if not peaks or not all_gt_pos:
        return False
    for peak in peaks[:k]:
        if peak and len(peak) >= 2:
            for gt in all_gt_pos:
                if gt and len(gt) >= 2:
                    d = math.sqrt((peak[0] - gt[0]) ** 2 +
                                  (peak[1] - gt[1]) ** 2)
                    if d < r_hit:
                        return True
    return False


# ────────────── 维度 B: 动态适应 ──────────────


def compute_d_sr(moved_results: List[bool]) -> float:
    """Dynamic Object Success Rate: 被移动物体的找到比例.

    Args:
        moved_results: 每个被移动物体是否被找到的列表

    Returns:
        成功率 [0, 1]
    """
    if not moved_results:
        return 0.0
    return sum(1 for r in moved_results if r) / len(moved_results)


def compute_as(mra_by_epoch: List[bool], moved_epoch: int) -> Optional[int]:
    """Adaptation Speed: 物体移动后需要多少轮 GMM Top-1 才指向新位置.

    Args:
        mra_by_epoch: 每个 epoch 的 MRA 结果列表 (从 epoch 0 开始)
        moved_epoch: 物体被移动的 epoch

    Returns:
        适应所需轮数, 或 None (如果一直未适应)
    """
    for n in range(len(mra_by_epoch)):
        epoch = moved_epoch + n
        if epoch < len(mra_by_epoch) and mra_by_epoch[epoch]:
            return n
    return None


def compute_sss(candidate_visits: int) -> int:
    """Semantic Search Steps: 机器人找到目标前访问的 GMM 高概率候选区域数量.

    SSS = 1 意味着一次命中.
    Note: 这个指标在轻量模拟中通过 GMM 多峰排序估算,
          完整导航模拟中由实际导航步记录.

    Args:
        candidate_visits: 访问的候选点数量

    Returns:
        SSS 值 (≥ 1)
    """
    return max(1, candidate_visits)


# ────────────── 维度 C: 长期学习 ──────────────


def compute_mrr(predictions: List[Dict]) -> Dict[str, float]:
    """Memory Retention Rate: 按 change_type 分组的记忆保持率.

    Args:
        predictions: 列表, 每项 {
            'obj_id': str,
            'change_type': str (SO/LRO/HRO/DO),
            'mra': bool,        # 盲预测是否命中
            'exist_prob': float, # 当前 exist_prob
        }

    Returns:
        {change_type: retention_rate} (每类的 MRA 平均值)
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for p in predictions:
        groups[p['change_type']].append(p['mra'])

    result = {}
    for ctype, hits in groups.items():
        result[ctype] = sum(1 for h in hits if h) / len(hits) if hits else 0.0
    result['overall'] = (sum(1 for p in predictions if p['mra']) /
                         len(predictions) if predictions else 0.0)
    return result


# ────────────── 传统指标 ──────────────


def compute_sr(results: List[bool]) -> float:
    """Success Rate: 整体成功率.

    Args:
        results: 每个查询是否成功

    Returns:
        成功率 [0, 1]
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r) / len(results)


def compute_fr_acc(predictions: List[bool]) -> float:
    """FR-Acc: LLM Agent 首选房间准确率.

    Args:
        predictions: 每个查询的 Agent 预测房间是否正确

    Returns:
        准确率 [0, 1]
    """
    if not predictions:
        return 0.0
    return sum(1 for p in predictions if p) / len(predictions)


# ────────────── 汇总接口 ──────────────


def compute_all_metrics(query_results: List[Dict],
                        r_hit: float = 2.0) -> Dict:
    """从一批查询结果中计算所有指标.

    Args:
        query_results: 列表, 每项 {
            'label': str,
            'change_type': str,            # SO/LRO/HRO/DO/NO
            'peaks': List[List[float]],    # GMM 概率场 Top-K 峰值 [x, z]
            'gt_positions': List[List[float]],  # 同标签所有 GT 位置
            'found': bool,                 # 是否找到 (导航/SSS 用)
            'sss': int,                    # SSS (候选点访问数)
            'moved': bool,                 # 本 epoch 是否被移动
            'moved_epoch': Optional[int],  # 被移动的 epoch
            'mra_by_epoch': Optional[List[bool]],  # 历史 MRA (AS 用)
        }
        r_hit: 命中半径

    Returns:
        汇总指标 dict
    """
    # MRA
    mra_hits = []
    for q in query_results:
        peaks = q.get('peaks', [])
        gt_positions = q.get('gt_positions', [])
        if peaks:
            hit, _ = compute_mra_min_dist(peaks[0], gt_positions, r_hit)
            mra_hits.append(hit)
        else:
            mra_hits.append(False)

    # GHR@3, GHR@5
    ghr3_hits = []
    ghr5_hits = []
    for q in query_results:
        peaks = q.get('peaks', [])
        gt = q.get('gt_positions', [])
        ghr3_hits.append(compute_ghr_at_k(peaks, gt, k=3, r_hit=r_hit))
        ghr5_hits.append(compute_ghr_at_k(peaks, gt, k=5, r_hit=r_hit))

    # D-SR (仅 moved 物体)
    moved_found = [q['found'] for q in query_results if q.get('moved')]

    # SSS
    sss_values = [q.get('sss', 1) for q in query_results if q.get('found')]

    # AS (仅 moved 物体且有 mra_by_epoch)
    as_values = []
    for q in query_results:
        if q.get('moved') and q.get('mra_by_epoch') is not None:
            me = q.get('moved_epoch', 0)
            a = compute_as(q['mra_by_epoch'], me)
            if a is not None:
                as_values.append(a)

    # SR
    all_found = [q.get('found', False) for q in query_results]

    return {
        'MRA': sum(mra_hits) / len(mra_hits) if mra_hits else 0.0,
        'GHR@3': sum(ghr3_hits) / len(ghr3_hits) if ghr3_hits else 0.0,
        'GHR@5': sum(ghr5_hits) / len(ghr5_hits) if ghr5_hits else 0.0,
        'D-SR': compute_d_sr(moved_found),
        'SSS_mean': sum(sss_values) / len(sss_values) if sss_values else 0.0,
        'AS_mean': sum(as_values) / len(as_values) if as_values else None,
        'SR': compute_sr(all_found),
        'n_queries': len(query_results),
        'n_moved': len(moved_found),
        'n_mra_hit': sum(mra_hits),
    }
