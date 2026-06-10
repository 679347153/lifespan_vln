"""P7: 端到端实验运行器 —— 串联 ChangeScript → EpochSimulator → SSS → Metrics.

用法:
    # 命令行
    conda run -n agentrag python eval/run_experiment.py

    # Python
    from eval.run_experiment import run_experiment
    results = run_experiment(
        base_map_path="semantic_map/scene_00814.json",
        change_script_path="eval/scenarios/00814_changes.yaml",
        grid_dir="semantic_map/grid_00814",
        ablation="Full",
    )
"""
from __future__ import annotations
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import yaml

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from eval.change_script import ChangeScript
from eval.epoch_simulator import EpochSimulator
from eval.gmm_evaluator import GMMEvaluator
from eval.nav_sim_lightweight import LightweightNavSim
from eval.metrics import compute_all_metrics, compute_sss, compute_mrr
from eval.ablation import ABLATION_CONFIGS, apply_ablation, get_nav_sim_kwargs


# ────────────── 辅助 ──────────────

def _load_config(config_path: str) -> dict:
    """加载 map.yaml 配置."""
    p = Path(config_path)
    if not p.is_absolute():
        p = _project_root / p
    with open(p) as f:
        return yaml.safe_load(f)


def _build_gt_by_label(gt_layout: dict) -> Dict[str, List[List[float]]]:
    """从 GT 布局构建 {label: [[x, z], ...]} 映射."""
    out: Dict[str, List[List[float]]] = {}
    for obj in gt_layout.get('objects', []):
        label = obj.get('label', '')
        pos_2d = obj.get('pos_2d')
        if pos_2d and label:
            if isinstance(pos_2d, dict):
                coord = [pos_2d.get('x', 0), pos_2d.get('y', 0)]
            elif isinstance(pos_2d, (list, tuple)):
                coord = list(pos_2d[:2])
            else:
                continue
            out.setdefault(label, []).append(coord)
    return out


def _collect_query_labels(gt_layout: dict, cs: ChangeScript,
                          epoch: int) -> List[dict]:
    """收集本 epoch 应查询的标签及元数据.

    返回: [{label, change_type, obj_id, moved, ...}, ...]
    """
    queries = []
    # 变更脚本中的物体 (HRO/LRO/DO/NO)
    for c in cs.changes:
        if not c.is_present(epoch):
            continue
        moved = c.has_moved(epoch) if c.change_type in ('HRO', 'LRO') else False
        queries.append({
            'label': c.label,
            'change_type': c.change_type,
            'obj_id': c.obj_id,
            'moved': moved,
        })

    # 补充一些 SO 物体 (随机采样几个常见类别)
    changed_labels = {c.label for c in cs.changes}
    so_added = set()
    for obj in gt_layout.get('objects', []):
        lbl = obj.get('label', '')
        ct = obj.get('change_type', 'SO')
        if ct == 'SO' and lbl not in changed_labels and lbl not in so_added:
            queries.append({
                'label': lbl,
                'change_type': 'SO',
                'obj_id': obj.get('obj_id'),
                'moved': False,
            })
            so_added.add(lbl)
            if len(so_added) >= 5:
                break
    return queries


# ────────────── 核心实验函数 ──────────────

def run_experiment(base_map_path: str,
                   change_script_path: str,
                   grid_dir: str,
                   config_path: str = "config/map.yaml",
                   ablation: str = "Full",
                   r_hit: float = 2.0,
                   max_candidates: int = 10,
                   verbose: bool = True) -> Dict:
    """执行单场景完整实验.

    流程:
    1. 加载 ChangeScript + 配置
    2. 对指定消融条件应用配置覆盖
    3. 运行 EpochSimulator (多 epoch merge)
    4. 每 epoch: 对变化物体 + 部分 SO 物体运行 SSS 搜索
    5. 汇总所有指标

    Args:
        base_map_path: T0 语义地图 JSON
        change_script_path: 变更脚本 YAML
        grid_dir: 占据栅格目录
        config_path: map.yaml 路径
        ablation: 消融条件名称 (ABLATION_CONFIGS 的键)
        r_hit: 命中半径 (m)
        max_candidates: SSS 最大候选数
        verbose: 是否输出详情

    Returns:
        {
            'ablation': str,
            'scene_id': str,
            'n_epochs': int,
            'epoch_results': [per-epoch result dicts],
            'aggregate': {metrics},
        }
    """
    t0 = time.time()

    # 确保相对路径基于项目根目录
    _saved_cwd = os.getcwd()
    os.chdir(_project_root)

    # 1. 加载
    cs = ChangeScript.from_yaml(change_script_path)
    base_cfg = _load_config(config_path)
    cfg = apply_ablation(base_cfg, ablation)
    abl_spec = ABLATION_CONFIGS[ablation]

    if verbose:
        print(f"\n{'='*70}")
        print(f"Experiment: scene={cs.scene_id}  ablation={ablation}")
        print(f"  {abl_spec['description']}")
        print(f"{'='*70}")

    # 2. Epoch 模拟
    epoch_sim = EpochSimulator(base_map_path, cs, cfg)

    # 处理 stability 消融
    if not abl_spec['enable_stability']:
        _force_stability(epoch_sim, 0.5)

    sim_result = epoch_sim.run_all_epochs(verbose=verbose)

    # 3. 构建 GMMEvaluator + NavSim
    evaluator = GMMEvaluator(
        config_path=config_path,
        grid_dir=grid_dir,
        clip_min_score=0.90,
    )
    nav_kwargs = get_nav_sim_kwargs(ablation)
    nav_sim = LightweightNavSim(
        evaluator,
        max_candidates=max_candidates,
        r_hit=r_hit,
        **nav_kwargs,
    )

    # 4. 每 epoch 评估
    all_query_results = []
    epoch_results = []

    # 若禁用 merge: 每 epoch 用 floors_now 而非 floors_history
    if not abl_spec['enable_merge']:
        # 重跑, 保存每 epoch 的 floors_now
        epoch_floors_list = []
        for epoch in range(cs.n_epochs):
            floors_now = epoch_sim.generate_floors_now(epoch)
            epoch_floors_list.append(floors_now)
    else:
        epoch_floors_list = None

    # 需要逐 epoch 的 floors_history → 重跑 (简化: 逐步 merge)
    import copy as _copy
    from semantic_map import Floor
    from semantic_map_Update.map_Update import run_merge

    floors_hist = _copy.deepcopy(epoch_sim.floors_gt)
    for epoch in range(cs.n_epochs):
        # 决定用哪个 floors 做查询
        if epoch_floors_list is not None:
            query_floors = epoch_floors_list[epoch]
        else:
            # merge
            floors_now = epoch_sim.generate_floors_now(epoch)
            floors_hist, _ = run_merge(floors_now, floors_hist, cfg,
                                       shape_check=True,
                                       allow_new_floors=True,
                                       allow_new_rooms=True)
            query_floors = floors_hist

        gt_layout = sim_result['gt_layouts'][epoch]
        gt_by_label = _build_gt_by_label(gt_layout)
        queries = _collect_query_labels(gt_layout, cs, epoch)

        epoch_qr = []
        for q in queries:
            label = q['label']
            gt_pos = gt_by_label.get(label, [])

            # SSS 搜索
            nav_result = nav_sim.search(label, [f for f in (
                query_floors if isinstance(query_floors, list)
                else [query_floors])], gt_pos)

            # 拼装 compute_all_metrics 所需格式
            qr = {
                'label': label,
                'change_type': q['change_type'],
                'peaks': [[p[0], p[1]] for p in nav_result['visited_points']],
                'gt_positions': gt_pos,
                'found': nav_result['found'],
                'sss': nav_result['sss'],
                'moved': q.get('moved', False),
                'moved_epoch': epoch if q.get('moved') else None,
                'mra_by_epoch': None,  # 跨 epoch MRA 由后处理计算
            }
            epoch_qr.append(qr)

        # 汇总本 epoch 指标
        epoch_metrics = compute_all_metrics(epoch_qr, r_hit=r_hit)
        epoch_results.append({
            'epoch': epoch,
            'day': cs.epoch_days[epoch] if epoch < len(cs.epoch_days) else epoch,
            'n_queries': len(epoch_qr),
            'metrics': epoch_metrics,
        })
        all_query_results.extend(epoch_qr)

        if verbose:
            m = epoch_metrics
            print(f"  Epoch {epoch}: MRA={m['MRA']:.2f} GHR@3={m['GHR@3']:.2f} "
                  f"D-SR={m['D-SR']:.2f} SSS_mean={m['SSS_mean']:.1f} "
                  f"SR={m['SR']:.2f}")

    # 5. 全局汇总
    aggregate = compute_all_metrics(all_query_results, r_hit=r_hit)

    # MRR: 按 change_type 分组
    mrr_inputs = []
    for qr in all_query_results:
        peaks = qr.get('peaks', [])
        gt = qr.get('gt_positions', [])
        from eval.metrics import compute_mra_min_dist
        mra_hit = False
        if peaks:
            mra_hit, _ = compute_mra_min_dist(peaks[0], gt, r_hit)
        mrr_inputs.append({
            'obj_id': qr.get('label', ''),
            'change_type': qr['change_type'],
            'mra': mra_hit,
            'exist_prob': 1.0,
        })
    mrr = compute_mrr(mrr_inputs)
    aggregate['MRR'] = mrr

    elapsed = time.time() - t0

    result = {
        'ablation': ablation,
        'scene_id': cs.scene_id,
        'n_epochs': cs.n_epochs,
        'epoch_results': epoch_results,
        'aggregate': aggregate,
        'elapsed_sec': elapsed,
    }

    if verbose:
        print(f"\n{'─'*50}")
        print(f"Aggregate ({ablation}):")
        for k, v in aggregate.items():
            if isinstance(v, float):
                print(f"  {k:12s}: {v:.4f}")
            elif isinstance(v, dict):
                print(f"  {k:12s}: {v}")
            else:
                print(f"  {k:12s}: {v}")
        print(f"  elapsed     : {elapsed:.1f}s")
        print(f"{'─'*50}")

    os.chdir(_saved_cwd)
    return result


def _force_stability(epoch_sim: EpochSimulator, val: float):
    """将 floors_gt 中所有物体的 stability 设为固定值."""
    for fl in epoch_sim.floors_gt:
        for rm in fl.rooms:
            for obj in rm.objects:
                obj.stability = val


# ────────────── 比较实验 ──────────────

def run_ablation_comparison(base_map_path: str,
                            change_script_path: str,
                            grid_dir: str,
                            config_path: str = "config/map.yaml",
                            ablations: List[str] = None,
                            verbose: bool = True) -> Dict[str, Dict]:
    """对同一场景运行多个消融条件并汇总对比.

    Args:
        ablations: 要运行的消融条件列表, 默认全部

    Returns:
        {ablation_name: result_dict, ...}
    """
    if ablations is None:
        ablations = list(ABLATION_CONFIGS.keys())

    all_results = {}
    for abl_name in ablations:
        print(f"\n{'#'*70}")
        print(f"# Running ablation: {abl_name}")
        print(f"{'#'*70}")
        result = run_experiment(
            base_map_path=base_map_path,
            change_script_path=change_script_path,
            grid_dir=grid_dir,
            config_path=config_path,
            ablation=abl_name,
            verbose=verbose,
        )
        all_results[abl_name] = result

    # 打印对比表格
    if verbose:
        _print_comparison_table(all_results)

    return all_results


def _print_comparison_table(results: Dict[str, Dict]):
    """打印消融对比表格."""
    metrics_keys = ['MRA', 'GHR@3', 'GHR@5', 'D-SR', 'SSS_mean', 'SR']

    print(f"\n{'='*80}")
    print(f"{'Ablation Comparison':^80}")
    print(f"{'='*80}")

    # 表头
    header = f"{'Ablation':18s}"
    for mk in metrics_keys:
        header += f" {mk:>8s}"
    print(header)
    print('─' * 80)

    # 每行
    for abl_name, result in results.items():
        agg = result.get('aggregate', {})
        row = f"{abl_name:18s}"
        for mk in metrics_keys:
            val = agg.get(mk, 0)
            if isinstance(val, float):
                row += f" {val:8.4f}"
            else:
                row += f" {val!s:>8s}"
        print(row)

    print(f"{'='*80}\n")


# ────────────── CLI ──────────────

def main():
    """命令行入口: 使用默认路径运行完整实验."""
    import argparse
    parser = argparse.ArgumentParser(description="运行消融实验")
    parser.add_argument("--base-map",
                        default="RAG_Graph/task_output/00814_chair/map_final.json",
                        help="T0 语义地图 JSON")
    parser.add_argument("--change-script",
                        default="eval/scenarios/00814_changes.yaml",
                        help="变更脚本 YAML")
    parser.add_argument("--grid-dir",
                        default="RAG_Graph/task_output/00814_chair/occupancy",
                        help="占据栅格目录")
    parser.add_argument("--config", default="config/map.yaml",
                        help="map.yaml 配置")
    parser.add_argument("--ablation", default=None,
                        help="指定消融条件 (默认运行全部)")
    parser.add_argument("--output", default=None,
                        help="结果输出 JSON 路径")
    parser.add_argument("--quiet", action="store_true",
                        help="减少输出")
    args = parser.parse_args()

    # 切换到项目根目录
    os.chdir(_project_root)

    if args.ablation:
        result = run_experiment(
            base_map_path=args.base_map,
            change_script_path=args.change_script,
            grid_dir=args.grid_dir,
            config_path=args.config,
            ablation=args.ablation,
            verbose=not args.quiet,
        )
        results = {args.ablation: result}
    else:
        results = run_ablation_comparison(
            base_map_path=args.base_map,
            change_script_path=args.change_script,
            grid_dir=args.grid_dir,
            config_path=args.config,
            verbose=not args.quiet,
        )

    if args.output:
        # 序列化 (跳过不可 JSON 化的字段)
        def _default(obj):
            if isinstance(obj, (set, frozenset)):
                return list(obj)
            return str(obj)

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False,
                      default=_default)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
