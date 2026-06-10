import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from GMM_map_Create import utility
from CLIP_RAG.query_clip_index import load_index as clip_load_index, encode_query as clip_encode_query, search as clip_search

CONFIG_PATH = Path("config/map.yaml")
DEFAULT_MERGED_MAP = Path("RAG_Graph/test_save/map_mergedALL.json")

#############################
# 配置加载与通用工具
#############################

def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    import yaml  # 局部导入，避免无依赖时报错
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def _load_merged_map(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    map_cfg = cfg.get('map_config') or {}
    # 优先使用显式 JSON 路径
    explicit = map_cfg.get('map_merged_json')
    if explicit and Path(explicit).exists():
        path = Path(explicit)
    else:
        merged_dir = map_cfg.get('map_merged_dir')
        if merged_dir:
            p = Path(merged_dir)
            if p.is_dir():
                candidate = p / 'map_merged.json'
                path = candidate if candidate.exists() else DEFAULT_MERGED_MAP
            else:
                path = DEFAULT_MERGED_MAP
        else:
            path = DEFAULT_MERGED_MAP
    if not path.exists():
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def _index_objects(floors: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for fl in floors:
        for room in fl.get('rooms', []):
            for obj in room.get('objects', []):
                idx[obj.get('obj_id')] = obj 
                #HACK 从字典里面读出来所以obj确实时DICT
                #虽然在对象里面他是objects: Optional[List[Object]] = None,
    return idx

def _read_Rscore_output_for_target(cfg: Dict[str, Any], target: str) -> Dict[str, float]:
    """
    读取指定目标{target}的 Rscore 输出结果。
    """
    agent_cfg = (cfg.get('agent_Rscore') or {})
    out_path_template = agent_cfg.get('output_path')
    if not isinstance(out_path_template, str) or not out_path_template.strip() or out_path_template.strip().upper() == 'NONE':
        return {}
    # 解析模板路径
    try:
        # HACK: 处理路径中的 {target} 占位符
        path_str = out_path_template.format(target=target) if '{target}' in out_path_template else out_path_template
    except Exception:
        path_str = out_path_template
    #HACK：双重保险
    p = Path(path_str)
    # 若为目录则拼接文件名
    if p.is_dir():
        p = p / f"{target}_Robj_score.json"
    if not p.exists():
        return {}
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 新结构中包含 Robj_scores 最小映射
        scores = data.get('Robj_scores') or {}
        return {k: float(v) for k, v in scores.items()}
    except Exception:
        return {}

def _try_cuda() -> bool:
    try:
        import torch  # type: ignore
        return torch.cuda.is_available()
    except Exception:
        return False


def _get_rscore_norm_max(cfg: Dict[str, Any]) -> float:
    agent_cfg = cfg.get('agent_Rscore') or {}
    obj_cfg = cfg.get('object') or {}
    val = agent_cfg.get('RscoreValue_max', obj_cfg.get('Rscore_max', 1.0))
    try:
        v = float(val)
    except Exception:
        v = 1.0
    return v if v > 0 else 1.0

# 安全
def _clip_search_targets(cfg: Dict[str, Any], query: str) -> List[Tuple[str, float]]:
    """使用 CLIP 索引与阈值挑选属于目标的所有实例。
    返回: [(obj_id, score)]
    """
    map_cfg = cfg.get('map_config') or {}
    index_dir = map_cfg.get('CLIP_RAG_map_dir')
    if not index_dir:
        return []
    p = Path(index_dir)
    if not p.exists():
        return []
    ids, vecs, _texts, meta = clip_load_index(p)
    device = 'cuda' if _try_cuda() else 'cpu'
    # 可能在离线环境，编码失败则回退到基于 ID 名称的标签匹配
    try:
        qv = clip_encode_query([query], meta['model'], device=device) # 上面自定义的函数
        use_vector = True
    except Exception:
        use_vector = False
    clip_cfg = cfg.get('CLIP_RAG') or {}
    min_score = clip_cfg.get('consine_similarity_filmin')
    thr = float(min_score) if isinstance(min_score, (int, float)) else None
    if use_vector:
        results = clip_search(vecs, ids, qv, top_k=None, min_score=thr)
        return [(oid, sc) for _rank, oid, sc, _idx in results]
    
    # HACK fallback: 通过 obj_id 前缀标签匹配
    def infer_label(oid: str) -> str:
        parts = oid.split('_')
        return '_'.join(parts[:-2]) if len(parts) >= 3 else oid
    pairs: List[Tuple[str, float]] = []
    for oid in ids:
        if infer_label(oid) == query:
            pairs.append((oid, 1.0))  # 视为命中
    return pairs

#HACK 写的好，值得学习思维
def compute_pair_relation_stats(target_obj: Dict[str, Any], related_obj: Dict[str, Any]) -> Tuple[int, int, float, float]:
    """计算 (Nr, N_target, Rcfd, Nr_over_N) 基于 target_obj.R_objs 映射。"""
    R_map = target_obj.get('R_objs') or {}

    entry = R_map.get(related_obj.get('obj_id')) if isinstance(R_map, dict) else None
    #TODO 这里可以换成平均试一试
    N_target = int(target_obj.get('N', 1) or 1)
    Nr = int((entry or {}).get('Nr', 0) or 0)
    Rcfd = float((entry or {}).get('Rcfd', 0.0) or 0.0)
    Nr_over_N = Nr / max(1, N_target)
    return Nr, N_target, Rcfd, Nr_over_N

#############################
# 1. 生成 GMM 对象列表 (结构化汇总)
#############################

def GMM_obj_list(target: str, *, config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """生成用于 GMM 的对象及其特征集合。
    找到target对应的所有实例id，及其相关对象分数。
    输出字段:
        target_obj: {obj_id, cfd, N, stability, base_score(=1.0)} 如存在地图
        related: list[{label, obj_id, Rscore, Rcfd, Nr_over_N, stability, N}]
        Robj_scores: RscoreAgent 的标签->分数 (原样)
    """
    cfg = _load_yaml(config_path)
    floors = _load_merged_map(cfg)
    obj_index = _index_objects(floors)
    # 1) 通过 CLIP 阈值选出所有属于目标的实例
    clip_hits = _clip_search_targets(cfg, target)
    target_ids = [oid for oid, _sc in clip_hits] # NOTE 本函数下面用了
    target_candidates = [obj_index[oid] for oid in target_ids if oid in obj_index]
    # fallback: 当 CLIP 索引与当前地图不一致/无命中时，回退到地图内按 label 精确匹配
    if not target_candidates:
        target_candidates = [o for o in obj_index.values() if (o.get('label') or '').lower() == target.lower()]
        clip_hits = [{'obj_id': o.get('obj_id'), 'score': 1.0} for o in target_candidates]
    # 读取 agent 输出

    label_scores = _read_Rscore_output_for_target(cfg, target)
    # 2) 过滤 agent 输出中地图不存在的标签，保留存在的；为每个 target_id × 每个实例 生成记录
    present_labels = set(o.get('label') for o in obj_index.values())
    filtered_scores = {lb: sc for lb, sc in label_scores.items() if lb in present_labels and lb != target}
    by_target: Dict[str, List[Dict[str, Any]]] = {}
    related_items: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()  # NOTE 元组(target_id, obj_id)
    for t in target_candidates:
        tid = t.get('obj_id')
        lst = by_target.setdefault(tid, []) #这个是以tid为key的列表，所以对于每一个key他的值在这里不一样
        # 2.a) NOTE 先增加：由 agent 标签驱动的相关实例（同标签所有实例）
        for lb, rscore in filtered_scores.items():
            instances = [o for o in obj_index.values() if o.get('label') == lb]
            for o in instances:
                key = (tid, o.get('obj_id'))
                if key in seen:
                    continue
                seen.add(key)
                Nr, N_target, Rcfd, Nr_over_N = compute_pair_relation_stats(t, o)
                #NOTICE 这里的Nr_over_N计算的是对的，他会先判断t的R_objs是否有o，如果无返回的都是0
                rec = {
                    'target_id': tid,
                    'label': lb,
                    'obj_id': o.get('obj_id'),
                    'Rscore': rscore,
                    'Rcfd': Rcfd,
                    'Nr_over_N': round(Nr_over_N, 4),
                    'stability': o.get('stability'),
                    'N_obj': o.get('N'),#与target相关的obj的更新次数
                    'exist_prob': o.get('exist_prob', 1.0),
                }
                lst.append(rec)
                related_items.append(rec)

        # 2.b) 再补充：地图中已有的直接关系 target.R_objs,Agent没有预测到的，所以自然Rscore为0
        R_map = t.get('R_objs') or {}
        if isinstance(R_map, dict):
            for rel_oid, _rel in R_map.items():
                o = obj_index.get(rel_oid)
                if not o:
                    continue  # 地图中不存在则跳过
                key = (tid, rel_oid)
                if key in seen:
                    continue
                seen.add(key)
                lb = o.get('label')
                rscore = float(label_scores.get(lb, 0.0))  # 若 agent 包含该物体，则 Rscore=0 由关系增强项补偿
                Nr, N_target, Rcfd, Nr_over_N = compute_pair_relation_stats(t, o)
                rec = {
                    'target_id': tid,
                    'label': lb,
                    'obj_id': rel_oid,
                    'Rscore': rscore,
                    'Rcfd': Rcfd,
                    'Nr_over_N': round(Nr_over_N, 4),
                    'stability': o.get('stability'),
                    'N_obj': o.get('N'),
                    'exist_prob': o.get('exist_prob', 1.0),
                }
                lst.append(rec)
                related_items.append(rec)
    result: Dict[str, Any] = {
        'target': target,
        'Robj_scores(with Agent Predict)': label_scores,
        'filtered_scores(only Real Labels)': filtered_scores,
        'targets_from_clip': [
            {'obj_id': item['obj_id'], 'score': item['score']} if isinstance(item, dict)
            else {'obj_id': item[0], 'score': item[1]}
            for item in clip_hits
            if ((item['obj_id'] if isinstance(item, dict) else item[0]) in obj_index)
        ],
        'related_obj_by_target(All, combined R_objs&agentR)': by_target,
        'related_obj': related_items,
    }
    return result

#############################
# 2. 计算目标物体自身得分
#############################

def Calculate_obj_Score(N: int, stability: float, cfd: Optional[float], *, config_path: Path = CONFIG_PATH) -> float:
    """计算目标物体自身得分 (归一化到 GMM.Score_max)。

    TODO公式思路: base = or ……
    TODOsigmoid_N = 1/(1+exp(-k*(N-N0))) (使用 GMM.weight_sigmoid 参数)
    """
    cfg = _load_yaml(config_path)
    gmm_cfg = cfg.get('GMM') or {}
    Score_max = float(gmm_cfg.get('Score_max', 1.0)) #NOTICE
    sig_conf = gmm_cfg.get('weight_sigmoid') or {}
    k = float(sig_conf.get('k', 1.0))
    N0 = float(sig_conf.get('N0', 5.0))

    N = max(0, int(N))
    stability = max(0.0, min(1.0, float(stability)))
    sigmoid_N = 1.0 / (1.0 + math.exp(-k * (N - N0)))
    
    if cfd is None:
        base = sigmoid_N * stability
    else:
        cfd_val = max(0.0, min(1.0, float(cfd)))
        sigmoid_N = 1.0 / (1.0 + math.exp(-k * (N + 1 - N0)))#TODO
        base = sigmoid_N * stability * cfd_val
    return round(base * Score_max, 4)

def Calculate_Robj_Score(
    total_N: int,
    Nr_over_N: float,
    Rscore: float,
    Rcfd: float,
    stability: float,
    exist_prob: float = 1.0,
    *,
    config_path: Path = CONFIG_PATH,
    weight_mode: Optional[str] = None,
) -> float:
    """相关对象综合得分 (归一化到 GMM.Score_max)。

    公式:
        weights = sigmoid/exp (随 total_N 增强关系权重)
        relation_enhance = exp( lambda * Nr_over_N + Rcfd )
        fused = ω1 * Rscore + ω2 * relation_enhance
        score = fused * stability_norm

    归一化: 将 Rscore (0~1) 与 relation_enhance 通过 Log 压缩再映射到 (0~1), 最终乘以 Score_max。
    """
    #下面只是配置列表
    cfg = _load_yaml(config_path)
    gmm_cfg = cfg.get('GMM') or {}
    Score_max = float(gmm_cfg.get('Score_max', 1.0))
    lambda_param = float(gmm_cfg.get('Nr_lambda_param', 0.5))
    exp_conf = gmm_cfg.get('weight_exponential') or {}
    sig_conf = gmm_cfg.get('weight_sigmoid') or {}
    mode = (weight_mode or gmm_cfg.get('weight_mode') or 'Sigmoid').strip().lower()
    exp_a = float(exp_conf.get('a', 0.1))
    exp_b = float(exp_conf.get('b', 0.25))
    sig_k = float(sig_conf.get('k', 1.0))
    sig_N0 = float(sig_conf.get('N0', 5.0))

    total_N = max(1, int(total_N))
    Nr_over_N = max(0.0, float(Nr_over_N))
    Rscore = max(0.0, float(Rscore))
    Rcfd = max(0.0, min(1.0, float(Rcfd)))
    stability = max(0.0, min(1.0, float(stability)))
    exist_prob = max(0.0, min(1.0, float(exist_prob)))

    if mode == 'exponential':
        omega_N1, omega_N2 = utility.calculate_exponential_weights(total_N, a=exp_a, b=exp_b)
    else:
        # 默认 Sigmoid
        omega_N1, omega_N2 = utility.calculate_sigmoid_weights(total_N, k=sig_k, N0=sig_N0)

    rel_norm = 0.0

    if Nr_over_N > 0 and Rcfd > 0:
        relation_enhance = math.exp(lambda_param * Nr_over_N + Rcfd)  # >=1
        # Log 压缩 relation_enhance 到 0~1 区域: log(1+x)/log(1+e^{lambda+1}) 近似
        # 取上界: Nr_over_N<=1, Rcfd<=1 => exponent<=lambda+1
        upper = math.exp(lambda_param * 1.0 + 1.0)
        # HACK 归一化了
        rel_norm = math.log(1 + relation_enhance) / math.log(1 + upper)

    #把Rscore，归一化到 [0,1]
    Rscore_max = _get_rscore_norm_max(cfg)
    Rscore = max(0.0, min(1.0, float(Rscore) / Rscore_max))
    
    fused = omega_N1 * Rscore + omega_N2 * rel_norm
    score = fused * stability * exist_prob
    return round(score * Score_max, 4)

#############################
# 3. 汇总打分管线 (可选)
#############################

def build_GMM_feature_set(target: str, *, config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    meta = GMM_obj_list(target, config_path=config_path)
    # 为每个 related 计算综合分，使用其对应 target_id 的 N
    # 准备 target_id -> N 映射
    cfg = _load_yaml(config_path)
    floors = _load_merged_map(cfg)
    obj_index = _index_objects(floors) # devided by id
    # NOTE 下面是从CLIP模型里面筛选出来的与我们要找的target最语义相似的物体（后续可以通过直接喂给AI找出来与装花的瓶子最相关的，肯定比CLIP语义相似更加好）
    target_N_map: Dict[str, int] = {oid: int((obj_index.get(oid) or {}).get('N', 1)) for oid in [t['obj_id'] for t in meta.get('targets_from_clip', [])]}
    weight_mode_cfg = (cfg.get('GMM') or {}).get('weight_mode')
    for item in meta['related_obj']:
        tN = target_N_map.get(item['target_id'], 1)
        # 添加相关物体的 GMM_score 计算
        item['GMM_score'] = Calculate_Robj_Score(
            total_N=tN,
            Nr_over_N=item.get('Nr_over_N', 0.0),
            Rscore=item.get('Rscore', 0.0),
            Rcfd=item.get('Rcfd', 0.0),
            stability=item.get('stability') or 0.5,
            exist_prob=item.get('exist_prob', 1.0),
            config_path=config_path,
            weight_mode=weight_mode_cfg,
        )

    #NOTE 多目标自身得分列表
    meta['targets_self'] = []
    for t in meta.get('targets_from_clip', []):
        oid = t['obj_id']
        o = obj_index.get(oid)
        if not o:
            continue
        self_score = Calculate_obj_Score(
            N=o.get('N', 0),
            stability=o.get('stability') or 0.5,
            cfd=o.get('cfd'),
            config_path=config_path,
        )
        meta['targets_self'].append({'obj_id': oid, 'GMM_self_score': self_score})
    # 基于 obj_id 的聚合（跨多个 target_id 的同一 related 对象）
    related_objs = meta.get('related_obj', [])

    # NOTICE List[Dict[str, Any]，意思是把related_objs里面的每一个ID的物体都放到一个字典里面
    by_obj_raw: Dict[str, List[Dict[str, Any]]] = {}
    for r in related_objs:
        oid = r.get('obj_id')
        if not oid:
            continue
        by_obj_raw.setdefault(oid, []).append(r)
    # TODO 读取 GMM.merge_rate（用于重复出现的放大，假设 n 次则乘以 merge_rate^(n-1)）
    gmm_cfg = (_load_yaml(config_path).get('GMM') or {})
    mr = float(gmm_cfg.get('merge_rate', 1.0))
    by_obj: Dict[str, Dict[str, Any]] = {}
    scores_map: Dict[str, float] = {}
    for oid, lst in by_obj_raw.items():
        n = max(1, len(lst))
        # 平均 GMM_score
        avg = sum((x.get('GMM_score') or 0.0) for x in lst) / n
        # 重复融合加权（假设指数叠乘）
        merge_factor = (mr ** (n - 1)) if n > 1 else 1.0

        final = round(avg * merge_factor, 4)
        
        # 选一个代表行（取分数最高的）
        # rep = max(lst, key=lambda x: x.get('GMM_score', 0.0) or 0.0)
        # by_obj[oid] = {
        #     'obj_id': oid,
        #     'label': rep.get('label'),
        #     'GMM_score': final,
        #     'avg_before_merge': round(avg, 4),
        #     'merge_factor_used': round(merge_factor, 4),
        #     'best_from_target': rep.get('target_id'),
        #     'Nr_over_N': rep.get('Nr_over_N'),
        #     'Rcfd': rep.get('Rcfd'),
        #     'stability': rep.get('stability'),
        #     'N': rep.get('N'),
        #     'count': n,
        # }
        # meta['related_by_obj'] = by_obj
        
        scores_map[oid] = final
    
    # 将目标对象的自评分一并纳入最终映射（NOTE若已存在则取较大值）
    for ts in meta['targets_self']:
        oid = ts.get('obj_id')
        if not oid:
            continue
        self_sc = float(ts.get('GMM_self_score') or 0.0)
        prev = float(scores_map.get(oid, 0.0))
        if self_sc > prev:
            scores_map[oid] = round(self_sc, 4)
    meta['GMM_scores(Target&Related)'] = scores_map
    return meta

__all__ = [
    'GMM_obj_list',
    'Calculate_obj_Score',
    'Calculate_Robj_Score',
    'build_GMM_feature_set'
]
