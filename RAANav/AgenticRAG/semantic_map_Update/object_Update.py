# obj_now 是从json中读取的新的物体信息，obj_history 是历史合并过的的物体信息
# 由于是确认同一个语义下才判断更新，所以stability不会更新
import copy
import math
from datetime import datetime, timezone
from semantic_map.object import Object,RelationMeta
from .utility import obj_IorU, obj_update_Rcfd,obj_judge_region, obj_update_Cfd,obj_id_to_label, append_media_sequential, obj_calculate_iou
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


def _now_iso_utc() -> str:
    # 优先使用虚拟时钟 (仿真环境), 回退到真实系统时间
    try:
        from semantic_map_Create.virtual_clock import now_iso_utc
        return now_iso_utc()
    except ImportError:
        pass
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _merge_cooccur_stats(now_stats: Any, hist_stats: Any, now_R_objs: Dict[str, RelationMeta]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    if isinstance(hist_stats, dict):
        merged.update(copy.deepcopy(hist_stats))
    if isinstance(now_stats, dict):
        for k, v in now_stats.items():
            if k not in merged:
                merged[k] = copy.deepcopy(v)
            else:
                try:
                    merged[k] = merged[k] + v  # type: ignore[operator]
                except Exception:
                    merged[k] = copy.deepcopy(v)
    # 兜底: 从 R_objs 关系提炼共现频次
    for target_id, rel in (now_R_objs or {}).items():
        if not isinstance(rel, dict):
            continue
        try:
            nr_val = int(rel.get('Nr', 0) or 0)
        except Exception:
            nr_val = 0
        if nr_val <= 0:
            continue
        try:
            merged[target_id] = int(merged.get(target_id, 0) or 0) + nr_val
        except Exception:
            merged[target_id] = nr_val
    return merged

def obj_id_update(objs_new: List[Object]):
    """为缺失 obj_id 的对象分配稳定的新 ID。
    规则: label_seq_roomId, HACK仅在该 (room,label) 组合下顺序递增；已有 ID 不改动。
    """
    from collections import defaultdict
    seen_ids = set()
    seq_map = defaultdict(int)  # (room_id,label)->max_seq
    # 先解析已有 id, 记录已使用序号
    for obj in objs_new:
        oid = getattr(obj, 'obj_id', None)
        if not oid:
            continue
        parts = oid.split('_')
        if len(parts) < 3:
            continue  # 非标准格式跳过
        room_id = parts[-1]
        seq_part = parts[-2]
        label = '_'.join(parts[:-2])
        if seq_part.isdigit():
            seq = int(seq_part)
            key = (room_id, label)
            if seq > seq_map[key]:
                seq_map[key] = seq
        seen_ids.add(oid)

    # 为缺失 id 的对象分配新 id
    for obj in objs_new:
        oid = getattr(obj, 'obj_id', None)
        if oid and oid in seen_ids:
            continue
        label = getattr(obj, 'label', 'obj')
        room_id = getattr(obj, 'room_id', 'R0')
        key = (room_id, label)
        seq_map[key] += 1
        seq = seq_map[key]
        new_id = f"{label}_{seq}_{room_id}"
        while new_id in seen_ids:
            seq_map[key] += 1
            seq = seq_map[key]
            new_id = f"{label}_{seq}_{room_id}"
        obj.obj_id = new_id
        seen_ids.add(new_id)


def obj_region_update(obj_now_region, obj_history_region, *, Intersection=True):
    # 更新物体的区域信息
    region = obj_IorU(obj_now_region, obj_history_region, Intersection=Intersection)
    return region

def obj_R_objs_update(
    now_R_objs: Dict[str, RelationMeta],
    history_R_objs: Dict[str, RelationMeta],
    objs_now_map: Dict[str, Object],
    history_by_label: Dict[str, List[Object]],
    obj_N_total: int,
    obj_N_mean: int,
    config_obj: Any
) -> Dict[str, RelationMeta]:
    """
    更新 R_objs 统计信息，根据标签和区域匹配。
    """

    updated_R_objs = copy.deepcopy(now_R_objs)
    
    # 遍历当前对象的 R_objs 字典
    for target_now_id, relation_now in updated_R_objs.items():
        
        # NOTICE这是找到当前对象列表中与 target_now_id 匹配的对象，接着匹配region
        target_obj_now = objs_now_map.get(target_now_id)
        
        target_obj_history = None
        if target_obj_now and target_obj_now.label in history_by_label:
            for o in history_by_label[target_obj_now.label]:
                region_threshold = getattr(config_obj, 'region_threshold', None)
                if region_threshold is None and isinstance(config_obj, dict):
                    region_threshold = config_obj.get('region_threshold', 0.8)
                if obj_judge_region(target_obj_now.region, o.region, region_threshold):
                    target_obj_history = o
                    break
            
            # 3. 如果找到了匹配的历史目标对象
        if target_obj_history:
            # 找到匹配的历史目标对象：关系在“新旧都存在”
            if target_obj_history.obj_id in history_R_objs:
                relation_history = history_R_objs[target_obj_history.obj_id]
                
                # 更新 Nr (累计) - 按需求当前仅统计 Nr，不累计 Nt
                relation_now['Nr'] = relation_now.get('Nr', 0) + relation_history.get('Nr', 0)
                
                # 更新 Rcfd (使用历史均值和当前值)
                epsilon_Rcfd = getattr(config_obj, 'epsilon_Rcfd', None)
                if epsilon_Rcfd is None and isinstance(config_obj, dict):
                    epsilon_Rcfd = config_obj.get('epsilon_Rcfd', 0.75)
                new_Rcfd = obj_update_Rcfd(
                    epsilon_Rcfd,
                    Rcfd_mean=relation_history['Rcfd'],
                    Rcfd_now=relation_now['Rcfd'],
                    N_total=obj_N_total,
                    N_mean=obj_N_mean
                )
                relation_now['Rcfd'] = new_Rcfd
            else:
                # 未找到匹配的历史关系条目：关系为“新增”
                # Rcfd_mean 明确传递为 0，因为没有历史数据
                epsilon_Rcfd = getattr(config_obj, 'epsilon_Rcfd', None)
                if epsilon_Rcfd is None and isinstance(config_obj, dict):
                    epsilon_Rcfd = config_obj.get('epsilon_Rcfd', 0.75)
                relation_now['Rcfd'] = obj_update_Rcfd(
                    epsilon_Rcfd,
                    Rcfd_mean=0.0,
                    Rcfd_now=relation_now['Rcfd'],
                    N_total=obj_N_total,
                    N_mean=obj_N_mean
                )


    # 第二次遍历: 找出只存在于历史中的关系 (本轮未出现 → 视为缺失, 对 Rcfd 做一次平滑衰减)
    for target_history_id, relation_history in history_R_objs.items():
        if target_history_id in updated_R_objs:  # 已经在“新”里出现(被第一轮处理) → 跳过
            continue
        new_entry = copy.deepcopy(relation_history)
        epsilon_Rcfd = getattr(config_obj, 'epsilon_Rcfd', None)
        if epsilon_Rcfd is None and isinstance(config_obj, dict):
            epsilon_Rcfd = config_obj.get('epsilon_Rcfd', 0.75)
        new_entry['Rcfd'] = obj_update_Rcfd(
            epsilon_Rcfd,
            Rcfd_mean=relation_history.get('Rcfd', 0.0),
            Rcfd_now=0.0,  # 本轮缺失
            N_total=obj_N_total,
            N_mean=obj_N_mean
        )
        updated_R_objs[target_history_id] = new_entry

    return updated_R_objs

def store_clip_embedding(epsilon_clip, clip_embedding_history, clip_embedding_now, N_total, N_mean=1):
    # 存储 clip_embedding
    # 暂时设想是和更新cfd以及Rcfd一个方法，最后效果是加权最新获取的clip_embedding，并且存储到文件中，后面会设置读取方法
    # FIX: 原定义参数与调用不一致；移除 obj_now.get_embedding() / obj_history.get_embedding() （不存在）
    # TODO 再仿真验证里面暂时使用不到，所以这里只做指数平滑占位
    if clip_embedding_now is None or clip_embedding_now == []:
        return clip_embedding_history or []
    if clip_embedding_history is None or clip_embedding_history == []:
        return clip_embedding_now
    try:
        hist = np.array(clip_embedding_history, dtype=np.float32)
        now = np.array(clip_embedding_now, dtype=np.float32)
        if hist.shape != now.shape:
            # 维度变化直接返回最新
            return clip_embedding_now
        # 指数累计: 模仿 obj_update_Cfd 公式
        numerator = (epsilon_clip**N_mean - epsilon_clip**N_total) * hist + (1 - epsilon_clip) * now
        denominator = (1 - epsilon_clip**N_total)
        blended = numerator / denominator
        return blended.tolist()
    except Exception:
        return clip_embedding_now



def _pos3d_distance(a: Optional[List[float]], b: Optional[List[float]]) -> Optional[float]:
    """计算两个 pos_3d 之间的欧氏距离，缺失时返回 None."""
    if a is None or b is None:
        return None
    if len(a) < 3 or len(b) < 3:
        return None
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a[:3], b[:3])))


def _clip_cosine(a: Optional[List[float]], b: Optional[List[float]]) -> Optional[float]:
    """计算两个 CLIP embedding 的余弦相似度，缺失时返回 None."""
    if not a or not b:
        return None
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    if va.shape != vb.shape or va.shape[0] == 0:
        return None
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return None
    return float(np.dot(va, vb) / (norm_a * norm_b))


def _find_best_match(
    obj_now: Object,
    candidates: List[Object],
    config_obj: Any,
) -> Optional[Object]:
    """多因子匹配：在候选中找最佳匹配.

    双模式匹配策略:
      A) CLIP 模式 (双方都有 clip_embedding):
         - CLIP 余弦 ≥ clip_match_threshold (默认 0.65) 作为前置过滤
         - 然后按 S_match = η1*IoU + η2*clip_cos - η3*dist 联合评分
         - S_match > τ_match 视为匹配
      B) 轻量模式 (任一方无 clip_embedding):
         - 要求 label 完全相同 (退回到字符串匹配)
         - pos_3d 距离 < dist_threshold 或 2D IoU ≥ region_threshold_loose
         - 选距离最近的
    """
    def _cfg(key, default):
        val = getattr(config_obj, key, None)
        if val is None and isinstance(config_obj, dict):
            val = config_obj.get(key, default)
        return val if val is not None else default

    dist_threshold = float(_cfg('match_dist_threshold', 1.0))
    region_threshold_loose = float(_cfg('region_threshold_loose', 0.3))
    clip_reject_threshold = float(_cfg('clip_reject_threshold', 0.4))
    clip_match_threshold = float(_cfg('clip_match_threshold', 0.65))
    # §3.1 融合系数
    eta1 = float(_cfg('match_eta1', 0.4))  # 空间 IoU
    eta2 = float(_cfg('match_eta2', 0.4))  # CLIP 外观
    eta3 = float(_cfg('match_eta3', 0.2))  # 距离惩罚
    tau_match = float(_cfg('match_tau', 0.55))
    # 距离归一化参数 (用于将米转为 [0,1] 范围)
    dist_norm = float(_cfg('match_dist_norm', 3.0))

    now_emb = getattr(obj_now, 'clip_embedding', None) or []
    has_clip_now = len(now_emb) > 0

    best: Optional[Object] = None
    best_score = -float('inf')  # S_match 越大越好

    for cand in candidates:
        cand_emb = getattr(cand, 'clip_embedding', None) or []
        has_clip_cand = len(cand_emb) > 0

        clip_sim = _clip_cosine(now_emb if has_clip_now else None,
                                cand_emb if has_clip_cand else None)

        if has_clip_now and has_clip_cand:
            # --- A) CLIP 模式: 用 CLIP 余弦做前置过滤 ---
            if clip_sim is None or clip_sim < clip_match_threshold:
                continue  # CLIP 不够相似, 跳过

            # 计算 S_match = η1*IoU + η2*clip_cos - η3*dist_normalized
            dist = _pos3d_distance(getattr(obj_now, 'pos_3d', None),
                                   getattr(cand, 'pos_3d', None))
            iou = obj_calculate_iou(obj_now.region, cand.region) if obj_now.region and cand.region else 0.0
            dist_val = min(dist / dist_norm, 1.0) if dist is not None else 0.5
            s_match = eta1 * iou + eta2 * clip_sim - eta3 * dist_val

            if s_match > tau_match and s_match > best_score:
                best_score = s_match
                best = cand
            elif s_match <= tau_match:
                # 即使 S_match 不过阈值, 如果距离非常近也可以匹配 (安全网)
                if dist is not None and dist < dist_threshold * 0.5:
                    fallback_score = clip_sim - dist_val * 0.1
                    if fallback_score > best_score:
                        best_score = fallback_score
                        best = cand
        else:
            # --- B) 轻量模式: label 严格匹配 + 空间条件 ---
            if obj_now.label != cand.label:
                continue

            dist = _pos3d_distance(getattr(obj_now, 'pos_3d', None),
                                   getattr(cand, 'pos_3d', None))
            iou_ok = obj_judge_region(obj_now.region, cand.region, region_threshold_loose)
            dist_ok = dist is not None and dist < dist_threshold

            if not dist_ok and not iou_ok:
                continue

            # CLIP 否决 (如果一方有)
            if clip_sim is not None and clip_sim < clip_reject_threshold:
                continue

            # 距离越近越好 (负距离 → 越大越好)
            score = -(dist if dist is not None else 999.0)
            if score > best_score:
                best_score = score
                best = cand

    return best


def object_update(objs_now: List[Object], objs_history: List[Object], config_obj: Any) -> List[Object]:
    """
    更新 objs_now 列表中的对象，通过 label 和 region 匹配历史对象。
    """
    # OPTIMIZE 可以在物体属性里面引入hash表，如果label&region一样，就
    # 将 objs_history 按 label 分组，以提高查找效率
    now_obj_map = {obj.obj_id: obj for obj in objs_now}
    history_obj_map = {obj.obj_id: obj for obj in objs_history}

    history_by_label = {}
    for obj in objs_history:
        label = obj.label
        if label not in history_by_label:
            history_by_label[label] = []
        history_by_label[label].append(obj)

    objs_new = []
    # 记录已经占用的稳定 ID (先放入历史所有ID, 新增物体若复用这些ID且未匹配成功需要强制让位)
    used_ids = set(history_obj_map.keys())
    processed_history_ids = set() # 追踪已经处理过的历史对象ID
    
    # 遍历当前对象列表
    for obj_now in objs_now:
        matching_obj_history = None

        # 判断是否有 CLIP embedding → 决定搜索范围
        now_emb = getattr(obj_now, 'clip_embedding', None) or []
        if len(now_emb) > 0:
            # CLIP 模式: 搜索全部历史物体 (CLIP cosine 会区分不同类)
            matching_obj_history = _find_best_match(
                obj_now, objs_history, config_obj
            )
        elif obj_now.label in history_by_label:
            # 轻量模式: 只搜同 label 候选
            matching_obj_history = _find_best_match(
                obj_now, history_by_label[obj_now.label], config_obj
            )

        if matching_obj_history:
            # 找到了匹配的对象，进行更新
            obj_new = copy.deepcopy(obj_now)
            # FIX: 变量名修正 (obj_history -> matching_obj_history)
            obj_new.stability = matching_obj_history.stability
            # NOTICE 保持使用历史 label / room_id 作为最终锚定身份
            obj_new.label = matching_obj_history.label
            obj_new.room_id = matching_obj_history.room_id
            obj_new.obj_id = matching_obj_history.obj_id  # 保持历史 obj_id

            # NOTICE--- 多次扫描合并: 累加两侧 N (两侧都可能已为多扫描聚合值)
            hist_N = getattr(matching_obj_history, 'N', 0) or 0
            now_N = getattr(obj_now, 'N', 1) or 1
            obj_new.N = hist_N + now_N
            epsilon_cfd = getattr(config_obj, 'epsilon_cfd', None)
            if epsilon_cfd is None and isinstance(config_obj, dict):
                epsilon_cfd = config_obj.get('epsilon_cfd', 0.75)
            obj_new.cfd = obj_update_Cfd(epsilon_cfd, matching_obj_history.cfd, obj_now.cfd, obj_new.N, matching_obj_history.N)
            # FIX: 调整 clip embedding 调用
            epsilon_clip = getattr(config_obj, 'epsilon_clip', None)
            if epsilon_clip is None and isinstance(config_obj, dict):
                epsilon_clip = config_obj.get('epsilon_clip', 0.75)
            obj_new.clip_embedding = store_clip_embedding(
                epsilon_clip,
                matching_obj_history.clip_embedding,
                obj_now.clip_embedding,
                obj_new.N,
                matching_obj_history.N
            )
            #OPTIMIZE 如果这里的clip太大，可以考虑需要更新时才加载入obj对象里面
            
            obj_new.region = obj_region_update(obj_now.region, matching_obj_history.region)

            # pos_3d EMA 更新: 新观测加权 α=pos_ema_alpha 融合历史位置
            now_p3d = getattr(obj_now, 'pos_3d', None)
            hist_p3d = getattr(matching_obj_history, 'pos_3d', None)
            if now_p3d is not None and hist_p3d is not None and len(now_p3d) >= 3 and len(hist_p3d) >= 3:
                def _cfg_val(key, default):
                    val = getattr(config_obj, key, None)
                    if val is None and isinstance(config_obj, dict):
                        val = config_obj.get(key, default)
                    return float(val) if val is not None else default
                alpha = _cfg_val('pos_ema_alpha', 0.3)
                obj_new.pos_3d = [
                    round(alpha * n + (1 - alpha) * h, 6)
                    for n, h in zip(now_p3d[:3], hist_p3d[:3])
                ]
            elif now_p3d is not None:
                obj_new.pos_3d = now_p3d
            else:
                obj_new.pos_3d = hist_p3d

            # pos_2d: 取最新观测值 (或保留历史)
            obj_new.pos_2d = getattr(obj_now, 'pos_2d', None) or getattr(matching_obj_history, 'pos_2d', None)

            obj_new.R_objs = obj_R_objs_update(
                obj_now.R_objs,
                matching_obj_history.R_objs,
                now_obj_map,
                history_by_label,
                obj_N_total=obj_new.N,
                obj_N_mean=matching_obj_history.N,
                config_obj=config_obj
            )
            obj_new.cooccur_stats = _merge_cooccur_stats(
                getattr(obj_now, 'cooccur_stats', None),
                getattr(matching_obj_history, 'cooccur_stats', None),
                obj_new.R_objs,
            )
            obj_new.last_update_time = getattr(obj_now, 'last_update_time', None) or _now_iso_utc()
            # exist_prob: 对象“当前仍存在”的置信度(0~1)
            # - 匹配到(被观测到)时, exist_prob 向 1 恢复
            # - 未匹配(缺失)时, 在下方按 missing_decay 衰减
            hist_exist = getattr(matching_obj_history, 'exist_prob', 1.0)
            now_exist = getattr(obj_now, 'exist_prob', 1.0)
            recover_rate = getattr(config_obj, 'exist_recover_rate', None)
            if recover_rate is None and isinstance(config_obj, dict):
                recover_rate = config_obj.get('exist_recover_rate', 0.75)
            try:
                recover_rate = float(recover_rate)
            except Exception:
                recover_rate = 0.75
            recover_rate = min(1.0, max(0.0, recover_rate))
            merged_exist = recover_rate * float(now_exist) + (1.0 - recover_rate) * float(hist_exist)
            obj_new.exist_prob = min(1.0, max(0.0, merged_exist))
            
            #FIXME 图片 / 描述: 改为连续追加 (history.N + i)
            mode_cfg = getattr(config_obj, 'imgs_description_append', None)
            append_mode = 'sequential_totalN'
            if isinstance(mode_cfg, dict):
                append_mode = mode_cfg.get('mode', 'sequential_totalN')
            imgs_new, desc_new, final_total_N = append_media_sequential(
                matching_obj_history.imgs if isinstance(matching_obj_history.imgs, dict) else {},
                matching_obj_history.description if isinstance(matching_obj_history.description, dict) else {},
                obj_now,
                hist_N=hist_N,
                now_N=now_N,
                mode=append_mode
            )
            # final_total_N 应与 obj_new.N 保持一致；若计算得到不一致，以累加逻辑为准
            obj_new.imgs = imgs_new
            obj_new.description = desc_new
            obj_new.N = max(obj_new.N, final_total_N)

            objs_new.append(obj_new)
            processed_history_ids.add(matching_obj_history.obj_id)
                
        else:
            # 没有找到匹配项：这是一个“新增”的对象(位置差异或真实新增)。
            obj_new = copy.deepcopy(obj_now)
            # 若其 obj_id 与历史已存在(或前面新增中已使用) → 视为需要新的编号；清空以便后续统一分配
            if getattr(obj_new, 'obj_id', None) in used_ids:
                obj_new.obj_id = ''
            else:
                # 占用该ID，防止后续新增再重复
                if getattr(obj_new, 'obj_id', None):
                    used_ids.add(obj_new.obj_id)
            # 如果新图中已经带有累计 N (来自另一聚合), 保留; 否则设为1
            if not getattr(obj_new, 'N', None) or obj_new.N <= 0:
                obj_new.N = 1  # 确保 N 至少为1
            obj_new.last_update_time = getattr(obj_now, 'last_update_time', None) or _now_iso_utc()
            obj_new.exist_prob = min(1.0, max(0.0, float(getattr(obj_now, 'exist_prob', 1.0) or 1.0)))
            obj_new.cooccur_stats = _merge_cooccur_stats(
                getattr(obj_now, 'cooccur_stats', None),
                None,
                getattr(obj_new, 'R_objs', {}) or {},
            )
            # 初次出现也要初始化 imgs/description 结构为 dict
            if isinstance(obj_new.imgs, list):
                obj_new.imgs = {str(obj_new.N): obj_new.imgs}
            elif not isinstance(obj_new.imgs, dict):
                obj_new.imgs = {str(obj_new.N): []}
            if isinstance(obj_new.description, str):
                obj_new.description = {str(obj_new.N): obj_new.description}
            elif not isinstance(obj_new.description, dict):
                obj_new.description = {str(obj_new.N): ''}
            # 这里是假设 R_objs 已经初始化有值
            objs_new.append(obj_new)

    for obj_history in objs_history:
        if obj_history.obj_id not in processed_history_ids:
            # 这个历史对象没有在第一阶段被匹配到，所以它已经“消失”
            obj_new = copy.deepcopy(obj_history)
            missing_decay = getattr(config_obj, 'missing_decay', None)
            if missing_decay is None and isinstance(config_obj, dict):
                missing_decay = config_obj.get('missing_decay', 0.8)
            try:
                missing_decay = float(missing_decay)
            except Exception:
                missing_decay = 0.8
            missing_decay = min(1.0, max(0.0, missing_decay))
            hist_exist = float(getattr(obj_history, 'exist_prob', 1.0) or 1.0)
            # missing_decay: 每轮“未观测到”时的存在概率保留系数
            # 例: missing_decay=0.8, 连续3轮未观测 => exist_prob 乘以 0.8^3
            # 作用: 防止一次漏检就删对象，同时让长期未出现对象自然降权
            obj_new.exist_prob = min(1.0, max(0.0, hist_exist * missing_decay))
            objs_new.append(obj_new)
    # 由于物体的id由label和room_id内的该物体的数量决定，所以需要最后更新物体id
    obj_id_update(objs_new)

    return objs_new