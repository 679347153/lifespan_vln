import math
from shapely.geometry import Polygon
from typing import List, Dict, Optional

def obj_id_to_label(obj_id: str) -> str:
    # <label>_<sequence>_<roomId>
    if not isinstance(obj_id, str) or '_' not in obj_id:
        return ''
    parts = obj_id.split('_')
    if len(parts) < 3:
        return ''
    seq_candidate = parts[-2]
    room_candidate = parts[-1]
    is_seq = seq_candidate.isdigit()
    is_room = (room_candidate.startswith(('R','r')) and room_candidate[1:].isdigit())
    if is_seq and is_room:
        label_parts = parts[:-2]
        return '_'.join(label_parts) if label_parts else ''
    return ''

def dict_to_polygon(points_list: List[Dict[str, float]]) -> Polygon:
    # 将 [{'x':..., 'y':...}, ...] 格式的点列表转换为 Shapely Polygon 对象
    coords = [(p['x'], p['y']) for p in points_list]
    return Polygon(coords)

def polygon_to_dict(polygon: Polygon) -> List[Dict[str, float]]:
    if not isinstance(polygon, Polygon):
        raise TypeError("输入必须是一个 Shapely Polygon 对象")
    # 提取多边形外部边界的坐标
    exterior_coords = polygon.exterior.coords
    
    # 使用列表推导式将坐标元组转换为字典列表
    points_dict_list = [{'x': coord[0], 'y': coord[1]} for coord in exterior_coords]
    
    return points_dict_list

def obj_IorU(region_now, region_history,*, Intersection):
    # 计算并返回物体的并集或交集区域
    polygon_now = dict_to_polygon(region_now)
    polygon_history = dict_to_polygon(region_history)
    if Intersection:
        region_new = polygon_now.intersection(polygon_history)
    else:
        region_new = polygon_now.union(polygon_history)
    return polygon_to_dict(region_new)

def obj_judge_region(obj_now_region, obj_history_region, region_threshold):
    """判断两个物体区域是否匹配 (使用 IOU >= 阈值)。

    之前版本 BUG: 使用的是 绝对交集面积 > threshold (例如 0.8) 判定，
    对于小物体 (面积 <0.8) 永远无法匹配 → 造成“同 id 物体”被当成新增，
    最终历史版本又被补回，出现重复 obj_id。这里改为 IOU (intersection / union)。
    """
    iou = obj_calculate_iou(obj_now_region, obj_history_region)
    return iou >= region_threshold


def obj_calculate_iou(region_a, region_b) -> float:
    """计算两个物体区域的 IOU (intersection / union), 返回 [0, 1]."""
    if not region_a or not region_b:
        return 0.0
    poly_a = dict_to_polygon(region_a)
    poly_b = dict_to_polygon(region_b)
    if poly_a.is_empty or poly_b.is_empty:
        return 0.0
    inter = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    if union <= 0:
        return 0.0
    return inter / union

def obj_update_Cfd(epsilon, Cfd_mean, Cfd_now, N_total, N_mean=1):
    # N 是这个对象的更新次数
    
    numerator = (epsilon**N_mean - epsilon**N_total) * Cfd_mean + (1 - epsilon) * Cfd_now
    denominator = (1 - epsilon**N_total)

    result = numerator / denominator

    return result

def obj_update_Rcfd(epsilon_Rcfd, Rcfd_mean, Rcfd_now, N_total, N_mean=1):
    # Rcfd_now = 1 or 0
    numerator = (epsilon_Rcfd**N_mean - epsilon_Rcfd**N_total) * Rcfd_mean + (1 - epsilon_Rcfd) * Rcfd_now
    denominator = (1 - epsilon_Rcfd**N_total)

    result = numerator / denominator

    return result

# ----------------- Media Append Utils -----------------
def append_media_sequential(history_imgs, history_desc, obj_now, *, hist_N: int, now_N: int, mode: str = 'sequential_totalN'):
    """公共媒体追加函数 (从 object_Update 抽取)。
    sequential_totalN: 生成 (hist_N+1 .. hist_N+now_N) 键；支持 obj_now.imgs/description 为 dict 或基础类型。
    返回 (imgs_new, desc_new, final_total_N)
    """
    from copy import deepcopy
    imgs_new = deepcopy(history_imgs) if isinstance(history_imgs, dict) else {}
    desc_new = deepcopy(history_desc) if isinstance(history_desc, dict) else {}
    if mode != 'sequential_totalN':
        # fallback: 原样合并
        if isinstance(obj_now.imgs, dict):
            imgs_new.update(obj_now.imgs)
        if isinstance(obj_now.description, dict):
            desc_new.update(obj_now.description)
        elif isinstance(obj_now.description, str):
            desc_new[str(len(desc_new)+1)] = obj_now.description
        return imgs_new, desc_new, max(hist_N, hist_N + now_N)
    effective_now = max(1, now_N)
    imgs_source = obj_now.imgs if isinstance(obj_now.imgs, dict) else {}
    desc_source = obj_now.description if isinstance(obj_now.description, dict) else {}
    desc_string = obj_now.description if isinstance(obj_now.description, str) else None
    last_hist_total = hist_N
    for i in range(1, effective_now + 1):
        total_idx = str(last_hist_total + i)
        if total_idx not in imgs_new:
            imgs_new[total_idx] = imgs_source.get(str(i), imgs_source.get(str(now_N), [])) if imgs_source else []
        if total_idx not in desc_new:
            if desc_string is not None:
                desc_new[total_idx] = desc_string if i == 1 else ''
            elif desc_source:
                desc_new[total_idx] = desc_source.get(str(i), desc_source.get(str(now_N), ''))
            else:
                desc_new[total_idx] = ''
    final_total = last_hist_total + effective_now
    return imgs_new, desc_new, final_total