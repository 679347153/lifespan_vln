from semantic_map import Object,Room,Floor 
from semantic_map_Update.object_Update import object_update  # FIX: 绝对导入兼容直接运行
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import List, Tuple, Dict, Any
from semantic_map_Update.utility import dict_to_polygon
# hydra 在 python 3.13 可能不可用，做兼容
try:  # FIX: 兼容 hydra 缺失
    import hydra  # type: ignore
except Exception:  # noqa: E722
    def _identity_decorator(*d_args, **d_kwargs):
        def wrapper(fn):
            return fn
        return wrapper
    hydra = type('hydra_stub', (), {'main': _identity_decorator})()  # type: ignore

# 我们用于合并的地图一定是同一个building下的，也就是房间和楼层的 ID 是唯一且相互有唯一对应关系的
# 扫描第一次建立初始图会给扫描到的房间和楼层命名 ID，后续更新会基于定位和已有的地图识别房间正确的分配这些 ID


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _hydrate_object_defaults(floors: List[Floor], source_tag: str) -> List[str]:
    """统一补齐对象关键字段，确保后续评分门控可用。

    补齐项:
    - last_update_time: 缺失时写入当前 UTC 时间，避免 ΔT 门控回退
    - exist_prob: 缺失时设 1.0，表示默认“当前存在”
    - cooccur_stats: 缺失时设 {}
    """
    warns: List[str] = []
    fixed_ts = 0
    fixed_exist = 0
    fixed_cooccur = 0
    for fl in floors:
        for room in fl.rooms:
            for obj in room.objects:
                if getattr(obj, 'last_update_time', None) in (None, ''):
                    obj.last_update_time = _now_iso_utc()
                    fixed_ts += 1
                if getattr(obj, 'exist_prob', None) is None:
                    obj.exist_prob = 1.0
                    fixed_exist += 1
                else:
                    try:
                        obj.exist_prob = min(1.0, max(0.0, float(obj.exist_prob)))
                    except Exception:
                        obj.exist_prob = 1.0
                        fixed_exist += 1
                if not isinstance(getattr(obj, 'cooccur_stats', None), dict):
                    obj.cooccur_stats = {}
                    fixed_cooccur += 1
    if fixed_ts or fixed_exist or fixed_cooccur:
        warns.append(
            f"[DATA_HYDRATE] source={source_tag} fixed last_update_time={fixed_ts}, exist_prob={fixed_exist}, cooccur_stats={fixed_cooccur}"
        )
    return warns

def _calc_room_shape_metrics(room_a: Room, room_b: Room) -> Dict[str, Any]:
    """房间形状/面积相似度检测 (多边形 IOU / 面积比等)
    返回:
        dict: { 'iou': float, 'area_a': float, 'area_b': float, 'area_ratio': float }
    """
    poly_a = dict_to_polygon(room_a.region or room_a.to_dict().get('Region', []))
    poly_b = dict_to_polygon(room_b.region or room_b.to_dict().get('Region', []))
    if poly_a.is_empty or poly_b.is_empty:
        return {'iou': 0.0, 'area_a': poly_a.area, 'area_b': poly_b.area, 'area_ratio': 0.0}
    inter = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    iou = inter / union if union > 0 else 0.0
    area_a = poly_a.area
    area_b = poly_b.area
    area_ratio = (area_a / area_b) if area_b > 0 else 0.0
    return {
        'iou': iou,
        'area_a': area_a,
        'area_b': area_b,
        'area_ratio': area_ratio
    }

def _validate_rooms(floor_now: Floor, floor_history: Floor, iou_thresh: float = 0.90, area_ratio_tol: float = 0.2) -> List[str]:
    """对同一楼层内交集的房间做形状安全检测。
    area_ratio_tol: 允许的 (area_now/area_history) 与 1 的偏离最大比例
    返回警告列表。"""
    warnings: List[str] = []
    hist_map = {r.room_id: r for r in floor_history.rooms}
    for r_now in floor_now.rooms:
        r_hist = hist_map.get(r_now.room_id)
        if not r_hist:
            continue
        metrics = _calc_room_shape_metrics(r_now, r_hist)
        area_ratio = metrics['area_ratio']
        # 归一化成最接近 1 的表示 (大于1 取倒数, 看偏离程度)
        dev = area_ratio if area_ratio >= 1 else (1/area_ratio if area_ratio>0 else 999)
        cond_area = dev <= (1 + area_ratio_tol)
        if metrics['iou'] < iou_thresh or not cond_area:
            warnings.append(
                f"[ROOM_SHAPE_WARN] floor={floor_now.floor_id} room={r_now.room_id} IOU={metrics['iou']:.3f} area_ratio={area_ratio:.3f}" \
                f" (阈值: IOU>={iou_thresh}, 偏离<=±{area_ratio_tol*100:.0f}%)"
            )
    return warnings

def _load_floors_flexible(path: str) -> List[Floor]:
    """兼容加载: 
    - 标准: list[Floor]
    - 房间列表: list[Room] (含 room_id / floor_id) → 按 floor_id 分组构造 Floor
    - 混合: (有 rooms 的 dict 视为 Floor, 其它含 room_id 的视为单房间)
    """
    p = Path(path)
    with p.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"期望 {path} 顶层为 list")
    if not data:
        return []
    is_floor_list = all(isinstance(it, dict) and 'floor_id' in it and 'rooms' in it for it in data if isinstance(it, dict))
    if is_floor_list:
        return [Floor.from_dict(d) for d in data]
    is_room_list = all(isinstance(it, dict) and 'room_id' in it for it in data if isinstance(it, dict))
    if is_room_list:
        groups: Dict[str,List[Room]] = {}
        for rd in data:
            if not isinstance(rd, dict):
                continue
            fid = rd.get('floor_id') or 'F1'
            r_obj = Room.from_dict(rd)
            r_obj.floor_id = fid
            groups.setdefault(fid, []).append(r_obj)
        floors: List[Floor] = []
        for fid, rooms in sorted(groups.items(), key=lambda kv: kv[0]):
            floors.append(Floor(floor_id=fid, rooms=rooms))
        return floors
    # 混合
    floors: List[Floor] = []
    leftover: List[Dict[str,Any]] = []
    for item in data:
        if isinstance(item, dict) and 'floor_id' in item and 'rooms' in item:
            try:
                floors.append(Floor.from_dict(item))
            except Exception:
                pass
        elif isinstance(item, dict) and 'room_id' in item:
            leftover.append(item)
    if leftover:
        idx = {f.floor_id: f for f in floors}
        for rd in leftover:
            fid = rd.get('floor_id') or 'F1'
            if fid not in idx:
                idx[fid] = Floor(floor_id=fid, rooms=[])
                floors.append(idx[fid])
            try:
                r_obj = Room.from_dict(rd)
                r_obj.floor_id = fid
                idx[fid].rooms.append(r_obj)
            except Exception:
                pass
        for f in floors:
            f._room_ids = [r.room_id for r in f.rooms]
    return floors

def run_merge(
    floors_now: List[Floor],
    floors_history: List[Floor],
    config_map,
    *,
    shape_check: bool = True,
    allow_new_floors: bool = True,
    allow_new_rooms: bool = True,
) -> Tuple[List[Floor], List[str]]:
    """合并主逻辑 (支持部分楼层/房间 & 新增楼层/房间)。

    策略:
    1. 以 *history 楼层顺序* 为基础遍历, 对存在于 now 的对应楼层做增量更新。
    2. 未在 now 提供的楼层保持原样(实现局部更新不会丢失其余楼层)。
    3. now 中出现的新增楼层 (history 不存在) 且 allow_new_floors=True 时直接整体加入。
    4. 楼层内: 以历史 rooms 为基底, 对 now 中存在且匹配 room_id 的房间调用 object_update; 新房间若 allow_new_rooms=True 则整体复制加入。
    5. 新增房间与楼层的对象 N / 房间 N 若为空则初始化为 1。
    6. 形状校验仅对 *同时存在于两侧* 的房间执行。
    返回: (floors_new, warnings)
    """
    history_floor_map = {f.floor_id: f for f in floors_history}
    now_floor_map = {f.floor_id: f for f in floors_now}
    floors_new: List[Floor] = []
    warnings: List[str] = []

    # 统一入口补字段: 确保所有对象都具备 ΔT 门控所需字段
    warnings.extend(_hydrate_object_defaults(floors_now, 'map_now'))
    warnings.extend(_hydrate_object_defaults(floors_history, 'map_history'))

    # ---- 配置: 房间形状阈值 ---- #
    iou_thresh = 0.90
    area_tol = 0.2
    try:
        update_cfg = getattr(config_map, 'update', None) if not isinstance(config_map, dict) else config_map.get('update', {})
        if update_cfg:
            rs_cfg = update_cfg.get('room_shape', {}) if isinstance(update_cfg, dict) else {}
            iou_thresh = rs_cfg.get('iou_threshold', iou_thresh)
            area_tol = rs_cfg.get('area_ratio_tolerance', area_tol)
    except Exception:
        pass

    # ---- 先遍历历史楼层 (保持顺序 & 局部更新) ---- #
    for hist_floor in floors_history:
        floor_id = hist_floor.floor_id
        now_floor = now_floor_map.get(floor_id)
        if not now_floor:
            # 未提供更新 → 原样复制
            floors_new.append(copy.deepcopy(hist_floor))
            continue
        # 有更新: 基于历史复制
        merged_floor = copy.deepcopy(hist_floor)
        rooms_map: Dict[str, Room] = {r.room_id: copy.deepcopy(r) for r in hist_floor.rooms}
        hist_rooms_map = {r.room_id: r for r in hist_floor.rooms}
        # object_update 配置获取
        cfg_obj = getattr(config_map, 'object', None)
        if cfg_obj is None and isinstance(config_map, dict):
            cfg_obj = config_map.get('object', {})
        for room_now in now_floor.rooms:
            room_hist = hist_rooms_map.get(room_now.room_id)
            if room_hist is None:
                if not allow_new_rooms:
                    continue
                # 新增房间: 深拷贝 now 房间; 初始化 N/对象N
                new_room = copy.deepcopy(room_now)
                # 对于新房间不增加 N, 保持其首次观测次数 (若缺省设为1)
                new_room.N = new_room.N if (new_room.N and new_room.N > 0) else 1
                # 严格假设 imgs / description / room_name 均为 dict；若不是则回退为空 dict
                if not isinstance(new_room.imgs, dict):
                    new_room.imgs = {}
                if not isinstance(new_room.description, dict):
                    new_room.description = {}
                if not isinstance(new_room.room_name, dict):
                    new_room.room_name = {}
                for o in new_room.objects:
                    # 新对象保留其自身 N (缺省设 1)
                    if getattr(o, 'N', None) in (None, 0):
                        try: o.N = 1  # type: ignore
                        except Exception: pass
                rooms_map[new_room.room_id] = new_room
            else:
                # 现有房间增量合并
                merged_room = copy.deepcopy(room_hist)
                merged_room.objects = object_update(room_now.objects, room_hist.objects, cfg_obj)
                merged_room._obj_ids = [obj.obj_id for obj in merged_room.objects]
                # 房间多扫描累计: 改为 sum 模式 (history.N + now.N 或 +1 若 now 缺省)
                merged_room.N = (room_hist.N or 0) + (getattr(room_now, 'N', None) or 1)
                # ---- 严格 dict 模式: room_name / imgs / description 均视为 dict ----
                if not isinstance(merged_room.room_name, dict):
                    merged_room.room_name = {}
                if not isinstance(room_now.room_name, dict):
                    room_now.room_name = {}
                merged_room.room_name[str(merged_room.N)] = room_now.room_name.get(str(merged_room.N), room_now.room_id or merged_room.room_id)

                # 复制历史媒体 (均为 dict 假设)
                merged_room.imgs = copy.deepcopy(room_hist.imgs) if isinstance(room_hist.imgs, dict) else {}
                merged_room.description = copy.deepcopy(room_hist.description) if isinstance(room_hist.description, dict) else {}
                hist_N = room_hist.N or 0
                # 解析当前媒体条目（按键排序）
                # 归一化媒体键 (可能存在 int / str 混合) 并排序: 纯数字键按数值优先，其次按字典序
                imgs_items: list[tuple[str, list]] = []
                if isinstance(room_now.imgs, dict):
                    normalized_imgs = {str(k): v for k, v in room_now.imgs.items()}
                    keys = list(normalized_imgs.keys())
                    keys.sort(key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
                    imgs_items = [(k, normalized_imgs[k]) for k in keys]
                desc_items_raw = room_now.description if isinstance(room_now.description, dict) else {}
                desc_items = {str(k): v for k, v in desc_items_raw.items()}
                for i, (k_src, imgs_list) in enumerate(imgs_items, start=1):
                    total_idx = str(hist_N + i)
                    if total_idx not in merged_room.imgs:
                        merged_room.imgs[total_idx] = imgs_list
                    if total_idx not in merged_room.description:
                        merged_room.description[total_idx] = desc_items.get(k_src, desc_items.get(str(i), ''))
                rooms_map[merged_room.room_id] = merged_room  # HACK & NOTICE
        merged_floor.rooms = list(rooms_map.values())
        merged_floor._room_ids = [r.room_id for r in merged_floor.rooms]
        # 形状检查 (只对共有房间)
        if shape_check:
            warnings.extend(_validate_rooms(now_floor, hist_floor, iou_thresh=iou_thresh, area_ratio_tol=area_tol))
        floors_new.append(merged_floor)

    # ---- 添加新增楼层 (history 不存在) ---- #
    if allow_new_floors:
        for floor_id, now_floor in now_floor_map.items():
            if floor_id in history_floor_map:
                continue
            nf = copy.deepcopy(now_floor)
            # 初始化房间/对象 N
            for r in nf.rooms:
                r.N = r.N if (r.N and r.N > 0) else 1
                for o in r.objects:
                    if getattr(o, 'N', None) in (None, 0):
                        try: o.N = 1  # type: ignore
                        except Exception: pass
            floors_new.append(nf)
            warnings.append(f"[NEW_FLOOR_ADDED] floor={floor_id} rooms={len(nf.rooms)}")

    return floors_new, warnings

@hydra.main(config_path="config", config_name="map")
def main(config_map):
    # 默认路径 (完整 building 更新)
    now_path = 'RAG_Graph/map_now.json'
    hist_path = 'RAG_Graph/map_history.json'
    save_path = 'RAG_Graph/map_merged.json'
    floors_now = _load_floors_flexible(now_path)
    floors_history = _load_floors_flexible(hist_path)
    floors_new, warns = run_merge(floors_now, floors_history, config_map)
    with open(save_path, 'w') as f:
        json.dump([floor.to_dict() for floor in floors_new], f, ensure_ascii=False, indent=4)
    if warns:
        print("\n".join(warns))

if __name__ == '__main__':
    # CLI 模式: python semantic_map_Update/map_Update.py <now_json> <history_json> <save_json>
    # 例: python semantic_map_Update/map_Update.py RAG_Graph/map_now.json RAG_Graph/test_map/map_historyR1F2.json RAG_Graph/test_save/map_mergedR1F2.json
    if len(sys.argv) >= 4:
        now_arg = sys.argv[1]
        hist_arg = sys.argv[2]
        save_arg = sys.argv[3]
        try:
            import yaml  # type: ignore
            with open('config/map.yaml', 'r') as cf:
                _cfg = yaml.safe_load(cf)
        except Exception:
            _cfg = {'object': {}}
        floors_now = _load_floors_flexible(now_arg)
        floors_history = _load_floors_flexible(hist_arg)
        floors_new, warns = run_merge(floors_now, floors_history, _cfg)
        import os
        os.makedirs(os.path.dirname(save_arg) or '.', exist_ok=True)
        with open(save_arg, 'w') as f:
            json.dump([floor.to_dict() for floor in floors_new], f, ensure_ascii=False, indent=4)
        if warns:
            print("[ROOM_SHAPE_VALIDATION] #NOTICE-提醒如下:")
            print("\n".join(warns))
        else:
            print("[ROOM_SHAPE_VALIDATION] 所有提供房间形状通过阈值检查")
    else:
        try:
            import yaml  # type: ignore
            with open('config/map.yaml', 'r') as cf:
                _cfg = yaml.safe_load(cf)
        except Exception:
            _cfg = {'object': {}}
        main(_cfg)
