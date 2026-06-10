"""Object 数据模型 (含关系 R_objs, N, imgs, description).

R_objs 关系结构(方案1: TypedDict):
    R_objs: Dict[str, RelationMeta]
    RelationMeta: { 'Nt': int, 'Nr': int, 'Rcfd': float }
含义:
    Nt   : 目标对象(target)自身的最新版本/更新计数(仅当更大时才更新, 防止旧版本覆盖)
    Nr   : 与当前对象共现/相关的累计次数(统计频次)
    Rcfd : 相关置信度 (0~1)
向后兼容: 旧数据若缺 Nt, 自动补 Nt=0; 如果键大小写不同 (rcfd) 也会兼容。
"""
from __future__ import annotations
from typing import List, Dict, Optional, Any, TypedDict

class RelationMeta(TypedDict, total=False):
    Nt: int
    Nr: int
    Rcfd: float

class Object:
    def __init__(
        self,
        obj_id: str,
        label: str,
        region: List[Dict[str, float]],
        stability: Optional[float] = None,
        clip_embedding: Optional[List[float]] = None,
        cfd: Optional[float] = None,
        room_id: Optional[str] = None,
        R_objs: Optional[Dict[str, RelationMeta]] = None,
        # R_objs: { target_obj_id: { 'Nt': int, 'Nr': int, 'Rcfd': float } }
        # Nt: 目标对象(target)的最新更新次数(或版本号)用于判断目标对象自身的时序新鲜度
        # Nr: 与当前对象共现/相关的累计次数 (相关性频次, 统计量)
        # Rcfd: 当前对象与 target 对象的相关置信度(0~1), 可以根据最近共现衰减/增强，这里注释掉了是因为我后面代码是使用时计算的吗？
        imgs: Optional[Dict[str, List[str]]] = None,  # FIX: 统一 imgs 为 dict[str, list[str]] 结构
        N: Optional[int] = None,
        description: Optional[Dict[str, List[str]]] = None,
        last_update_time: Optional[Any] = None,
        cooccur_stats: Optional[Dict[str, Any]] = None,
        exist_prob: Optional[float] = None,
        pos_3d: Optional[List[float]] = None,
        pos_2d: Optional[Any] = None,  # dict {"x":..,"y":..} 或 list [x,z]
        bbox_3d: Optional[Dict[str, Any]] = None,
    ):
        self.obj_id = obj_id
        self.label = label
        self.region = region
        self.stability = stability
        self.clip_embedding = clip_embedding or []
        self.cfd = cfd
        self.room_id = room_id
        # 不补齐缺失字段；仅保留已完整 (Nt/Nr/Rcfd) 的关系元数据
        fixed_R_objs: Dict[str, RelationMeta] = {}
        for k, meta in (R_objs or {}).items():
            if not isinstance(meta, dict):
                continue
            if all(x in meta for x in ('Nt','Nr','Rcfd')):
                fixed_R_objs[k] = { 'Nt': meta['Nt'], 'Nr': meta['Nr'], 'Rcfd': meta['Rcfd'] }  # type: ignore[assignment]
        self.R_objs = fixed_R_objs

        # NOTE imgs 输入兼容：若传入 list（旧格式）则转换为{'1': list}
        if isinstance(imgs, list):  # type: ignore[arg-type]
            # 旧 JSON 里对象 imgs 是列表单版本
            self.imgs: Dict[str, List[str]] = {'1': imgs}
        elif imgs is None:
            self.imgs = {}
        else:
            self.imgs = imgs

        self.N = N
        self.description = description
        self.last_update_time = last_update_time
        self.cooccur_stats = cooccur_stats if isinstance(cooccur_stats, dict) else {}
        if exist_prob is None:
            self.exist_prob = 1.0
        else:
            try:
                ep = float(exist_prob)
            except Exception:
                ep = 1.0
        self.exist_prob = max(0.0, min(1.0, ep))
        self.pos_3d = pos_3d
        self.pos_2d = pos_2d
        self.bbox_3d = bbox_3d

    def add_R_objs(self, target_id: str, Nt: Optional[int] = None, Rcfd: Optional[float] = 1.0, Nr_inc: int = 1) -> None:
        """更新self的R_objs统计信息.

        参数:
            target_id: 目标对象 ID
            Nt: (可选) 目标对象当前版本/更新次数, 若提供则覆盖保存
            Nr_inc: 相关次数
            Rcfd: (可选) 新的相关置信度 (0 or 1)
        新数返回，不传入Nt、Nr：target_id, {"Nt": 0, "Nr": 1, "Rcfd": 1.0}

        NOTICE 新得的物体，Nt也是1，但是后面与obj_history合并时会更新所以此处放着没有写
        """
        entry = self.R_objs.setdefault(target_id, {"Nt": 0, "Nr": 0, "Rcfd": 0.0})  # type: ignore[arg-type]
        if Nt is not None and isinstance(Nt, int) and Nt >= entry.get('Nt', 0):
            entry['Nt'] = Nt

        entry['Nr'] = int(entry.get('Nr', 0)) + max(0, Nr_inc)

        if entry['Nr'] > entry['Nt']:
            entry['Nt'] = entry['Nr']

        if Rcfd is not None:
            try:
                rcfd_val = float(Rcfd)
            except Exception:
                rcfd_val = entry.get('Rcfd', 0.0)
            entry['Rcfd'] = max(0.0, min(1.0, rcfd_val))
        

    def load_embedding(self):
        pass

    def to_dict(self) -> Dict[str, Any]:
        data = {
            'obj_id': self.obj_id,
            'label': self.label,
            'region': self.region,
            'stability': self.stability,
            'clip_embedding': self.clip_embedding if self.clip_embedding else [],
            'cfd': self.cfd,
            'room_id': self.room_id,
            'R_objs': self.R_objs or {},
            'imgs': self.imgs,  # dict 结构
            'N': self.N,
            'description': self.description,
            'last_update_time': self.last_update_time,
            'cooccur_stats': self.cooccur_stats,
            'exist_prob': self.exist_prob,
            'pos_3d': self.pos_3d,
            'pos_2d': self.pos_2d,
            'bbox_3d': self.bbox_3d,
        }
        return {k: v for k, v in data.items() if v is not None}

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> 'Object':
        label = data_dict.get('label', data_dict.get('name'))
        obj_id = data_dict['obj_id']
        region = data_dict.get('region', data_dict.get('polygon', data_dict.get('Region', [])))
        # NOTE 向后兼容: 若 imgs 是 list 则构造时会自动封装
        imgs_val = data_dict.get('imgs')
        if isinstance(imgs_val, list):
            imgs_val = {'1': imgs_val}
        return cls(
            obj_id=obj_id,
            label=label,
            region=region,
            stability=data_dict.get('stability'),
            clip_embedding=data_dict.get('clip_embedding'),
            cfd=data_dict.get('cfd'),
            room_id=data_dict.get('room_id'),
            R_objs=data_dict.get('R_objs'),
            imgs=imgs_val,
            N=data_dict.get('N'),
            description=data_dict.get('description'),
            last_update_time=data_dict.get('last_update_time'),
            cooccur_stats=data_dict.get('cooccur_stats'),
            exist_prob=data_dict.get('exist_prob', 1.0),
            pos_3d=data_dict.get('pos_3d'),
            pos_2d=data_dict.get('pos_2d'),
            bbox_3d=data_dict.get('bbox_3d'),
        )

    def __repr__(self) -> str:
        rel_count = len(self.R_objs)
        return (
            f"Object(id='{self.obj_id}', label='{self.label}', region_pts={len(self.region)}, "
            f"related_objs={rel_count}, N={self.N}, desc={self.description})"
        )
