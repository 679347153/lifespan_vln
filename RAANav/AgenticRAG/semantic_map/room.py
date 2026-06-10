"""Room 数据模型 (含 door_positions, N, imgs, description)."""
from __future__ import annotations
from typing import List, Dict, Optional
from .object import Object


class Room:
    def __init__(
        self,
        room_id: str,
        room_name: Optional[Dict[str, str]] = None,
        objects: Optional[List[Object]] = None,
        region: Optional[List[Dict[str, float]]] = None,
        floor_id: Optional[str] = None,
        obj_ids: Optional[List[str]] = None,
        door_positions: Optional[List[Dict[str, float]]] = None,
        N: Optional[int] = None,
        imgs: Optional[Dict[str, List[str]]] = None,
        description: Optional[Dict[str, str]] = None,
    ):
        self.room_id = room_id
        # 统一 room_name 为 dict[str,str]
        if isinstance(room_name, dict):
            self.room_name = room_name
        elif isinstance(room_name, str):
            self.room_name = {'1': room_name}
        else:
            self.room_name = {}
        self.objects = objects or []
        self.region = region or []
        self.floor_id = floor_id
        self._obj_ids = obj_ids or [o.obj_id for o in self.objects]
        self.door_positions = door_positions or []
        self.N = N
        # 统一 imgs/description/room_name 为 dict[str, ...] 结构 (上游已经做迁移); 若仍旧格式则回退空 dict
        if isinstance(imgs, dict):
            self.imgs = imgs
        elif isinstance(imgs, list):  # 旧格式: list → {'1': list}
            self.imgs = {str(i): img for i, img in enumerate(imgs, start=1)}
        else:
            self.imgs = {}
        # 描述字段统一为 dict[str,str]
        if isinstance(description, dict):
            self.description = description
        elif isinstance(description, str):
            self.description = {'1': description}
        else:
            self.description = {}

    @property
    def obj_ids(self) -> List[str]:
        return list(self._obj_ids)

    def to_dict(self) -> Dict[str, any]:
        data = {
            'room_id': self.room_id,
            'room_name': self.room_name,
            'objects': [obj.to_dict() for obj in self.objects],
            'Region': self.region if self.region else [],
            'floor_id': self.floor_id,
            'obj_ids': self.obj_ids,
            'door_positions': self.door_positions if self.door_positions else [],
            'N': self.N,
            'imgs': self.imgs,
            'description': self.description,
        }
        return {k: v for k, v in data.items() if v is not None}
    # 返回的字典中不包含None值的键值对，也就是说这里只是做了一个筛选

    @classmethod
    def from_dict(cls, data_dict: Dict[str, any]) -> 'Room':
        objects_data = data_dict.get('objects', [])
        objects = [Object.from_dict(o) for o in objects_data]
        region = data_dict.get('Region', data_dict.get('region', []))
        return cls(
            room_id=data_dict['room_id'],
            room_name=data_dict.get('room_name', data_dict['room_id']),
            objects=objects,
            region=region,
            floor_id=data_dict.get('floor_id'),
            obj_ids=data_dict.get('obj_ids'),
            door_positions=data_dict.get('door_positions'),
            N=data_dict.get('N'),
            imgs=data_dict.get('imgs'),
            description=data_dict.get('description'),
        )

    def add_object(self, obj: Object):
        self.objects.append(obj)
        if obj.obj_id not in self._obj_ids:
            self._obj_ids.append(obj.obj_id)

    def __repr__(self) -> str:
        return (
            f"Room(id='{self.room_id}', name='{self.room_name}', objects={len(self.objects)}, "
            f"region_pts={len(self.region)}, doors={len(self.door_positions)}, N={self.N}, desc={self.description})"
        )
