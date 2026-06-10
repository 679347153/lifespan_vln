"""Floor 数据模型 (含 description)."""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from .room import Room
import json


class Floor:
    def __init__(
        self,
        floor_id: str,
        rooms: Optional[List[Room]] = None,
        z_range: Optional[Dict[str, float]] = None,
        room_ids: Optional[List[str]] = None,
        description: Optional[str] = None,
    ):
        self.floor_id = floor_id
        self.rooms = rooms or []
        self.z_range = z_range or {'z_min': 0.0, 'z_max': 4.0}
        self._room_ids = room_ids or [r.room_id for r in self.rooms]
        self.description = description

    @property
    def room_ids(self) -> List[str]:
        return list(self._room_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'floor_id': self.floor_id,
            'z_range': self.z_range,
            'rooms': [room.to_dict() for room in self.rooms],
            'room_ids': self.room_ids,
            'description': self.description,
        }

    @classmethod
    def from_json(cls, path: str) -> 'List[Floor]':
        with open(path, 'r') as f:
            data = json.load(f)
        return [cls.from_dict(item) for item in data]

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> 'Floor':
        rooms_data = data_dict.get('rooms', [])
        rooms = [Room.from_dict(r) for r in rooms_data]
        z_range = data_dict.get('z_range') or {'z_min': 0.0, 'z_max': 4.0}
        return cls(
            floor_id=data_dict['floor_id'],
            rooms=rooms,
            z_range=z_range,
            room_ids=data_dict.get('room_ids'),
            description=data_dict.get('description'),
        )

    def add_room(self, room: Room):
        self.rooms.append(room)
        if room.room_id not in self._room_ids:
            self._room_ids.append(room.room_id)

    def __repr__(self) -> str:
        return f"Floor(id='{self.floor_id}', rooms={len(self.rooms)}, z_range={self.z_range}, desc={self.description})"
