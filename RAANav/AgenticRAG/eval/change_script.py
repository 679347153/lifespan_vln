"""P1: 变更脚本解析器 —— 定义并加载场景的跨 epoch 物体变化描述。

change_script.yaml 格式:
```yaml
scene_id: "00814-p53SfW6mjZe"
n_epochs: 5
epoch_days: [1, 3, 7, 14, 30]

changes:
  - obj_id: "chair_1_R0"
    label: "chair"
    type: HRO            # SO | LRO | HRO | DO | NO
    positions:
      T0: {room: R0, pos_3d: [-15.55, 1.49, -5.98]}
      T1: {room: R1, pos_3d: [-1.50, 1.49, 5.00]}
      T2: {room: R0, pos_3d: [-15.00, 1.49, -6.20]}
      T3: {room: R1, pos_3d: [-1.20, 1.49, 4.80]}
      T4: {room: R0, pos_3d: [-15.55, 1.49, -5.98]}
  - obj_id: "box_3_R0"
    label: "box"
    type: DO
    positions:
      T0: {room: R0, pos_3d: [-10.0, 1.0, -3.0]}
      T2: null            # 消失: T2 起不出现
  - obj_id: null          # NO: 首次出现时才分配 obj_id
    label: "speaker"
    type: NO
    positions:
      T3: {room: R0, pos_3d: [-12.0, 1.5, -4.0]}
```

用法:
    from eval.change_script import ChangeScript
    cs = ChangeScript.from_yaml("eval/scenarios/00814_changes.yaml")
    moved_ids = cs.get_moved_obj_ids(epoch=2)
    layout = cs.get_epoch_layout(epoch=2, base_floors=floors_gt)
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml


# ────────────── 数据结构 ──────────────


@dataclass
class ObjectChange:
    """单个物体在各 epoch 的位置变化描述."""
    obj_id: Optional[str]          # SO/LRO/HRO/DO 有原始 ID; NO 为 None
    label: str
    change_type: str               # SO | LRO | HRO | DO | NO
    positions: Dict[int, Optional[Dict]]  # epoch -> {room, pos_3d} 或 None(消失)

    @property
    def first_epoch(self) -> int:
        """物体首次出现的 epoch."""
        for e in sorted(self.positions):
            if self.positions[e] is not None:
                return e
        return 0

    @property
    def last_epoch(self) -> Optional[int]:
        """物体最后出现的 epoch (DO 消失后返回最后可见 epoch)."""
        last = None
        for e in sorted(self.positions):
            if self.positions[e] is not None:
                last = e
        return last

    def is_present(self, epoch: int) -> bool:
        """该物体在指定 epoch 是否存在."""
        if epoch in self.positions:
            return self.positions[epoch] is not None
        # 未显式列出的 epoch: SO/LRO/HRO 沿用上一个非 null 位置; DO/NO 看上下文
        if self.change_type == 'NO':
            return epoch >= self.first_epoch
        if self.change_type == 'DO':
            # DO: 在最后可见 epoch 之后消失
            return self.last_epoch is not None and epoch <= self.last_epoch
        # SO/LRO/HRO: 始终存在
        return True

    def get_pos_3d(self, epoch: int) -> Optional[List[float]]:
        """获取指定 epoch 的 pos_3d (未显式列出则向前查找最近的)."""
        if not self.is_present(epoch):
            return None
        # 直接命中
        if epoch in self.positions and self.positions[epoch] is not None:
            return self.positions[epoch]['pos_3d']
        # 向前查找最近的已知位置
        for e in sorted(self.positions.keys(), reverse=True):
            if e < epoch and self.positions[e] is not None:
                return self.positions[e]['pos_3d']
        return None

    def get_room(self, epoch: int) -> Optional[str]:
        """获取指定 epoch 的 room_id."""
        if not self.is_present(epoch):
            return None
        if epoch in self.positions and self.positions[epoch] is not None:
            return self.positions[epoch].get('room')
        for e in sorted(self.positions.keys(), reverse=True):
            if e < epoch and self.positions[e] is not None:
                return self.positions[e].get('room')
        return None

    def has_moved(self, epoch: int, dist_threshold: float = 1.0) -> bool:
        """该物体在 epoch 相比前一个 epoch 是否发生了超阈值移动."""
        if epoch == 0:
            return False
        pos_now = self.get_pos_3d(epoch)
        pos_prev = self.get_pos_3d(epoch - 1)
        if pos_now is None or pos_prev is None:
            return False
        dist = sum((a - b) ** 2 for a, b in zip(pos_now, pos_prev)) ** 0.5
        return dist > dist_threshold


# ────────────── 主类 ──────────────


class ChangeScript:
    """场景变更脚本: 描述一个场景各 epoch 的物体变化."""

    def __init__(self, scene_id: str, n_epochs: int,
                 epoch_days: List[int],
                 changes: List[ObjectChange]):
        self.scene_id = scene_id
        self.n_epochs = n_epochs
        self.epoch_days = epoch_days
        self.changes = changes
        # 按 obj_id 建索引 (NO 物体可能无 obj_id)
        self._by_id: Dict[str, ObjectChange] = {}
        for c in changes:
            if c.obj_id:
                self._by_id[c.obj_id] = c

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'ChangeScript':
        """从 YAML 文件加载变更脚本."""
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)

        scene_id = raw['scene_id']
        n_epochs = raw.get('n_epochs', 5)
        epoch_days = raw.get('epoch_days', list(range(n_epochs)))

        changes = []
        for item in raw.get('changes', []):
            positions = {}
            for key, val in item.get('positions', {}).items():
                # key: "T0", "T1" etc.
                epoch_idx = int(str(key).replace('T', ''))
                if val is None:
                    positions[epoch_idx] = None
                else:
                    positions[epoch_idx] = {
                        'room': val.get('room'),
                        'pos_3d': val.get('pos_3d'),
                    }
            # YAML 会把裸 NO 解析为 False, DO 解析为字符串 — 统一处理
            raw_type = item['type']
            if raw_type is False:
                raw_type = 'NO'
            elif raw_type is True:
                raw_type = 'YES'
            change_type = str(raw_type).upper()

            changes.append(ObjectChange(
                obj_id=item.get('obj_id'),
                label=item['label'],
                change_type=change_type,
                positions=positions,
            ))

        return cls(scene_id, n_epochs, epoch_days, changes)

    # ──── 查询接口 ────

    def get_changes_by_type(self, change_type: str) -> List[ObjectChange]:
        """按类型筛选: SO, LRO, HRO, DO, NO."""
        return [c for c in self.changes if c.change_type == change_type]

    def get_moved_obj_ids(self, epoch: int,
                          dist_threshold: float = 1.0) -> List[str]:
        """返回在指定 epoch 发生超阈值移动的物体 ID 列表."""
        result = []
        for c in self.changes:
            if c.obj_id and c.has_moved(epoch, dist_threshold):
                result.append(c.obj_id)
        return result

    def get_disappeared_obj_ids(self, epoch: int) -> List[str]:
        """返回在指定 epoch 消失的物体 ID 列表 (前一 epoch 存在, 本 epoch 不存在)."""
        result = []
        for c in self.changes:
            if c.obj_id and c.change_type == 'DO':
                if epoch > 0 and c.is_present(epoch - 1) and not c.is_present(epoch):
                    result.append(c.obj_id)
        return result

    def get_new_obj_labels(self, epoch: int) -> List[Tuple[str, List[float], str]]:
        """返回在指定 epoch 首次出现的新物体: [(label, pos_3d, room_id), ...]."""
        result = []
        for c in self.changes:
            if c.change_type == 'NO' and c.first_epoch == epoch:
                pos = c.get_pos_3d(epoch)
                room = c.get_room(epoch)
                if pos:
                    result.append((c.label, pos, room or 'R0'))
        return result

    def get_all_present_at_epoch(self, epoch: int) -> List[ObjectChange]:
        """返回指定 epoch 存在的所有变更物体."""
        return [c for c in self.changes if c.is_present(epoch)]

    def summary(self) -> str:
        """打印变更脚本摘要."""
        lines = [f"ChangeScript: {self.scene_id}",
                 f"  Epochs: {self.n_epochs}, Days: {self.epoch_days}"]
        from collections import Counter
        type_counts = Counter(c.change_type for c in self.changes)
        lines.append(f"  Changes: {dict(type_counts)}")
        for epoch in range(self.n_epochs):
            moved = self.get_moved_obj_ids(epoch)
            disappeared = self.get_disappeared_obj_ids(epoch)
            new = self.get_new_obj_labels(epoch)
            lines.append(f"  Epoch {epoch} (Day {self.epoch_days[epoch]}): "
                         f"moved={len(moved)}, disappeared={len(disappeared)}, "
                         f"new={len(new)}")
        return '\n'.join(lines)
