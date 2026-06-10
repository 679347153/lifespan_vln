"""虚拟时钟: 将仿真步骤映射为可配置的虚拟时间.

在仿真闭环中, 真实帧间隔 ≈ 0 秒, 导致时间衰减
exp(-λ·ΔT) ≈ 1.0 完全无效. VirtualClock 允许每个仿真
步骤推进可配置的虚拟时间 (如 1步 = 1小时), 使衰减有意义.

用法:
    clock = VirtualClock(step_hours=1.0)  # 每步 1 小时
    ts_0 = clock.now_iso()                # 初始时间
    clock.step()                          # 推进 1 步
    ts_1 = clock.now_iso()                # 1小时后

    # 或指定起始时间
    clock = VirtualClock(
        start="2026-01-01T00:00:00Z",
        step_hours=6.0,              # 每步 6 小时
    )

配置集成 (config/map.yaml):
    virtual_clock:
      enabled: true
      step_hours: 1.0
      start: "2026-01-01T00:00:00Z"
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Union


class VirtualClock:
    """可控虚拟时钟, 按步推进."""

    def __init__(
        self,
        step_hours: float = 1.0,
        start: Optional[Union[str, datetime]] = None,
    ):
        """
        Args:
            step_hours: 每步推进的虚拟小时数.
            start: 起始时间 (ISO字符串或datetime). 默认 2026-01-01T00:00:00Z.
        """
        self._step_delta = timedelta(hours=step_hours)
        self._step_count = 0

        if start is None:
            self._current = datetime(2026, 1, 1, tzinfo=timezone.utc)
        elif isinstance(start, str):
            s = start.replace("Z", "+00:00")
            self._current = datetime.fromisoformat(s)
            if self._current.tzinfo is None:
                self._current = self._current.replace(tzinfo=timezone.utc)
        else:
            self._current = start
            if self._current.tzinfo is None:
                self._current = self._current.replace(tzinfo=timezone.utc)

        self._start = self._current

    # ---- 核心 API ---- #

    def step(self, n: int = 1) -> "VirtualClock":
        """推进 n 步."""
        self._current += self._step_delta * n
        self._step_count += n
        return self

    def now(self) -> datetime:
        """当前虚拟时间 (datetime)."""
        return self._current

    def now_iso(self) -> str:
        """当前虚拟时间 (ISO 8601 UTC 字符串)."""
        return self._current.strftime("%Y-%m-%dT%H:%M:%SZ")

    def now_timestamp(self) -> float:
        """当前虚拟时间 (UNIX 时间戳)."""
        return self._current.timestamp()

    def elapsed_hours(self) -> float:
        """从起始到现在的虚拟小时数."""
        return (self._current - self._start).total_seconds() / 3600.0

    def elapsed_days(self) -> float:
        """从起始到现在的虚拟天数."""
        return self.elapsed_hours() / 24.0

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def step_hours(self) -> float:
        return self._step_delta.total_seconds() / 3600.0

    # ---- 辅助 ---- #

    def reset(self) -> "VirtualClock":
        """重置到起始时间."""
        self._current = self._start
        self._step_count = 0
        return self

    def __repr__(self) -> str:
        return (
            f"VirtualClock(step={self.step_hours}h, "
            f"step_count={self._step_count}, "
            f"now={self.now_iso()})"
        )


# ---- 工厂函数: 从 config 创建 ---- #

def clock_from_config(cfg: dict) -> Optional[VirtualClock]:
    """从 map.yaml 的 virtual_clock 配置创建时钟.
    
    Returns:
        VirtualClock 实例, 若未启用返回 None.
    """
    vc_cfg = cfg.get("virtual_clock") or {}
    if not vc_cfg.get("enabled", False):
        return None
    return VirtualClock(
        step_hours=float(vc_cfg.get("step_hours", 1.0)),
        start=vc_cfg.get("start"),
    )


# ---- 兼容层: 统一获取时间戳 ---- #

_global_clock: Optional[VirtualClock] = None


def set_global_clock(clock: Optional[VirtualClock]):
    """设置全局虚拟时钟 (供 object_Update 等透明使用)."""
    global _global_clock
    _global_clock = clock


def get_global_clock() -> Optional[VirtualClock]:
    return _global_clock


def now_iso_utc() -> str:
    """返回当前时间戳 (ISO UTC 字符串).
    
    如果设置了全局虚拟时钟, 使用虚拟时间;
    否则使用真实系统时间.
    """
    if _global_clock is not None:
        return _global_clock.now_iso()
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
