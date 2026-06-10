"""P6: 消融实验配置 —— 定义各消融条件及其参数覆盖。

消融目标 (对齐论文 Table 1):
  Full         : 完整系统
  w/o NegGMM   : 禁用负向高斯场 → SSS 退化
  w/o Decay    : 禁用 exist_prob 时间衰减 → DO 物体不衰减, D-SR 退化
  w/o Stability: 禁用 stability prior → GMM 评分不含 stability 项
  w/o Merge    : 单次地图 (不 merge, 仅用 floors_now)
  SingleMap    : 单帧基线 (不做历史合并, 每 epoch 重建)

用法:
    from eval.ablation import ABLATION_CONFIGS, apply_ablation
    cfg = apply_ablation(base_config, "w/o NegGMM")
"""
from __future__ import annotations
import copy
from typing import Dict, Any


# ────────────── 消融配置注册表 ──────────────

ABLATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Full": {
        "description": "完整系统 (所有组件启用)",
        "neg_field_enabled": True,
        "enable_decay": True,
        "enable_stability": True,
        "enable_merge": True,
    },
    "w/o NegGMM": {
        "description": "禁用负向高斯场",
        "neg_field_enabled": False,
        "enable_decay": True,
        "enable_stability": True,
        "enable_merge": True,
    },
    "w/o Decay": {
        "description": "禁用 exist_prob 时间衰减 (missing_decay=1.0)",
        "neg_field_enabled": True,
        "enable_decay": False,
        "enable_stability": True,
        "enable_merge": True,
        "config_overrides": {
            "missing_decay": 1.0,    # 不衰减
        },
    },
    "w/o Stability": {
        "description": "禁用 stability prior (所有物体 stability=0.5)",
        "neg_field_enabled": True,
        "enable_decay": True,
        "enable_stability": False,
        "enable_merge": True,
    },
    "w/o Merge": {
        "description": "不做历史 merge, 仅用当帧 floors_now",
        "neg_field_enabled": True,
        "enable_decay": True,
        "enable_stability": True,
        "enable_merge": False,
    },
    "SingleMap": {
        "description": "单帧基线 (等同 w/o Merge, 额外禁用 NegGMM)",
        "neg_field_enabled": False,
        "enable_decay": True,
        "enable_stability": True,
        "enable_merge": False,
    },
}


def apply_ablation(base_config: dict, ablation_name: str) -> dict:
    """将消融配置覆盖到基础 map.yaml 配置上.

    Args:
        base_config: 原始 map.yaml 配置 dict
        ablation_name: 消融条件名称 (ABLATION_CONFIGS 的键)

    Returns:
        修改后的配置 dict (深拷贝, 不影响原始)

    Raises:
        KeyError: 未知的消融条件
    """
    if ablation_name not in ABLATION_CONFIGS:
        raise KeyError(
            f"未知消融条件: {ablation_name!r}, "
            f"可用: {list(ABLATION_CONFIGS.keys())}")

    abl = ABLATION_CONFIGS[ablation_name]
    cfg = copy.deepcopy(base_config)

    # 应用 config_overrides (如 missing_decay)
    overrides = abl.get("config_overrides", {})
    for k, v in overrides.items():
        cfg[k] = v

    # stability 禁用: 统一设为 0.5
    if not abl["enable_stability"]:
        cfg["_force_stability"] = 0.5

    return cfg


def get_nav_sim_kwargs(ablation_name: str) -> dict:
    """获取传给 LightweightNavSim 构造器的参数覆盖.

    Returns:
        {"neg_field_enabled": bool, ...}
    """
    abl = ABLATION_CONFIGS.get(ablation_name, ABLATION_CONFIGS["Full"])
    return {
        "neg_field_enabled": abl["neg_field_enabled"],
    }


def list_ablations() -> None:
    """打印所有可用的消融配置."""
    for name, abl in ABLATION_CONFIGS.items():
        print(f"  [{name:16s}] {abl['description']}")
