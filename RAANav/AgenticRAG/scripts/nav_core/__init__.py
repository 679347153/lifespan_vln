"""nav_core — sim_nav_loop 模块拆分.

从 sim_nav_loop.py 中提取的可复用组件:
  habitat_agent: HabitatAgent 封装
  nav_strategy:  NegativeGaussianField, frontier_exploration_step
  perception:    物体构建, CLIP 编码, 帧内去重, room 分配
  visualization: 轨迹可视化
"""
from scripts.nav_core.habitat_agent import HabitatAgent  # noqa: F401
from scripts.nav_core.nav_strategy import NegativeGaussianField, frontier_exploration_step  # noqa: F401
from scripts.nav_core.perception import (  # noqa: F401
    STRUCTURAL_CATEGORIES,
    objects_from_observation,
    objects_from_detection,
    compute_clip_embeddings_for_detections,
    dedup_intra_frame,
    assign_room_ids,
    build_floors_now,
)
from scripts.nav_core.visualization import (  # noqa: F401
    visualize_trajectory_on_occ_grid,
    visualize_trajectory,
)
