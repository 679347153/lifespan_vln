# DAAAM 代码阅读与对接指南（面向 AgenticRAG 实时部署）

## 已读材料（本笔记的证据来源）
- [参考开源代码/DAAAM/README.md](参考开源代码/DAAAM/README.md)
- [参考开源代码/DAAAM/CODEBASE.md](参考开源代码/DAAAM/CODEBASE.md)
- [参考开源代码/DAAAM/INSTALL.md](参考开源代码/DAAAM/INSTALL.md)
- [参考开源代码/DAAAM/RUNNING.md](参考开源代码/DAAAM/RUNNING.md)
- [计划/AgenticRAG_可执行技术规范_v2.md](计划/AgenticRAG_可执行技术规范_v2.md)
- [计划/当前代码结构总结.md](计划/当前代码结构总结.md)

## DAAAM 初步理解（给你的心智模型）
- DAAAM 是实时语义建图管线，核心由 `PipelineOrchestrator` 串联分割、跟踪、分配、语义描述与场景图更新。
- 核心库是可安装的 Python 包，实物部署通过独立的 ROS 2 接口仓库完成，两种运行模式共享同一管线逻辑。
- 主要配置入口是 [参考开源代码/DAAAM/config/pipeline_config.yaml](参考开源代码/DAAAM/config/pipeline_config.yaml) 和 [参考开源代码/DAAAM/config/pipeline_config_coda.yaml](参考开源代码/DAAAM/config/pipeline_config_coda.yaml)。
- 运行时输出为 Hydra 风格的动态场景图与语义修正记录，可继续做后处理、可视化和查询。

## 推荐阅读路径（目的导向）
- 先读 [参考开源代码/DAAAM/README.md](参考开源代码/DAAAM/README.md) 把握目标、外部依赖和 ROS2 入口。
- 再读 [参考开源代码/DAAAM/CODEBASE.md](参考开源代码/DAAAM/CODEBASE.md) 理清服务分工与 worker 插件模式。
- 读 [参考开源代码/DAAAM/RUNNING.md](参考开源代码/DAAAM/RUNNING.md) 了解端到端流程与 ROS2 运行方式。
- 读 [参考开源代码/DAAAM/INSTALL.md](参考开源代码/DAAAM/INSTALL.md) 明确完整工作空间依赖。
- 看 [参考开源代码/DAAAM/config/pipeline_config.yaml](参考开源代码/DAAAM/config/pipeline_config.yaml) 建立参数直觉。
- 读 [参考开源代码/DAAAM/src/daaam/config.py](参考开源代码/DAAAM/src/daaam/config.py) 理解配置结构如何进入代码。
- 读 [参考开源代码/DAAAM/src/daaam/pipeline/orchestrator.py](参考开源代码/DAAAM/src/daaam/pipeline/orchestrator.py) 追踪一帧数据如何串起各服务。

## 对接清单（DAAAM -> AgenticRAG）
- 输入与时序：确认 RGB、Depth、CameraInfo 与 TF 坐标链一致，深度尺度统一，时间戳可靠。
- 模型与算力：确定分割模型、跟踪 ReID 权重与 grounding 后端，结合显存预算选择分辨率和调用频率。
- 标签空间：明确 DAAAM 类别列表与 AgenticRAG 稳定性/白名单体系的映射关系。
- 结构映射：把 DAAAM 对象节点映射到 AgenticRAG 的 `obj_id`、`category`、`pos_3d`、`bbox_3d`、`clip_embedding`、`last_update_time` 字段。
- 语义增强：若 DAAAM 未输出你所需的 embedding 或描述，准备补充编码步骤，保证与 AgenticRAG 检索空间一致。
- 融合策略：对齐 [计划/AgenticRAG_可执行技术规范_v2.md](计划/AgenticRAG_可执行技术规范_v2.md) 中 `stability`、`exist_prob`、`cooccur_stats` 的更新规则。
- 运行节奏：确定 DAAAM 的语义查询频率与 AgenticRAG 的导航/评分频率如何协同，避免频繁重算。
- 输出落地：明确 DAAAM 场景图如何增量合并进 AgenticRAG 的长期记忆存储格式。

## 改造点（你项目侧需要做）
- 适配器：新增“DAAAM 场景图 → AgenticRAG 语义地图”的转换模块，保持增量更新而非全量重写。
- 稳定 ID：建立跨帧稳定的 `obj_id` 生成规则，避免同一物体反复新建。
- 标签映射：根据你的白名单与稳定性先验做类别映射与缺省处理。
- 时序与衰减：接入真实时间戳，驱动时间衰减与 `exist_prob`。
- 共现统计：基于图中邻接关系累计 `cooccur_stats`，供 RAG 评分使用。
- 深度探索衔接：明确 DAAAM 输出在你的 Stage 1/Stage 2 体系中扮演的角色，并标注 `source_mode`。
- 性能约束：若实时性不足，优先调整 `query_interval`、分割分辨率、`min_obs_per_track` 与 worker 并行度。

## 最小可行验证路线
- 先在 ROS2 或 rosbag 上跑通 DAAAM 全流程，确认场景图稳定产出。
- 再把场景图适配为 AgenticRAG 的语义地图结构，验证查询评分与导航输入无结构性错误。
- 最后引入实时更新与衰减机制，检查 `exist_prob` 与动态对象稳定性。

## 我可以继续帮你做的事
- 给出“DAAAM 场景图 → AgenticRAG 对象结构”的字段级映射模板。
- 根据你的 TF/话题/相机参数，给出 DAAAM-ROS 的最小可行运行参数。
- 结合你的 Stage 1/Stage 2 逻辑，把 DAAAM 接入点写成明确的工程任务清单。