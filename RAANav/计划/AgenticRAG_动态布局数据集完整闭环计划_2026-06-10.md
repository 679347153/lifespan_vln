# Agentic-RAG 动态布局数据集完整闭环计划

> 日期：2026-06-10  
> 目标：在已完成离线 MVP 的基础上，继续接入真实 Habitat-Sim 闭环、CLIP 图像检索和正式 Agent/RAG 融合评分，形成可用于论文实验的完整动态 ObjectNav 系统。

---

## 1. 当前基线

已完成离线 MVP：

- `benchmark_adapter` 可读取 `benchmark_split_longterm_v1.json`。
- 可根据 `seen_layout_count_before` 构建历史 semantic memory。
- 可将 layout object 转为 RAANav `Floor/Room/Object`。
- 可生成 pseudo trajectory JSONL。
- 可复用 `benchmark.evaluate` 输出 SR/SPL/exploration curve。
- 已在 `longterm_v1` val 全量 150 episodes 上跑通。

当前不足：

- 仍是离线轻量搜索，不是真实 Habitat-Sim 闭环。
- `image_goal` 暂按 label 查询，没有真正使用图像 embedding。
- 评分仍是 lightweight heuristic，不是正式 `S_rag + S_agent` 融合。
- 负向反馈只在候选访问层面模拟，没有真实视觉 miss、视锥覆盖、节点级 `exist_prob` 更新。

---

## 2. 完整闭环目标

最终系统应支持：

1. 从 benchmark episode 启动 Habitat-Sim 环境。
2. 加载当前 layout 中的动态物体。
3. 根据 `seen_layout_count_before` 构建历史长期记忆。
4. 对三类任务执行统一查询：
   - `open_vocab`
   - `image_goal`
   - `language_goal`
5. 使用正式 RAG 通道计算 `S_rag`。
6. 使用 Agent 通道计算 room/receptacle/object prior。
7. 动态融合 `S = w_rag * S_rag + w_agent * S_agent`。
8. 生成 GMM 概率地图和 Top-K 探索点。
9. 在 Habitat-Sim 中导航到候选点并观测。
10. 若目标未发现，触发负向 GMM 和节点级 `exist_prob` 更新。
11. 输出标准 trajectory JSONL。
12. 复用 benchmark evaluator 输出 SR/SPL，同时输出 RAANav memory diagnostics。

---

## 3. 阶段 A：真实 Habitat-Sim 闭环

### 3.1 目标

将当前 pseudo trajectory 替换为真实 Habitat-Sim episode runner。

### 3.2 实现内容

- 新增或扩展：
  - `RAANav/AgenticRAG/benchmark_adapter/habitat_closed_loop_agent.py`
  - `RAANav/AgenticRAG/benchmark_adapter/run_closed_loop.py`
- 基于现有 `benchmark.runner.AgentAdapter` 协议实现 RAANav agent：
  - `reset(episode)`
  - `act(observation, subtask)`
  - `on_subtask_done(episode, subtask, trace)`
- 使用 `benchmark.habitat_adapter.HabitatLayoutAdapter`：
  - 解析 HM3D scene
  - 加载 layout objects
  - 采样或读取 start pose
  - 计算 geodesic distance
- 将 RAANav Top-K 探索点转成 Habitat 可导航目标：
  - 使用 PathFinder snap point
  - 失败时寻找最近 navigable point
  - 保留 Euclidean fallback 仅用于 smoke test

### 3.3 验收

- 能运行单个 `longterm_v1` episode。
- trajectory JSONL 中每个 subtask 都有真实 `final_pose/path_length/steps`。
- `benchmark.evaluate --no-euclidean-fallback` 在 Habitat 可用时能运行。
- 无 Habitat 环境时仍保留离线 MVP 路径。

---

## 4. 阶段 B：CLIP 图像检索

### 4.1 目标

让 `image_goal` 不再退化为 label 查询，而是真正使用 `objects_images/*.webp` 和 memory object visual/text embedding 做检索。

### 4.2 实现内容

- 新增：
  - `RAANav/AgenticRAG/benchmark_adapter/image_goal_index.py`
- 支持两类 embedding：
  - reference image embedding：来自 `subtask.image_path`
  - memory object embedding：来自 layout object 对应 `objects_images/<stem>.webp`，或 Habitat 渲染 crop
- 复用现有 `CLIP_RAG` 模块：
  - `build_clip_index.py`
  - `query_clip_index.py`
  - `runtime_clip_index.py`
- 统一 query encoder：
  - `open_vocab`：text embedding
  - `image_goal`：image embedding
  - `language_goal`：text embedding + metadata prior
- 输出 `R_sim`：
  - text query：text-to-text 或 text-to-image similarity
  - image query：image-to-image similarity

### 4.3 验收

- `image_goal` 查询不依赖 `target_object` label 也能返回候选。
- `objects_images/*.webp` 缺失时有明确 warning 和 label fallback。
- 输出 diagnostics 中包含 `query_modality=image`、`clip_topk`、`R_sim`。

---

## 5. 阶段 C：正式 Agent/RAG 融合评分

### 5.1 目标

替换当前 lightweight heuristic，落地规范中的正式评分：

```text
S_base = beta1 * R_sim + beta2 * R_cfd
G_ts = exp(-lambda * delta_t / (1 + kappa * stability))
S_rag = S_base * G_ts
S = w_rag(n) * S_rag + w_agent(n) * S_agent
```

### 5.2 RAG 通道

每个候选 object 计算：

- `R_sim`
  - 来自 CLIP text/image 检索
- `R_cfd`
  - 来自 `R_objs/cooccur_stats`
  - 同 room/receptacle/near_2d 关系加权
- `delta_t`
  - 当前 state index 与 object last_update_time 的差
- `stability`
  - 来自 `stability_priors.yaml` 或 placement class fallback
- `exist_prob`
  - 初始 1.0
  - 负向反馈后降低

### 5.3 Agent 通道

Agent 输入：

- query 文本或 image goal caption/label
- 当前 memory room inventory
- candidate summary
- language goal 中的 `sampled_region_id/target_instance_id`
- 近期失败候选点

Agent 输出：

- `room_priors`
- `receptacle_priors`
- `obj_scores`
- 简短 reason

优先复用：

- `RAANav/AgenticRAG/GMM_map_Create/agent_Rscore.py`
- `score_for_navigation`

### 5.4 动态融合

实现：

```text
w_agent(n) = max(w_min, exp(-k * n))
w_rag(n) = 1 - w_agent(n)
```

其中 `n` 是当前 episode/subtask 内已访问候选点或查询轮次。

### 5.5 验收

- diagnostics 中每个候选包含：
  - `R_sim`
  - `R_cfd`
  - `G_ts`
  - `S_rag`
  - `S_agent`
  - `w_agent`
  - `w_rag`
  - `S_final`
- 可通过配置关闭 Agent 或 RAG 做消融。

---

## 6. 阶段 D：GMM 概率地图与导航目标

### 6.1 目标

从正式 `S_final` 生成概率地图，并把 Top-K 峰值变成可导航点。

### 6.2 实现内容

- 复用或扩展：
  - `GMM_map_Create/GMM_map_calcualte.py`
  - `eval/gmm_evaluator.py`
  - `scripts/nav_core/nav_strategy.py`
- 支持 occupancy mask：
  - Habitat PathFinder navmesh -> free mask
  - 无 Habitat 时 Euclidean canvas fallback
- Top-K 提取：
  - NMS
  - 最小间距
  - 同楼层过滤
  - navigable snap

### 6.3 验收

- 每次查询输出：
  - probability field metadata
  - Top-K candidate points
  - candidate score breakdown
- Habitat 模式下 candidate point 可被 PathFinder 接受或 snap 到可导航点。

---

## 7. 阶段 E：负向反馈与闭环更新

### 7.1 目标

真实搜索失败后，不只是访问下一个点，而是更新 memory 和概率场。

### 7.2 实现内容

- 失败判定：
  - 到达候选点
  - 环视观测
  - 目标未检测到
  - 或当前视锥覆盖候选 bbox 但未看到目标
- 节点级更新：
  - `exist_prob *= gamma_neg`
  - `consecutive_miss_count += 1`
- 概率场更新：
  - 注入 negative Gaussian
  - clamp >= 0
  - renormalize
- 共现负向统计：
  - `negative_count += 1`
  - 降低 `cooccur_reliability`

### 7.3 验收

- diagnostics 记录每次 miss：
  - candidate point
  - affected obj_id
  - old/new `exist_prob`
  - negative kernel params
- 连续 miss 后下一轮 Top-K 不再优先访问同一区域。

---

## 8. 阶段 F：评测与消融

### 8.1 标准输出

继续输出 benchmark 标准文件：

```text
run.jsonl
summary.json
episode_results.jsonl
by_task_type.json
exploration_curve.json
```

额外输出 RAANav 文件：

```text
memory_diagnostics.jsonl
memory_summary.json
memory_by_task_type.json
memory_exploration_curve.json
candidate_traces.jsonl
failure_feedback.jsonl
```

### 8.2 消融配置

需要支持：

- full：完整 Agentic-RAG
- no_agent：关闭 Agent 通道
- no_rag：只使用 Agent prior
- no_time_decay：关闭时间衰减
- no_cooccur：关闭共现关系
- no_image_clip：image goal 退化 label 查询
- no_negative_gmm：关闭负向反馈
- no_history：只用当前探索，不用历史 layout

### 8.3 验收

- 每个配置可跑 smoke test。
- 至少对 `longterm_v1` val 输出完整表格。
- 能按 `seen_layout_count_before` 画长期收益曲线。

---

## 9. 推荐实施顺序

1. Habitat-Sim runner 接通单 episode。
2. Top-K candidate -> navigable point。
3. CLIP image goal index。
4. 正式 `S_rag`。
5. Agent 通道结构化输出。
6. 动态融合。
7. 负向反馈。
8. 全量评测和消融。

---

## 10. 成功判据

完整闭环完成时，应满足：

- `python -m benchmark.runner` 可加载 RAANav agent 跑 episode。
- RAANav agent 在 Habitat-Sim 中真实移动并输出 trajectory。
- 三类任务均可运行。
- `image_goal` 使用图像 embedding。
- 查询候选具备完整评分分解。
- 失败后会更新概率场和 memory。
- `benchmark.evaluate` 可生成标准结果。
- RAANav diagnostics 可解释每个候选点来源和每次失败反馈。

---

## 11. 执行状态更新（2026-06-10）

本计划已完成一版可运行闭环实现，新增模块：

- `image_goal_index.py`
- `formal_scoring.py`
- `run_closed_loop.py`

已完成：

- 从 benchmark episode 启动 RAANav closed-loop adapter。
- 根据 `seen_layout_count_before` 构建历史长期记忆。
- 对 `open_vocab`、`image_goal`、`language_goal` 统一生成查询。
- `image_goal` 已使用 image reference stem / object image stem 的可运行检索 fallback，并保留 CLIP 后端接口位置。
- 已实现正式评分字段：
  - `R_sim`
  - `R_cfd`
  - `G_ts`
  - `S_rag`
  - `S_agent`
  - `w_agent`
  - `w_rag`
  - `S_final`
- 已实现动态融合权重。
- 已实现候选访问后的负向反馈：
  - `exist_prob` 下降
  - `consecutive_miss_count` 增加
  - `failure_feedback.jsonl` 记录
- 已输出：
  - `run.jsonl`
  - `summary.json`
  - `episode_results.jsonl`
  - `by_task_type.json`
  - `exploration_curve.json`
  - `memory_diagnostics.jsonl`
  - `candidate_traces.jsonl`
  - `failure_feedback.jsonl`

当前环境限制：

- 当前 Python 环境未安装 `habitat_sim`，因此本次验证走 Euclidean fallback。
- Habitat-Sim 接入口已实现，安装 Habitat-Sim 后可通过 `--no-euclidean-fallback` 强制验证真实 Habitat 路径。
- CLIP 图像检索当前为 dependency-light fallback；后续可在 `ImageGoalIndex` 内接入真实 CLIP image embedding。

全量验证：

- 数据集：`longterm_v1` val
- episodes：150
- subtasks：1136
- missing episodes：0
- failure feedback events：2778

