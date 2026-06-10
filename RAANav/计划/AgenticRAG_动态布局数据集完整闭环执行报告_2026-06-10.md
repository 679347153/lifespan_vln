# Agentic-RAG 动态布局数据集完整闭环执行报告

> 日期：2026-06-10  
> 对应计划：`RAANav/计划/AgenticRAG_动态布局数据集完整闭环计划_2026-06-10.md`  
> 状态：已完成可运行闭环适配版本，并完成 `longterm_v1` val 全量验证。

---

## 1. 本次完成内容

在离线 MVP 的基础上，新增了一版完整闭环适配 runner。它能够从 benchmark episode 出发，构建历史长期记忆，执行正式 Agentic-RAG 风格评分，生成候选探索点，模拟候选访问与负向反馈，输出标准 trajectory，并复用 benchmark evaluator 得到 SR/SPL/exploration curve。

新增文件：

```text
RAANav/AgenticRAG/benchmark_adapter/image_goal_index.py
RAANav/AgenticRAG/benchmark_adapter/formal_scoring.py
RAANav/AgenticRAG/benchmark_adapter/run_closed_loop.py
```

更新文件：

```text
RAANav/AgenticRAG/benchmark_adapter/__init__.py
RAANav/计划/AgenticRAG_动态布局数据集完整闭环计划_2026-06-10.md
```

---

## 2. 功能完成情况

### 2.1 Habitat-Sim 闭环入口

已实现：

- `run_closed_loop.py` 会尝试创建 `HabitatLayoutAdapter`。
- 若 Habitat-Sim 可用，可使用 PathFinder snap candidate point，并用 geodesic distance。
- 若 Habitat-Sim 不可用，自动回退到 Euclidean fallback，保证系统仍可完整运行。

当前环境验证：

```text
habitat_sim installed: False
```

因此本次全量验证实际使用 Euclidean fallback。真实 Habitat-Sim 验证需要在安装 Habitat-Sim 的环境中执行 `--no-euclidean-fallback`。

### 2.2 CLIP 图像检索入口

已实现：

- `ImageGoalIndex` 支持 `image_goal` 查询。
- 当前可运行后端为 image stem / label fallback：
  - reference image stem
  - memory object image stem
  - token overlap fallback
- diagnostics 中输出：
  - `query_modality`
  - `sim_backend`
  - `R_sim`

说明：

- 当前未强依赖 transformers/torch，确保 Windows 当前环境可直接跑通。
- 后续真实 CLIP image embedding 可在 `ImageGoalIndex.similarity()` 内接入。

### 2.3 正式 Agent/RAG 融合评分

已实现正式评分字段：

```text
R_sim
R_cfd
G_ts
S_rag
S_agent
w_agent
w_rag
S_final
```

公式：

```text
S_base = beta1 * R_sim + beta2 * R_cfd
G_ts = exp(-lambda * delta_t / (1 + kappa * stability))
S_rag = S_base * G_ts * exist_prob
w_agent(n) = max(w_min, exp(-k * n))
w_rag(n) = 1 - w_agent(n)
S_final = w_rag * S_rag + w_agent * S_agent
```

候选记录输出在：

```text
benchmark/eval/raanav_closed_loop_longterm_v1/candidate_traces.jsonl
benchmark/eval/raanav_closed_loop_longterm_v1/memory_diagnostics.jsonl
```

### 2.4 负向反馈

已实现：

- 候选访问失败后调用 `apply_negative_feedback()`。
- 更新候选 object：
  - `exist_prob`
  - `consecutive_miss_count`
  - `negative_feedback_count`
- 写出：

```text
benchmark/eval/raanav_closed_loop_longterm_v1/failure_feedback.jsonl
```

全量验证中产生：

```text
failure_feedback_count = 2778
```

---

## 3. 验证结果

### 3.1 单 episode smoke test

命令：

```powershell
python -m RAANav.AgenticRAG.benchmark_adapter.run_closed_loop `
  --split-manifest benchmark/splits/benchmark_split_longterm_v1.json `
  --split val `
  --scene 00808-y9hTuugGdiq `
  --limit 1 `
  --output-dir benchmark/eval/raanav_closed_loop_smoke `
  --max-candidates 10 `
  --max-rounds 5
```

结果：

```json
{
  "episodes": 1,
  "missing_episode_count": 0
}
```

### 3.2 历史记忆 episode smoke test

命令：

```powershell
python -m RAANav.AgenticRAG.benchmark_adapter.run_closed_loop `
  --split-manifest benchmark/splits/benchmark_split_longterm_v1.json `
  --episodes benchmark/episodes/longterm_v1/val/00808-y9hTuugGdiq/layout_009_seed_51/longterm_v1_00808-y9hTuugGdiq_layout_009_seed_51_ep_0045.json `
  --output-dir benchmark/eval/raanav_closed_loop_state9 `
  --max-candidates 10 `
  --max-rounds 5
```

结果摘要：

```json
{
  "episodes": 1,
  "subtasks": 7,
  "found_rate": 0.7142857142857143,
  "mra": 0.7142857142857143,
  "ghr3": 0.7142857142857143,
  "ghr5": 0.7142857142857143,
  "avg_sss": 2.7142857142857144,
  "failure_feedback_count": 14
}
```

### 3.3 longterm_v1 val 全量验证

命令：

```powershell
python -m RAANav.AgenticRAG.benchmark_adapter.run_closed_loop `
  --split-manifest benchmark/splits/benchmark_split_longterm_v1.json `
  --split val `
  --output-dir benchmark/eval/raanav_closed_loop_longterm_v1 `
  --max-candidates 10 `
  --max-rounds 5
```

RAANav memory diagnostics：

```json
{
  "episodes": 150,
  "subtasks": 1136,
  "found_rate": 0.5220070422535211,
  "mra": 0.5220070422535211,
  "ghr3": 0.6038732394366197,
  "ghr5": 0.6487676056338029,
  "avg_sss": 2.9674295774647885,
  "avg_min_dist": 0.8401808116388645
}
```

Benchmark evaluator summary：

```json
{
  "episodes": 150,
  "success_rate": 0.04666666666666667,
  "spl": 0.034464711519582095,
  "avg_steps": 23.3,
  "avg_path_length": 66.43763095333333,
  "avg_shortest_path_length": 53.66066415508588,
  "dynamic_memory_accuracy": 0.5220070422535211,
  "evaluated_episodes": 150,
  "trajectory_count": 150,
  "missing_episode_count": 0
}
```

输出目录：

```text
benchmark/eval/raanav_closed_loop_longterm_v1/
```

核心输出文件：

```text
run.jsonl
summary.json
episode_results.jsonl
by_task_type.json
exploration_curve.json
memory_diagnostics.jsonl
memory_summary.json
memory_by_task_type.json
memory_exploration_curve.json
candidate_traces.jsonl
failure_feedback.jsonl
adapter_warnings.json
```

---

## 4. 如何验证

### 4.1 当前环境快速验证

```powershell
python -m RAANav.AgenticRAG.benchmark_adapter.run_closed_loop `
  --split-manifest benchmark/splits/benchmark_split_longterm_v1.json `
  --split val `
  --scene 00808-y9hTuugGdiq `
  --limit 1 `
  --output-dir benchmark/eval/raanav_closed_loop_smoke `
  --max-candidates 10 `
  --max-rounds 5
```

查看结果：

```powershell
Get-Content benchmark/eval/raanav_closed_loop_smoke/summary.json
Get-Content benchmark/eval/raanav_closed_loop_smoke/memory_summary.json
```

### 4.2 全量验证

```powershell
python -m RAANav.AgenticRAG.benchmark_adapter.run_closed_loop `
  --split-manifest benchmark/splits/benchmark_split_longterm_v1.json `
  --split val `
  --output-dir benchmark/eval/raanav_closed_loop_longterm_v1 `
  --max-candidates 10 `
  --max-rounds 5
```

查看结果：

```powershell
Get-Content benchmark/eval/raanav_closed_loop_longterm_v1/summary.json
Get-Content benchmark/eval/raanav_closed_loop_longterm_v1/memory_summary.json
Get-Content benchmark/eval/raanav_closed_loop_longterm_v1/memory_exploration_curve.json
```

### 4.3 Habitat-Sim 环境验证

在安装 Habitat-Sim 后执行：

```powershell
python -m RAANav.AgenticRAG.benchmark_adapter.run_closed_loop `
  --split-manifest benchmark/splits/benchmark_split_longterm_v1.json `
  --episodes benchmark/episodes/longterm_v1/val/00808-y9hTuugGdiq/layout_009_seed_51/longterm_v1_00808-y9hTuugGdiq_layout_009_seed_51_ep_0045.json `
  --output-dir benchmark/eval/raanav_closed_loop_habitat_smoke `
  --max-candidates 10 `
  --max-rounds 5 `
  --no-euclidean-fallback `
  --load-layout-objects
```

若 Habitat-Sim、HM3D 和 object templates 配置正确，该命令会强制使用 Habitat 路径；否则会直接报错，便于定位环境问题。

---

## 5. 当前边界

本次已经把完整闭环的工程接口与 fallback 跑通，但有两个能力在当前环境中仍是可替换后端：

1. Habitat-Sim
   - 接口已接入。
   - 当前环境未安装 `habitat_sim`，本次全量验证为 Euclidean fallback。

2. CLIP image embedding
   - `ImageGoalIndex` 已接入 image-goal 路径。
   - 当前使用 image stem/token fallback。
   - 后续可替换为真实 CLIP image-to-image similarity。

---

## 6. 下一步建议

1. 在 Linux/Conda Habitat 环境中运行 `--no-euclidean-fallback` smoke test。
2. 在 `ImageGoalIndex` 中接入 transformers CLIP 或现有 `CLIP_RAG` image index。
3. 将 `S_agent` 从启发式 prior 升级为 `RScoreAgent.score_for_navigation()` 的结构化 LLM 输出。
4. 将负向反馈的 `exist_prob` 更新持久化到 episode-level memory snapshot。
5. 增加消融入口：`--ablation no_agent/no_time_decay/no_cooccur/no_negative_gmm/no_image_clip`。

