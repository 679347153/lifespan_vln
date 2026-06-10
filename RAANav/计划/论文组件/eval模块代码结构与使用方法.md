# eval/ 评估模块 —— 代码结构与使用方法

> 最后更新: 2026-04-12

---

## 目录结构

```
AgenticRAG/eval/
├── change_script.py          # P1: 变更脚本解析 (YAML → ChangeScript)
├── epoch_simulator.py        # P2: 多 epoch merge 模拟器
├── metrics.py                # P3: 全指标计算 (MRA/GHR/D-SR/AS/SSS/MRR/SR)
├── gmm_evaluator.py          # P4: CLIP + GMM + 概率场封装
├── nav_sim_lightweight.py    # P5: 轻量导航 (SSS + 负向高斯场模拟)
├── ablation.py               # P6: 消融配置注册表
├── run_experiment.py         # P7: 端到端实验运行器 + CLI
├── test_static_pipeline.py   # P0b: 静态基线验证
├── scenarios/
│   └── 00814_changes.yaml    # 场景 00814 的 5-epoch 变更脚本
└── __init__.py
```

## 依赖关系

```
                 change_script.py ─────┐
                                       ▼
config/map.yaml ──► epoch_simulator.py ──► floors_history (各 epoch)
                         │                       │
config/stability_priors.yaml                     ▼
                                       gmm_evaluator.py
                                       (CLIP搜索 + GMM评分 + 概率场)
                                              │
                                              ▼
                                    nav_sim_lightweight.py
                                    (SSS 候选点逐步访问)
                                              │
                                              ▼
                                         metrics.py
                                    (计算 MRA/GHR/D-SR/SSS…)
                                              │
                   ablation.py ───────────────┘
                   (配置覆盖)                  │
                                              ▼
                                     run_experiment.py
                                     (端到端串联 + 消融对比)
```

## 各模块一句话职责

| 文件 | 做什么 | 输入 | 输出 |
|------|--------|------|------|
| `change_script.py` | 解析 YAML，描述每个物体在各 epoch 的位置/状态 | `scenarios/*.yaml` | `ChangeScript` 对象 |
| `epoch_simulator.py` | 按变更脚本生成每 epoch 的 `floors_now`，调用 `run_merge` 得到 `floors_history` | `ChangeScript` + T0 地图 JSON | 逐 epoch 的 `floors_history` + GT 布局 |
| `metrics.py` | 纯函数，接收预测/GT 坐标，算数值 | 预测峰值 + GT 位置 | float / bool |
| `gmm_evaluator.py` | 对 `floors_history` 做 CLIP 文本匹配 + GMM 评分 → 概率场 → Top-K 峰值 | 查询标签 + floors | peaks + scores |
| `nav_sim_lightweight.py` | 模拟导航：依次访问概率峰值，miss 时注入负向高斯，直到命中或耗尽 | gmm_evaluator + GT | SSS + found/mra |
| `ablation.py` | 定义 6 种消融条件（Full / w/o NegGMM / w/o Decay 等），覆盖 config | 消融名称 | 修改后的 config dict |
| `run_experiment.py` | 全流程胶水：加载 → epoch 模拟 → SSS 搜索 → 汇总指标 → 对比表格 | 路径 + 消融 | 指标 dict + 打印 |

## 关键外部依赖

eval/ 调用了项目其他模块：

| 被调用模块 | 位置 | eval 中谁调用 |
|-----------|------|--------------|
| `Floor` / `Room` / `Object` | `semantic_map/` | epoch_simulator, gmm_evaluator |
| `run_merge()` | `semantic_map_Update/map_Update.py` | epoch_simulator, run_experiment |
| `Calculate_obj_Score()` | `GMM_map_Create/GMM_map_calcualte.py` | gmm_evaluator |
| `build_probability_field()` | `scripts/query_e2e.py` | gmm_evaluator, nav_sim_lightweight |
| CLIP (ViT-B/32) | HuggingFace `transformers` | gmm_evaluator (text-to-text cos sim) |

---

## 使用方法

### 方法 A：一键运行全部消融 (推荐)

```bash
cd /home/adminer/agentRAG/AgenticRAG

conda run -n agentrag python eval/run_experiment.py \
    --base-map  RAG_Graph/task_output/00814_chair/map_final.json \
    --grid-dir  RAG_Graph/task_output/00814_chair/occupancy \
    --change-script eval/scenarios/00814_changes.yaml
```

默认运行全部 6 种消融条件，输出对比表格。

加 `--ablation Full` 只跑单条件；加 `--output results.json` 保存结果。

### 方法 B：Python API 粒度控制

```python
import os, sys
os.chdir("/home/adminer/agentRAG/AgenticRAG")
sys.path.insert(0, ".")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ── 单条件实验 ──
from eval.run_experiment import run_experiment
result = run_experiment(
    base_map_path="RAG_Graph/task_output/00814_chair/map_final.json",
    change_script_path="eval/scenarios/00814_changes.yaml",
    grid_dir="RAG_Graph/task_output/00814_chair/occupancy",
    ablation="Full",          # 或 "w/o NegGMM", "w/o Decay" 等
    verbose=True,
)
print(result["aggregate"])    # {MRA, GHR@3, D-SR, SSS_mean, SR, MRR, ...}

# ── 多条件对比 ──
from eval.run_experiment import run_ablation_comparison
results = run_ablation_comparison(
    base_map_path="RAG_Graph/task_output/00814_chair/map_final.json",
    change_script_path="eval/scenarios/00814_changes.yaml",
    grid_dir="RAG_Graph/task_output/00814_chair/occupancy",
    ablations=["Full", "w/o NegGMM", "w/o Decay"],
)
# 自动打印对比表格
```

### 方法 C：逐步手动调用（调试/理解用）

```python
# 1. 解析变更脚本
from eval.change_script import ChangeScript
cs = ChangeScript.from_yaml("eval/scenarios/00814_changes.yaml")

# 2. 模拟 5 epoch merge
from eval.epoch_simulator import EpochSimulator
import yaml
with open("config/map.yaml") as f:
    cfg = yaml.safe_load(f)
sim = EpochSimulator("RAG_Graph/task_output/00814_chair/map_final.json", cs, cfg)
result = sim.run_all_epochs()
floors_history = result["floors_history"]
gt_layouts     = result["gt_layouts"]

# 3. 对某 epoch 的某标签做 GMM 查询
from eval.gmm_evaluator import GMMEvaluator
evaluator = GMMEvaluator(grid_dir="RAG_Graph/task_output/00814_chair/occupancy")
qr = evaluator.query("chair", floors_history)
print(qr["peaks"])  # Top-5 概率峰值 [{world_x, world_z, prob}, ...]

# 4. 轻量 SSS 搜索
from eval.nav_sim_lightweight import LightweightNavSim
nav = LightweightNavSim(evaluator, neg_field_enabled=True)
sr = nav.search("chair", floors_history, gt_positions=[[1.0, 2.0]])
print(sr["sss"], sr["found"])

# 5. 计算指标
from eval.metrics import compute_mra, compute_ghr_at_k
hit = compute_mra(qr["peaks"][0], [1.0, 2.0], r_hit=2.0)
```

### 添加新场景

1. 将该场景的 T0 地图 JSON 和占据栅格准备好
2. 在 `eval/scenarios/` 下新建 `{scene_id}_changes.yaml`，格式照抄 `00814_changes.yaml`
3. 运行 `run_experiment.py --base-map <新地图> --change-script <新yaml> --grid-dir <新grid>`

### 添加新消融条件

在 `eval/ablation.py` 的 `ABLATION_CONFIGS` 字典中新增一项即可，`run_experiment` 会自动识别。

---

## 当前已验证数据

| 消融条件 | MRA | GHR@3 | D-SR | SSS_mean | SR |
|----------|-----|-------|------|----------|-----|
| Full | 0.917 | 0.950 | 1.000 | 1.035 | 0.950 |
| w/o NegGMM | 0.917 | 0.933 | 1.000 | 1.088 | 0.950 |

场景: 00814-p53SfW6mjZe, 5 epochs, 60 queries/run
