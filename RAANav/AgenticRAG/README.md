# Brief, 已实现的功能：
实体测试需要的功能：
- [ ] 给building的floor和room唯一ID

- [ ] 增量更新room或者floor的时候需要提供唯一ID作为标识，我们在实际扫描的过程中会通过机器人的定位标记生成这些ID

- [ ] 目前测试用的obj的CLIP向量是直接通过文本转化的，实际上我们预期
 完全模仿HOVSG通过 全景、局部、局部mask 三张机器人的RGB-D传感器获取的图像来生成CLIP向量，这样可以更好地反映实际环境中的物体特征。


- 参数运行保存和合并文件，灵活读取配置文件修改参数
- 局部更新Floor、房间产生新的地图，简单通过房间面积判断确认
```python
PYTHONPATH=. python semantic_map_Update/map_Update.py RAG_Graph/map_now.json RAG_Graph/test_map/map_historyR2.json RAG_Graph/test_save/map_merged_R2only.json
```
- 并排对比可视化

# TODO
## 检查-Recheck
- [ ] 我的Rscore和GMM_score的最大值是否是通过Score_max平均的，先在的Rscore是将原始分数==截断==到 [0,1] 后按 RscoreValue_max 线性缩放，最终呈现到Rscore

- [ ] GMM_map_Create/GMM_map_calcualte.py的Calculate_Robj_Score计算逻辑：这个物体旁边没有过obj，那么她的相关分数就是0
同理，如果没有agent的打分，那么Rscore就是0

- [ ] 检查GMM_map_Create/GMM_map_calcualte.py的_clip_search_targets计算的clip_search(vecs, ids, qv, top_k=None, min_score=thr)计算出来的余弦相似读可以大于1？

- [ ] 计算得出的Rscore由于物体可能与两个相关目标都有关系，所以会出现一个物体有两个Rscore的情况，那么这样通过related_obj_by_target可以方便的合并

- [ ] Cfd怎么计算？CLIP计算AI生成的文本和原本获取的图像向量

- [ ] 现在是从CLIP模型里面筛选出来的与我们要找的target最语义相似的物体，模仿VDB里面直接拉取（后续可以通过直接喂给AI找出来与装花的瓶子最相关的，肯定比CLIP语义相似更加好），也可以照样给AI我们的obj_list，这样他给的结果就一定是我们obj_list的东西了

- [ ] N_over_N、Rcfd:计算两边的平均相关度以及平均Nr/N

- [ ] 相关物体的Nr和Rcfd如果是同一个相关的物体，注意计算不要覆写而是相加

- [ ] Calculate_obj_Score
    if cfd is None:
        base = 0.5 * stability + 0.5 * sigmoid_N
    else:
        cfd_val = max(0.0, min(1.0, float(cfd)))
        base = 0.4 * stability + 0.4 * sigmoid_N + 0.2 * cfd_val
    return round(base * Score_max, 4)

- [ ] 用CLIP判断是不是要找的同一个物体不如输入objlist给Agent让他揪出来

- [ ] 物体的稳定性现在是<1其实是扣分项

- [ ] 没有用上物体的更新次数加权

- [ ] 对于要查找的物体本身，stability太小削弱他的概率图

- [ ] 一个伞架子，本来的stability就低，还位移了导致变成了新的物体导致他的概率权重（于更新次数N有关）变得非常小？或许stability低的物体判断 ==是否为同一个物体进行合并更新== 后面可以考虑把位移变成一个区域，也是先膨胀，stability低的物体膨胀因子相对于大一点，接着在看有多少落在里面进行合并更新

- [ ] 像HOVSG一样把物体的文本嵌入作出许多情景的模板减小其他影响

- [ ] 语义地图的合并和更新的BUG，并且检查自己的更新逻辑
之前有过ID没有看出来错误导致重复、region-judge的IOU问题

- [ ] 检查CLIP的实时和json建立索引，调用实时运行构建的函数测试

### 搜索 obj
- [ ] 开发agent输出最可能是的物体的objlist，以及相应的最可能相关的物体的objlist
- [ ] 检查名字解析问题比如“umbrella_stand_1_R3”因为这个label是两个单词不能向前判断
- [ ] 利用CLIP实现文本检索
1. 将物体的文本或者图片转化为向量存储，并且带有物体的ID索引
存储，单单存向量 & 存向量+对应的物体ID
objects.jsonl：JSON Lines 格式（又叫 jsonl / NDJSON）。文件内每一行是一个独立 JSON 对象，方便流式读取与追加，不需要一次载入整个文件就能逐行处理，适合追加或增量更新。字段现为 { obj_id, floor_id, room_id, description }，其中 description 优先使用对象 description.brief，其次最后一次扫描描述(最大数字键)，再回退任意非空描述或 label。
embeddings.npy 是一个==二维数组==(N, D)，N = 对象条目数（包括重复），D = 向量维度（比如 512）。因为构建阶段严格按 ids 列表顺序生成/拼接，所以顺序一致是有保障的。

- [ ] 引入Qwen-API

- [ ] 物体向量的实时存储：在未保存为地图的时候，与转化为clip向量，后面存储到json地图里面

- [ ] 引入CLIP，语义向量RAG搜索模式
> 比如搜索一个杯子，蓝色的杯子也可以，想找到语义为杯子的，接着拉取Robj

- [ ] 语义打分

- [ ] 构建搜索物体的 GMM 概率图像
> 如何计算Robj和obj自己的权重？Stability！？如果Stability低就狠狠的降低权重
不过其他的还有吗

- [ ] Make obj region and label, clip_embedding, restore corresponding imgs
> 向量: RGB-D -> CLIP: 
> 语义标签_ID & description & ?cfd: RGB-D -> VLM /不用YOLO这样的有限检测
> 占据区域：odometry 点云？还是RGB-D深度信息？
关键点在于他设置了这个物体的语义区分，那么怎么保证这region的区域一定是这个区域了
我其实只需要画出一个矩形或者三角形的大致粗糙区域就够后面使用了

- [ ] 建图的时候如何区分 room 给obj打上room_id

### GMM图像生成

### 数据集的转化 & 实际建图
- [ ] 想办法搞到可以手动移动并且生成可以生成统一个楼层新的json文件的RAG_Graph
> 解析HM3D-Sem的文件到pgm和手动标注 + 可视化移动？生成新的数据集？
> 学习osmAG是如何利用HM3D数据集生成他们想要的oms地图的

osmAG-LLM/HM3DSEM-mapping/get_area_description.py

## HACK-DONE
### 地图的合并和更新
- [x] 地图的合并和更新
> 情景：有两个地图，一个是历史的一个是新的，有物体移动，增加，修改（初期先保证这些增加都是合理的，并且关系图谱也是合理的）

> 可以先考虑obj再room的更新，obj都做出来了room也差不多
- [x] 关系图谱的更新
- [x] 支持单个房间更新。房间自带楼层号码所以自己次数也会更新 暂时未考虑如果房间扫描不全的情况
- [ ] 接着我需要给地图更新添加只更新房间内部obj，不更新房间的功能，测试文件如下：RAG_Graph/test_map/map_objOnly.json
- [x] 可视化

## GMM-Score返回

返回结构与规则已与实现对齐：

- GMM_obj_list(target)
	- target: 查询标签
	- Robj_scores: {label: score}（从 RScoreAgent 输出读取，若配置为 NONE 则为空）
	- targets_from_clip: [{obj_id, score}]（使用 CLIP 阈值 consine_similarity_filmin 选出所有实例，非只取 Top-1）
	- by_target: {target_id: [related_record, ...]}
	- related: 扁平列表，每条含字段：
		- {target_id, label, obj_id, Rscore, Rcfd, Nr_over_N, stability, N}

- build_GMM_feature_set(target)
	- 在 related 上新增 GMM_score
	- 新增 targets_self: [{obj_id, GMM_self_score}]

关键计算与约束：
- 相关集合不过滤分数高低，仅过滤“地图中不存在的标签”，且去除与 target 相同标签。
- Nr_over_N、Rcfd 来自语义图 target_obj.R_objs；若缺失则按 0 处理，Nr_over_N = Nr / max(1, N_target)。
- Calculate_obj_Score(N, stability, cfd):
	- 若 cfd is None: base = 0.5*stability + 0.5*sigmoid_N
	- 否则: base = 0.4*stability + 0.4*sigmoid_N + 0.2*cfd
	- sigmoid_N = 1/(1+exp(-k*(N-N0)))，最终乘以 GMM.Score_max
- Calculate_Robj_Score(total_N, Nr_over_N, Rscore, Rcfd, stability):
	- 动态权重由 GMM.weight_mode 决定：
		- Sigmoid: ω2 = 1/(1+exp(-k*(N-N0)))，ω1 = 1-ω2
		- Exponential: ω2 = clamp(a*exp(b*N),0,1)，ω1 = 1-ω2
	- relation_enhance = exp(lambda * Nr_over_N + Rcfd)
	- rel_norm = log(1+relation_enhance) / log(1+exp(lambda*1 + 1))（压缩映射到 0~1）
	- fused = ω1*Rscore + ω2*rel_norm；score = fused * stability，再乘以 GMM.Score_max

配置键位（config/map.yaml）：
- map_config.CLIP_RAG_map_dir：读取 CLIP 索引（ids/embeddings/meta）
- CLIP_RAG.consine_similarity_filmin：阈值（用于目标实例选择）
- GMM.Score_max、GMM.Nr_lambda_param、GMM.weight_mode、GMM.weight_exponential.{a,b}、GMM.weight_sigmoid.{k,N0}
- agent_Rscore.output_path：读取 RScoreAgent 结果以获得 Robj_scores；支持 {target} 模板；设为 NONE/空则不读取

最小测试脚本：
- 运行：`python scripts/min_gmm_test.py --target umbrella_stand`（可选 `--top 10`）
- 输出：
	- Targets from CLIP（阈值命中）
	- Target self scores（GMM_self_score）
	- Related items by GMM_score（Top-K）
	- Agent label Robj_scores（Top-K）
	- 完整结果保存到 `scripts/_last_min_gmm_result.json`


## 🔍 CLIP 文本检索 (初版)

### 参数与文件结构说明
本模块包含两个脚本：`CLIP_RAG/build_clip_index.py` 与 `CLIP_RAG/query_clip_index.py`。

#### 1. 索引构建脚本 `build_clip_index.py`
核心功能：遍历语义地图中所有对象，抽取对象描述文本编码为 CLIP 向量并持久化为本地检索索引。

命令行参数：
- `--map <path>` 必填。语义地图 JSON 路径（可含多楼层）。
- `--root-dir <dir>` 索引根目录；内部为多个索引子目录。默认 `CLIP_RAG`。
- `--index-name <name>` 手动指定索引子目录名称；缺省则自动生成：`<mapStem>__<model>__<timestamp>__<hash>`。
- `--model <huggingface-id>` CLIP 模型标识，默认 `openai/clip-vit-base-patch32`。
- `--field {label|desc}` 文本来源策略 (默认 label；命令行参数优先于配置文件 `CLIP_RAG.text_embedding.field_mode`):
	- `label`: 仅用 label/name。
	- `desc`: 使用描述优先级（brief → 最新扫描(最大数字键) → 任意非空描述 → 回退 label）。
- `--device {cuda|cpu}` 指定推理设备；默认优先 GPU。
- `--batch-size <int>` 文本批量编码大小，默认 32。
- `--force` 若目标索引目录已存在且非空则覆盖（否则报错保护）。

描述抽取优先级 (写入 objects.jsonl 的 `description` 字段)：
1) `description.brief`
2) `description` 中最大数字键对应值（代表“最后一次扫描”）
3) 其它任意首个非空字符串值
4) 回退 `label`/`name`

输出文件结构（位于 `<root-dir>/<index-name>/`）：
- `ids.json`            对象 ID 顺序列表 (用于与向量矩阵行索引对齐)
- `embeddings.npy`      Float32 矩阵 `[N, D]` (已单位化) 
- `objects.jsonl`       每行一个对象元数据：`{obj_id, floor_id, room_id, description}`
- `meta.json`           元信息：模型、向量维度、对象数量、字段策略、源地图路径、索引名称等

#### 2. 查询脚本 `query_clip_index.py`
核心功能：对已构建索引执行文本相似度检索，返回按余弦相似度降序的对象列表。
== 因为会自动读取配置，所以如果命令行只输入--top-k 5还是会通过--min-score筛选，所以除非自己设置min-score=0.0 ==

命令行参数：
- `--root_dir <dir>` 索引根目录；也可直接指向具体索引目录（含 meta.json 时自动识别）。
- `--index-name <name>` 指定使用的索引子目录；不提供则自动选取最近修改时间最新的索引。
- `--query <text>` 必填。查询文本。
- `--top-k <int>` 仅返回前 K 个结果；缺省返回全部（受阈值过滤影响）。
- `--min-score <float>` 相似度阈值（0~1），过滤低于该值的结果。
- `--config <path>` 配置文件 (默认 `config/map.yaml`)；当未显式提供 `--min-score` 且未禁用阈值时，会尝试读取 `CLIP_RAG.consine_similarity_filmin` 作为默认阈值。
- `--no-threshold` 忽略配置文件默认阈值（等价于只按 top-k / 全量返回）。
- `--device {cuda|cpu}` 指定推理设备。

检索流程：加载向量 → 编码查询文本 → 计算余弦相似度 → (可选) 应用阈值过滤 → (可选) 截取 top-k。

输出：标准输出按行打印 JSON，每行字段：
`{rank, obj_id, consine_similarity, description, floor_id, room_id}`

说明：`consine_similarity` 为归一化文本向量点积结果；字段名称沿用配置键拼写以便匹配 (`consine_similarity_filmin`)。

兼容性：旧版本若 `objects.jsonl` 中字段仍为 `text`，查询脚本会自动回退读取为 `description`。无需重建即可升级查询脚本。

错误与诊断：
- 当结果为空且设置了阈值，会额外显示最高相似度与建议（降低阈值 / 使用 `--no-threshold` 等）。
- 索引名称不存在时列出可用候选并提供相似名称建议。

性能提示：如需批量频繁查询，可将 `query_clip_index.py` 中模型加载部分改为长驻服务（FastAPI / Gradio 等）以避免重复加载权重。

---
### 索引构建
```bash
PYTHONPATH=. python CLIP_RAG/build_clip_index.py \
	--map RAG_Graph/test_save/map_mergedALL.json \
	--field label \
	--root-dir CLIP_RAG  # (默认根目录) \
	--index-name demo_idx  # 可选: 指定名称, 否则自动 mapName__model__时间戳__hash
```
### 查询示例
```bash
PYTHONPATH=. python CLIP_RAG/query_clip_index.py --root_dir CLIP_RAG --index-name demo_idx --query "一个绿色的植物" --top-k 3
```
输出按 JSON 行列出 rank / obj_id / consine_similarity / description。

参数优先级: 命令行 > config/map.yaml 中 CLIP_RAG.text_embedding > 内置默认(label / batch_size=32 / device=cuda 可用否则 cpu)。

支持阈值过滤 (map.yaml -> CLIP_RAG.consine_similarity_filmin 或命令行 --min-score)：
```bash
PYTHONPATH=. python CLIP_RAG/query_clip_index.py --root_dir CLIP_RAG --index-name demo_idx --query "trash bin" --min-score 0.85
```
若同时提供 --top-k 与 --min-score：先按阈值过滤，再截取 top-k。

## 🧮 RScore 相关对象打分 Agent (Qwen)
为一个查询目标物体 (如要找“cup” / “umbrella”) 生成其最相关的环境锚点对象列表及评分，用于后续导航/概率图 (GMM) 构建。支持两种模式：
1. 完整地图模式: 读取 `agent_Rscore.map_full_path` 获取稳定度/次数/关系等上下文。
2. 轻量 ID 模式: `map_full_path: "NONE"` 仅根据 `ids.json` 与常识推理，缺失属性取默认估计。

### 组件说明
RScoreAgent 会读取:
1. 语义地图: `RAG_Graph/test_save/map_mergedALL.json`
2. CLIP 索引对象 ID 列表: `config/map.yaml -> map_config.CLIP_RAG_map_dir/ids.json`
3. 配置: `config/map.yaml -> agent_Rscore` (含 `model`, `predict_loss_rate`, `qwen_APIKEY`)

LLM(Qwen OpenAI-Compatible) 获得精简对象属性 (obj_id, label, room_id, stability, N, cfd, 关系键列表) 和房间标签统计，输出结构化 JSON。

### 评分维度 (0~1)
| 组件 | 含义 | 计算/指引 |
|------|------|-----------|
| FA | Functional Affinity 功能/使用场景亲和度 | 与目标物体共同使用 / 物理交互频繁 |
| SAS | Spatial Anchor Stability 空间锚定稳定度 | 0.5*stability + 0.5*sigmoid(N; k=1,N0=2) (缺失取0.5) |
| OR | Observed Reliability 观测可靠度 | = cfd (缺失估 0.5) |
| RP | Relation Predictability 关系可预测性 | 历史共现或常识搭配 |
| CC | Category Complementarity 类别互补性 | 支撑/容器/承放/搭配关系 |

基础公式: `Rscore_base = 0.3*FA + 0.2*SAS + 0.2*OR + 0.2*RP + 0.1*CC` (四舍五入 2 位)。

若对象是“纯预测”(没有地图历史关系，仅常识推断) => `Rscore = Rscore_base * predict_loss_rate` (keep factor)。

`predict_loss_rate` (keep factor) 在 `config/map.yaml -> agent_Rscore.predict_loss_rate` 中配置 (例如 0.8 => 保留 80% 基准分，折损 20%)。

### 配置片段
```yaml
agent_Rscore:
	qwen_APIKEY: "YOUR_API_KEY"     # 也可用命令行 --api_key 覆盖
	model: "qwen-plus"              # 模型名称
	predict_loss_rate: 0.8           # keep factor: predicted Rscore = base * 0.8
	map_full_path: "RAG_Graph/test_save/map_mergedALL.json"  # 或 "NONE" 进入轻量 ID 模式
	top_k: 100                       # 截断 (<=0 或 大值视为不限制)
	Rscore_min: 0.0                  # 最小阈值 (<=0 不限制)
	Rscore_max: 100.0                # (预留) 归一化上限
	output_path: "RAG_Graph/test_save/"   # 保存结果目录/文件; 可含 {target}; 若为目录自动生成 <target>_Rscore.json
	# 若设为 NONE (或省略/空字符串) 则不写出文件
```

### 命令行用法
```bash
PYTHONPATH=. python GMM_map_Create/agent_Rscore.py \
	--target "umbrella" \
	--config config/map.yaml \
	--api_key $QWEN_KEY

# 覆盖保存路径示例 (文件)
PYTHONPATH=. python GMM_map_Create/agent_Rscore.py \
	--target "umbrella" \
	--config config/map.yaml \
	--api_key $QWEN_KEY \
	--save out/umbrella_custom.json

# 覆盖保存路径示例 (目录 + 模板)
PYTHONPATH=. python GMM_map_Create/agent_Rscore.py \
	--target "cup" \
	--config config/map.yaml \
	--api_key $QWEN_KEY \
	--save out/rscore/{target}.json
```
可选参数:
- `--model qwen-plus` (若想临时覆盖配置)
- `--predict_loss_rate 0.6`
- `--no_auto_config` (禁用自动读取 map.yaml)

### 输出示例 (节选)
```json
{
	"target": "umbrella",
	"candidates": [
		{
			"obj_id": "shoe_cabinet_1_R3",
			"label": "shoe_cabinet",
			"predicted": false,
			"score_components": {"FA":0.82,"SAS":0.78,"OR":0.61,"RP":0.75,"CC":0.70},
			"Rscore_base": 0.75,
			"Rscore": 0.75,
			"reasons": "Umbrella stand usually near shoe cabinet at entrance; stable anchor."
		}
	],
	"predict_candidates": [
		{
			"obj_id": "mat_1_R3",
			"label": "mat",
			"predicted": true,
			"score_components": {"FA":0.70,"SAS":0.65,"OR":0.56,"RP":0.55,"CC":0.60},
			"Rscore_base": 0.63,
			"Rscore": 0.13,
			"reasons": "Entrance mat often co-located; no explicit historical relation hence penalized."
		}
	],
	"Robjlist_with_Rscores": [
	  {"label":"shoe_cabinet","Rscore":0.78,"merge_factor_used":1.0},
	  {"label":"mat","Rscore":0.65,"merge_factor_used":1.05}
	],
		"scoring_formula": "Rscore_base = 0.3*FA + 0.2*SAS + 0.2*OR + 0.2*RP + 0.1*CC; predicted => *0.80 (keep factor)",
		"predict_loss_rate_used": 0.8,
	"config_ids_path": "/abs/path/CLIP_RAG/.../ids.json"
}
```

### 字段解释
- `candidates`: 来自地图已有关系/高置信的锚点对象。
- `predict_candidates`: 常识推断但缺少历史关系的候选；分数已折减。
- `Robjlist_with_Rscores`: 基于 real+predicted 按标签聚合。若同标签出现 n 次(含 real/predicted 多来源) => 平均分 * (merge_rate)^(n-1)。`merge_factor_used` 给出该标签实际使用的倍率；单独出现则为 1.0。
- `score_components`: 五个子维度，方便调试与可解释性。
- `predicted`: 是否纯预测 (控制折减)。
- `Rscore_base` / `Rscore`: 折减前 / 后最终得分。

### 常见问题
| 问题 | 处理 |
|------|------|
| 输出不是 JSON | 调低 `temperature` 或重试；后处理检测失败会返回 `LLM_NON_JSON` |
| 预测对象得分过高 | 降低 `predict_loss_rate` (keep factor) 或提高 `Rscore_min` |
| 英文标签混入中文 | 已在 system prompt 强制英文；若仍发生可再加规则过滤 |
| 想加入 Nt/Nr/Rcfd 直接加权 | 可在 `_build_prompt` 中追加这些统计并扩展公式 (预留扩展点) |

### 下一步可扩展
- 引入 Nt/Nr/Rcfd 形成先验修正项，例如 `RP' = α*RP_llm + (1-α)*f(Nr,Rcfd)`
- 对 Rscore 进行归一化或温度缩放，服务下游概率图构建。
- 增量缓存 LLM 输出 (target, map_hash) 以减少重复调用。



## ✅ 语义地图合并/更新最新特性 (2025-08)
| 类别 | 说明 |
|------|------|
| 稳定 ID | Object ID 规则: label_seq_roomId；只为缺失 ID 的新对象分配，保持历史稳定性 |
| 多次扫描累计 | 对象/房间 N = 历史 N + 当前 N (双方都可能是聚合值) |
| 媒体累积 | imgs/description/room_name 强制 dict[str,*]；按 total_N 顺序追加键 '1'..'N'；迁移脚本已清理旧 list/str 格式 |
| 关系合并 | 仅累计 Nr；Rcfd 采用指数平滑 (epsilon_Rcfd)；缺失轮次的历史关系执行一次衰减追加 |
| 置信度平滑 | cfd / Rcfd / clip_embedding 使用统一指数公式参数化 (epsilon_*) |
| 房间形状校验 | Polygon IOU 与面积偏差(对称)双阈值；低于阈值输出警告不阻断合并 |
| 局部输入 | 允许 now 仅包含某些楼层/房间；未出现的历史楼层/房间保持原样 |
| 新增注入 | 允许新增楼层/房间 (可配置 allow_new_*)，其对象/房间缺省 N 初始化为 1 |
| 键排序健壮 | 统一媒体键为字符串；混合 int/str 时安全排序避免 TypeError |
| Schema 收紧 | 运行逻辑已移除对旧 list / str 媒体形式的兼容，仅入口模型做最小兜底 |
| 评分函数 | 提供试验性 obj_calculate_obj_Score(total_N, Nr, ...) 用于后续检索排序 |

### 合并命令行示例
```bash
PYTHONPATH=. python semantic_map_Update/map_Update.py RAG_Graph/map_now.json RAG_Graph/map_history.json RAG_Graph/map_merged.json
```

### 配置片段 (`config/map.yaml`)
```yaml
object:
	epsilon_cfd: 0.75
	epsilon_Rcfd: 0.75
	epsilon_clip: 0.75
	region_threshold: 0.8
	imgs_description_append:
		mode: sequential_totalN
update:
	room_shape:
		iou_threshold: 0.9
		area_ratio_tolerance: 0.2
```

### 关键更新说明
1. Rcfd 衰减：历史存在但本轮缺失的关系会被带入并执行一次 0 观测平滑，避免瞬间丢失。
2. clip_embedding 若维度改变，直接采用最新向量防止错误加权。
3. 房间媒体合并时使用历史 N 作为偏移，当前媒体 dict 顺序化追加，确保键连续无冲突。
4. Room 与 Object 内部兼容旧格式的地方不再向前传播 (转换后统一结构)。
5. 排序混合键策略: 数字键优先按数值升序，其次其它键按字典序，确保 deterministic。

### 待办方向 / Next Steps
- 引入 JSON Schema 校验与自动修复报告
- 为合并逻辑补充单元测试 (对象匹配 / 多扫描媒体连续性 / 关系衰减)
- 将评分函数接入检索流程 (CLIP + 文本查询融合)
- 按需延迟加载 clip_embedding (filesystem streaming)

# 机器人语义地图管理系统 (JSON 版)

简洁的 Floor → Room → Object 分层语义地图与 3D 可视化。当前主数据格式已迁移为 **JSON**，支持对象间聚合关系 `R_objs` (含出现次数与置信度)。

## ✨ 功能概览
| 模块 | 说明 |
|------|------|
| 分层结构 | Floor / Room / Object 三级，显式 `room_ids` / `obj_ids` 维护引用列表 |
| 关系建模 | `R_objs: {target_obj_id: {Nr:int, Rcfd:float}}` 去除旧多类型关系，聚合展示 |
| 数据格式 | 统一 JSON，仍兼容读取 YAML（自动按扩展判断） |
| 3D 楼层图 | 房间/对象 Mesh，所有对象名称常显，关系线按 Rcfd 灰→红、宽度加权 |
| 3D 层级图 | root→floor→room→object 自适应分布，名称常显；对象颜色 / 尺寸 = 关系度映射 |
| 转换脚本 | `convert_yaml_to_full_json.py` 可将旧 YAML 合并生成 JSON |

## 📁 目录结构
```
app.py                         # Streamlit 3D 可视化入口 (JSON 优先)
semantic_map/floor.py          # Floor 数据类
semantic_map/room.py           # Room 数据类
semantic_map/object.py         # Object 数据类 (含 R_objs)
clean_semantic_map.json        # 主语义地图数据 (可多楼层)
clean_semantic_map.yaml        # 原始 YAML（保留，转换来源）
convert_yaml_to_full_json.py   # YAML -> JSON 生成与多楼层合并
README.md
```

## 🧱 数据 Schema
### Floor
```jsonc
{
	"floor_id": "F1",
	"z_range": {"z_min": 0.0, "z_max": 4.0},
	"rooms": [ ... Room ... ],
	"room_ids": ["R1", "R2"]
}
```
### Room
```jsonc
{
	"room_id": "R1",
	"room_name": "Demo Room",
	"floor_id": "F1",
	"Region": [ {"x":2,"y":2}, ... ],
	"objects": [ ... Object ... ],
	"obj_ids": ["sofa_001", "table_002"]
}
```
### Object
```jsonc
{
	"obj_id": "sofa_001",
	"label": "sofa",
	"room_id": "R1",
	"region": [ {"x":6.4,"y":4.19}, ... ],
	"stability": 0.62,
	"R_objs": {
		"table_002": {"Nr": 1, "Rcfd": 0.83}
	},
	"imgs": []
}
```

## 🖥 运行可视化
```bash
pip install -r requirements.txt
streamlit run app.py
```
侧栏选择 `clean_semantic_map.json`（默认已指向）。

### 并排对比模式 (双文件自由选择)
在侧栏勾选 “并排对比 (history vs now)”：
1. 第一文件 = 主视图(通常为历史 / 合并结果)。
2. 输入第二个 JSON 路径（可任意文件，不限制必须是 now，也可与测试子集对比）。
3. 选择移动阈值 `移动判定阈值(m)`：用于判定对象是否属于 `moved`（同 id，region 重心距离 > 阈值）。
4. 勾选 “ghost” 后，会把只存在于第一文件的对象以半透明“ghost”显示。
5. 差异标记说明：
	- added: 只在第二文件
	- removed: 只在第一文件
	- moved: 同 id 出现但位置超阈值
	- changed: 占位（后续可扩展 label/属性变化）
	- same: 未变化

对比视图下，可同步相机（试验性），方便视觉差异检查。

## 🔍 关系与视觉编码
| 图层 | 表达 |
|------|------|
| 楼层/房间 | 半透明 Mesh3d |
| 对象 | 实心 Mesh3d + 名称标签 |
| Rcfd | 线条颜色：灰 (#969696) → 红 (#ff0000) 渐变；线宽按 Rcfd 放大 |
| 关系度 (degree) | 层级图中对象节点颜色与大小：绿色→红色，14px→22px |

## 🔄 YAML → JSON 转换
运行：
```bash
python convert_yaml_to_full_json.py
```
会读取 `clean_semantic_map.yaml`，并与现有 JSON 合并（附加为第二楼层）重写 `clean_semantic_map.json`。

## ✅ 设计约束
- Room 多边形不重叠（允许边界接触）；Object 可重叠。
- `room_ids` / `obj_ids` 与嵌套结构保持一致，否则视为数据错误。
- `R_objs` 不做方向推断：是否双向由数据供应方决定；可视化去重显示一条线。
- 地图合并支持“部分楼层/房间”输入；未提供的历史楼层/房间保持不变。
- 通过 `config/map.yaml` 中 `update.room_shape` 配置房间形状检测阈值:
	```yaml
	update:
		room_shape:
			iou_threshold: 0.55        # IOU 低于警告
			area_ratio_tolerance: 0.5  # 面积偏离 ±50% 超出警告
	```


## 🚀 后续可拓展
- JSON Schema 校验与自动修复
- 关系过滤/搜索与节点高亮
- 导出当前 3D layout / 截图 API
- 增加路径规划与空间查询