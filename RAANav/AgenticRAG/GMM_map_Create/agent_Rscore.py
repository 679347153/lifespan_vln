"""RScore Agent

功能: 基于当前已经构建的地图(`map_mergedALL.json`) 与 CLIP 过滤出的对象 id 列表(ids.json),
为一个待查找目标物体 (例如: "cup") 推理一组强相关的对象列表 Robjlist 及其评分 Rscore。

核心思路:
1. 充分利用地图中对象的结构信息: label, stability, N(更新次数/版本), cfd(视觉置信), 以及对象之间的历史关系字段 R_objs (Nt/Nr/Rcfd)。
2. 调用 LLM(Qwen/OpenAI 兼容接口) 进行基于常识 + 已知对象集合的推理，输出候选相关对象及组成评分的组件，降低随机性。
3. 对 LLM 预测而目前地图中未明确与目标存在历史关系的对象, 通过 predict_loss_rate 进行分数折减。

输出 JSON 结构(期望 LLM 遵守):
{
  "target": "cup",
  "candidates": [
	 {
	   "obj_id": "coffee_table_1_R1",
	   "label": "coffee_table",
	   "score_components": {
		  "FA": 0.90,            # Functional Affinity 功能/使用情境贴近
		  "SAS": 0.78,           # Spatial Anchor Stability = 0.5*stability + 0.5*sigmoid(N)
		  "OR": 0.91,            # Observed Reliability = cfd 或估计值
		  "RP": 0.80,            # Relation Predictability (若历史共现/常识共现)
		  "CC": 0.85             # Category Complementarity 支撑/承放/容器/辅助关系
	   },
	   "predicted": false,       # true 表示纯预测(无历史关系), 需要后处理折减
	   "Rscore_base": 0.83,      # 未折减之前 0.3*FA + 0.2*SAS + 0.2*OR + 0.2*RP + 0.1*CC
	   "Rscore": 0.83,           # 若 predicted==true 则 = Rscore_base * (1 - predict_loss_rate) 或 * predict_factor
	   "reasons": "与杯子功能场景(喝水/放置)强相关, 常位于茶几表面, 稳定且多次更新." 
	 }
  ],
  "predict_candidates": [ ... 纯预测对象 ... ],
  "merged_Robjlist": [ "coffee_table_1_R1", "sink_1_R2", ... ],
  "scoring_formula": "Rscore_base = 0.3*FA + 0.2*SAS + 0.2*OR + 0.2*RP + 0.1*CC; Rscore = predicted ? Rscore_base * PREDICT_FACTOR : Rscore_base"
}

LLM 提示中显式给出评分维度与计算公式, 促使其输出稳定结构化结果, 减少随机波动。

使用方式:
from agent_Rscore import RScoreAgent
agent = RScoreAgent(api_key="YOUR_KEY", model="qwen-max")
result = agent.score_related_objects(target="cup")

NOTE: 不在仓库中写入任何真实 API KEY, 使用者在运行时传入。
"""

from __future__ import annotations
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml
from openai import OpenAI  # 兼容 OpenAI & DashScope (Qwen) 接口

MAP_PATH_DEFAULT = Path("RAG_Graph/test_save/map_mergedALL.json")
IDS_PATH_DEFAULT = Path("CLIP_RAG/map_mergedALL__openai-clip-vit-base-patch32__20250821-155707__b2a67722/ids.json")
CONFIG_PATH_DEFAULT = Path("config/map.yaml")


def sigmoid(x: float, k: float = 1.0, x0: float = 2.0) -> float:
	return 1.0 / (1.0 + math.exp(-k * (x - x0)))


class RScoreAgent:
	"""RScore 负责构建提示、调用 LLM、解析与后处理评分。"""

	def __init__(
		self,
		api_key: Optional[str] = None,
		base_url: Optional[str] = None,
		model: str = "qwen-max",
		predict_loss_rate: float = 0.8,
		map_path: Path = MAP_PATH_DEFAULT,
		ids_path: Path = IDS_PATH_DEFAULT,
		config_path: Path = CONFIG_PATH_DEFAULT,
		auto_load_config: bool = True,
	) -> None:
		self.api_key = api_key
		self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
		self.model = model
		# predict_loss_rate 现语义: keep factor (保留比例) 而非折损比例
		# predicted 对象 Rscore = base * predict_loss_rate
		self.predict_loss_rate = predict_loss_rate
		self.map_path = map_path
		self.ids_path = ids_path
		self.config_path = config_path
		self.config: Dict[str, Any] = {}
		self.top_k: Optional[int] = None
		self.Rscore_min: Optional[float] = None
		self.real_top_k: Optional[int] = None
		self.predict_top_k: Optional[int] = None
		self.output_path: Optional[str] = None
		self.merge_rate: float = 1.05  # 重复标签融合加权系数
		self.minimal_output: bool = True  # 控制台打印最小映射, 文件保存详细
		# Rscore 最终输出的最大值(区间 [0, RscoreValue_max])，默认 10，来自配置 agent_Rscore.RscoreValue_max
		self.Rscore_value_max: float = 1.0
		# RAG 指数门控参数:
		#   S_rag = (beta1*R_sim + beta2*R_cfd) * exp(-time_lambda*ΔT/(1+kappa*stability))
		# 解释:
		# - kappa 控制“稳定性对时间衰减的抵消强度”
		# - kappa 越大, 高 stability 物体衰减越慢; kappa=0 时退化为纯时间衰减
		self.beta1: float = 0.65
		self.beta2: float = 0.35
		self.time_lambda: float = 0.2  # 按“天”衰减
		self.kappa: float = 1.0
		self.rcfd_epsilon: float = 1e-8
		self.min_cooccur_count: int = 1
		# 创建客户端 (延迟创建: 按需)
		self._client: Optional[OpenAI] = None
		if auto_load_config:
			self.load_config(apply_params=True)

	# ---------------------- 配置加载 ----------------------
	def load_config(self, apply_params: bool = True) -> Dict[str, Any]:
		if not self.config_path.exists():
			return {}
		try:
			with open(self.config_path, 'r', encoding='utf-8') as f:
				cfg = yaml.safe_load(f) or {}
		except Exception:
			return {}
		self.config = cfg
		if not apply_params:
			return cfg
		# 解析路径配置
		map_cfg = (cfg.get('map_config') or {})
		clip_dir = map_cfg.get('CLIP_RAG_map_dir')
		if clip_dir:
			clip_dir_path = Path(clip_dir)
			# 若配置为相对路径, 以仓库根为基准
			if not clip_dir_path.is_absolute():
				clip_dir_path = Path.cwd() / clip_dir_path
			candidate_ids = clip_dir_path / 'ids.json'
			if candidate_ids.exists():
				self.ids_path = candidate_ids
		# 可选: 单独地图路径覆盖 (map_full_path)
		map_full_path = (cfg.get('agent_Rscore') or {}).get('map_full_path')
		if isinstance(map_full_path, str):
			if map_full_path.strip().upper() == 'NONE':
				# 标记不加载完整地图
				self.map_path = Path('__DISABLED__')
			elif map_full_path.strip():
				p = Path(map_full_path.strip())
				if not p.is_absolute():
					p = Path.cwd() / p
				self.map_path = p
		# agent_Rscore 参数
		agent_cfg = (cfg.get('agent_Rscore') or {})
		cfg_predict = agent_cfg.get('predict_loss_rate')
		if isinstance(cfg_predict, (int, float)):
			self.predict_loss_rate = float(cfg_predict)
		# top_k / Rscore_min
		# 旧 top_k 兼容 (已被 real_top_k / predict_top_k 取代)
		cfg_top_k = agent_cfg.get('top_k') if 'top_k' in agent_cfg else agent_cfg.get('top-k')
		if isinstance(cfg_top_k, int) and not agent_cfg.get('real_top_k'):
			self.real_top_k = cfg_top_k if cfg_top_k > 0 else None
		# 新参数
		cfg_real_top = agent_cfg.get('real_top_k')
		if isinstance(cfg_real_top, int):
			self.real_top_k = cfg_real_top if cfg_real_top > 0 else None
		cfg_pred_top = agent_cfg.get('predict_top_k')
		if isinstance(cfg_pred_top, int):
			self.predict_top_k = cfg_pred_top if cfg_pred_top > 0 else None
		cfg_rmin = agent_cfg.get('Rscore_min')
		if isinstance(cfg_rmin, (int, float)):
			self.Rscore_min = float(cfg_rmin) if cfg_rmin > 0 else None
		# 输出路径
		out_path_cfg = agent_cfg.get('output_path')
		if isinstance(out_path_cfg, str):
			val = out_path_cfg.strip()
			if val.upper() == 'NONE':
				self.output_path = None  # 显式禁用输出
			elif val:
				self.output_path = val
		# merge_rate
		cfg_merge = agent_cfg.get('merge_rate')
		if isinstance(cfg_merge, (int, float)) and cfg_merge > 0:
			self.merge_rate = float(cfg_merge)
		# 允许配置里提供 api key
		key_in_cfg = agent_cfg.get('qwen_APIKEY')
		if (not self.api_key) and key_in_cfg and 'YOUR_API_KEY' not in str(key_in_cfg):
			self.api_key = key_in_cfg
		# 读取/兼容 RscoreValue_max 作为最终标签分数的上限
		val_max = agent_cfg.get('RscoreValue_max', 1.0)
		try:
			val_max = float(val_max)
		except Exception:
			val_max = 1.0
		# 合理边界
		if not (val_max > 0):
			val_max = 1.0
		self.Rscore_value_max = val_max
		# 模型名称覆盖 (若配置提供)
		cfg_model = agent_cfg.get('model')
		if isinstance(cfg_model, str) and cfg_model.strip():
			self.model = cfg_model.strip()
		# RAG 指数门控评分参数
		rag_cfg = agent_cfg.get('rag_score') or {}
		beta1 = rag_cfg.get('beta1', self.beta1)
		beta2 = rag_cfg.get('beta2', self.beta2)
		try:
			beta1 = float(beta1)
			beta2 = float(beta2)
		except Exception:
			beta1, beta2 = self.beta1, self.beta2
		total = beta1 + beta2
		if total <= 0:
			self.beta1, self.beta2 = 0.65, 0.35
		else:
			self.beta1, self.beta2 = beta1 / total, beta2 / total
		try:
			self.time_lambda = max(0.0, float(rag_cfg.get('time_lambda', self.time_lambda)))
		except Exception:
			pass
		try:
			self.kappa = max(0.0, float(rag_cfg.get('kappa', self.kappa)))
		except Exception:
			pass
		try:
			self.rcfd_epsilon = max(1e-12, float(rag_cfg.get('epsilon', self.rcfd_epsilon)))
		except Exception:
			pass
		try:
			self.min_cooccur_count = max(0, int(rag_cfg.get('min_cooccur_count', self.min_cooccur_count)))
		except Exception:
			pass
		return cfg

	@property
	def client(self) -> OpenAI:
		if self._client is None:
			# 用户需在环境变量或参数中提供 api_key
			self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
		return self._client

	# ---------------------- 数据加载 ----------------------
	def _load_map(self) -> List[Dict[str, Any]]:
		# 若 map_path 被标记为禁用则返回空列表
		if str(self.map_path).endswith('__DISABLED__') or not self.map_path.exists():
			return []
		with open(self.map_path, "r", encoding="utf-8") as f:
			return json.load(f)

	def _load_ids(self) -> List[str]:
		with open(self.ids_path, "r", encoding="utf-8") as f:
			return json.load(f)

	def _index_objects(self, floors: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
		obj_index: Dict[str, Dict[str, Any]] = {}
		for floor in floors:
			for room in floor.get("rooms", []):
				for obj in room.get("objects", []):
					obj_index[obj["obj_id"]] = obj
		return obj_index

	def _aggregate_room_inventory(self, floors: List[Dict[str, Any]]) -> Dict[str, List[str]]:
		room_labels: Dict[str, List[str]] = {}
		for floor in floors:
			for room in floor.get("rooms", []):
				labels = []
				for obj in room.get("objects", []):
					labels.append(obj.get("label", ""))
				room_labels[room.get("room_id", "?")] = labels
		return room_labels

	# ---------------------- 提示词构建 ----------------------
	def _build_prompt(self, target: str, ids_list: List[str], obj_index: Dict[str, Dict[str, Any]], room_inventory: Dict[str, List[str]], *, existing_selected: List[str]=None, predicted_selected: List[str]=None, need_real: int=0, need_pred: int=0) -> str:
		existing_selected = existing_selected or []
		predicted_selected = predicted_selected or []
		# 构造对象属性摘要 (仅保留必要字段，防止提示过长)
		def brief(o: Dict[str, Any]) -> Dict[str, Any]:
			return {
				"obj_id": o.get("obj_id"),
				"label": o.get("label"),
				"room_id": o.get("room_id"),
				"N": o.get("N"),
				"stability": o.get("stability"),
				"cfd": o.get("cfd"),
				"R_objs_keys": list((o.get("R_objs") or {}).keys())[:6],  # 只展示少量 key 防止爆长度
			}

		if obj_index:
			inventory_snippets = [brief(obj_index[i]) for i in ids_list if i in obj_index]
		else:
			# 不加载地图模式: 仅通过 obj_id 推断 label (截取末两段前面的部分)
			def infer_label(oid: str) -> str:
				parts = oid.split('_')
				if len(parts) < 3:
					return oid
				return '_'.join(parts[:-2])
			inventory_snippets = [{
				"obj_id": oid,
				"label": infer_label(oid),
				"room_id": oid.split('_')[-1] if '_' in oid else None,
				"N": None,
				"stability": None,
				"cfd": None,
				"R_objs_keys": []
			} for oid in ids_list]

		room_summary_parts = []
		for rid, labels in room_inventory.items():
			# 聚合统计: label -> count
			freq: Dict[str, int] = {}
			for lb in labels:
				if not lb:
					continue
				freq[lb] = freq.get(lb, 0) + 1
			top_items = sorted(freq.items(), key=lambda x: -x[1])[:8]
			room_summary_parts.append({"room_id": rid, "top_labels": top_items})

		need_real = max(0, need_real)
		need_pred = max(0, need_pred)
		keep_factor = self.predict_loss_rate
		existing_labels = [self._infer_label_from_id(i) for i in ids_list]
		already_real = [self._infer_label_from_id(i) for i in existing_selected]
		scoring_spec = (
			"你是一个室内语义关系推理助手。目标物体 T='" + target + "'.\n"
			"定义: \n"
			"- real_candidates: 必须从给定的现有对象 ID 列表中选择 (ids_list)。\n"
			"- predicted_candidates: 不在 ids_list 的合理潜在相关对象 (常识推测)。与 ids_list 有重叠的标签不要放在 predicted 中。\n"
			f"本轮仍需 real_candidates >= {need_real} 个, predicted_candidates >= {need_pred} 个 (若不足则尽量多给)。\n"
			"不得重复已选 real IDs: " + ",".join(existing_selected) + "\n"
			"评分组件 (0-1 两位小数): FA,SAS,OR,RP,CC。SAS = 体现空间锚点稳定度: 结合 stability(0~1) 与出现次数 N 的可靠性; 高 stability(>=0.6) 更优。避免选择低 stability (<0.3) 的临时/易移动物品作为锚点, predicted 尽量也要稳定。\n"
			"基础公式 base_score=0.3*FA+0.2*SAS+0.2*OR+0.2*RP+0.1*CC。\n"
			f"predicted 最终将乘以 keep_factor={keep_factor:.2f} (你只需输出 base_score)。\n"
			"输出 JSON 严格格式: { 'real_candidates': [...], 'predicted_candidates': [...] }。\n"
			"real_candidates 每项: {obj_id,label,components:{FA,SAS,OR,RP,CC},base_score,reason}。\n"
			"predicted_candidates 每项: {label,components:{FA,SAS,OR,RP,CC},base_score,reason} (无 obj_id)。\n"
			"所有 label 使用英文小写。不要包含多余文本。"
		)

		prompt_dict = {
			"instruction": scoring_spec,
			"target": target,
			"ids_list": ids_list,
			"objects_brief": inventory_snippets[:120],  # 截断防超长
			"room_label_summary": room_summary_parts,
			"already_real_ids": existing_selected,
			"already_predicted_labels": predicted_selected,
			"need_real": need_real,
			"need_pred": need_pred,
		}
		return json.dumps(prompt_dict, ensure_ascii=False)

	def _infer_label_from_id(self, obj_id: str) -> str:
		parts = obj_id.split('_')
		if len(parts) < 3:
			return obj_id
		return '_'.join(parts[:-2])

	def _safe_float(self, v: Any, default: float = 0.0) -> float:
		try:
			return float(v)
		except Exception:
			return default

	def _to_utc_ts(self, raw: Any) -> Optional[float]:
		if raw is None:
			return None
		if isinstance(raw, (int, float)):
			val = float(raw)
			if val > 0:
				return val
			return None
		if isinstance(raw, str):
			txt = raw.strip()
			if not txt:
				return None
			try:
				v = float(txt)
				if v > 0:
					return v
			except Exception:
				pass
			# ISO 格式
			try:
				if txt.endswith('Z'):
					txt = txt[:-1] + '+00:00'
				dt = datetime.fromisoformat(txt)
				if dt.tzinfo is None:
					dt = dt.replace(tzinfo=timezone.utc)
				return dt.timestamp()
			except Exception:
				return None
		return None

	def _delta_days(self, obj: Dict[str, Any], now_ts: float) -> float:
		ts = self._to_utc_ts(obj.get('last_update_time'))
		if ts is None:
			# 未写入时间戳时回退为 0 天, 仅为兼容旧数据。
			# 工程上应尽量保证每次构图/更新都写 last_update_time，避免门控退化。
			return 0.0
		delta_sec = max(0.0, now_ts - ts)
		return delta_sec / 86400.0

	def _stability_from_obj_or_components(self, obj: Optional[Dict[str, Any]], comps: Dict[str, Any]) -> float:
		if obj is not None and obj.get('stability') is not None:
			return min(1.0, max(0.0, self._safe_float(obj.get('stability'), 0.5)))
		# 回退: SAS 是 stability 与出现次数的混合，可作近似稳定度
		return min(1.0, max(0.0, self._safe_float(comps.get('SAS', 0.5), 0.5)))

	def _build_target_relation_context(self, target: str, obj_index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
		target_objs = [o for o in obj_index.values() if (o.get('label') or '').lower() == target.lower()]
		objid_to_label = {oid: (o.get('label') or self._infer_label_from_id(oid)) for oid, o in obj_index.items()}
		obj_stats: Dict[str, Dict[str, float]] = {}
		label_stats: Dict[str, Dict[str, float]] = {}
		for t in target_objs:
			rel_map = t.get('R_objs') or {}
			if not isinstance(rel_map, dict):
				continue
			for rel_oid, rel_meta in rel_map.items():
				if not isinstance(rel_meta, dict):
					continue
				nr = max(0.0, self._safe_float(rel_meta.get('Nr', 0.0), 0.0))
				rcfd = min(1.0, max(0.0, self._safe_float(rel_meta.get('Rcfd', 0.0), 0.0)))
				# obj 级
				st = obj_stats.setdefault(rel_oid, {'num': 0.0, 'den': 0.0, 'nr_sum': 0.0})
				st['num'] += rcfd * nr
				st['den'] += nr
				st['nr_sum'] += nr
				# label 级
				lb = (objid_to_label.get(rel_oid) or self._infer_label_from_id(rel_oid)).lower()
				lst = label_stats.setdefault(lb, {'num': 0.0, 'den': 0.0, 'nr_sum': 0.0})
				lst['num'] += rcfd * nr
				lst['den'] += nr
				lst['nr_sum'] += nr
		def _finalize(x: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
			out: Dict[str, Dict[str, float]] = {}
			for k, v in x.items():
				den = v.get('den', 0.0)
				nr_sum = v.get('nr_sum', 0.0)
				if nr_sum < self.min_cooccur_count:
					rcfd = 0.0
				else:
					rcfd = (v.get('num', 0.0)) / (den + self.rcfd_epsilon)
				out[k] = {
					'rcfd': min(1.0, max(0.0, rcfd)),
					'nr_sum': nr_sum,
				}
			return out
		return {
			'obj': _finalize(obj_stats),
			'label': _finalize(label_stats),
		}

	def _compute_rag_score(self, base_sim: float, rcfd: float, delta_days: float, stability: float) -> float:
		base = self.beta1 * base_sim + self.beta2 * rcfd
		gate = math.exp(-(self.time_lambda * max(0.0, delta_days)) / (1.0 + self.kappa * min(1.0, max(0.0, stability))))
		score = base * gate
		return min(1.0, max(0.0, score))

	def _rerank_with_rag_formula(self, collected_real: List[Dict[str, Any]], collected_pred: List[Dict[str, Any]], target: str, obj_index: Dict[str, Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
		now_ts = datetime.now(timezone.utc).timestamp()
		rel_ctx = self._build_target_relation_context(target, obj_index)
		obj_rcfd = rel_ctx.get('obj', {})
		label_rcfd = rel_ctx.get('label', {})
		for r in collected_real:
			obj_id = r.get('obj_id')
			label = (r.get('label') or '').lower()
			obj = obj_index.get(obj_id) if obj_id else None
			base_sim = min(1.0, max(0.0, self._safe_float(r.get('base_score', 0.0), 0.0)))
			if obj_id and obj_id in obj_rcfd:
				rcfd = self._safe_float(obj_rcfd[obj_id].get('rcfd', 0.0), 0.0)
			else:
				rcfd = self._safe_float((label_rcfd.get(label) or {}).get('rcfd', 0.0), 0.0)
			stability = self._stability_from_obj_or_components(obj, r.get('components') or {})
			delta_days = self._delta_days(obj, now_ts) if obj else 0.0
			rag_score = self._compute_rag_score(base_sim, rcfd, delta_days, stability)
			r['Rscore'] = round(rag_score, 4)
			r['rag_terms'] = {
				'R_sim': round(base_sim, 4),
				'R_cfd': round(rcfd, 4),
				'delta_days': round(delta_days, 4),
				'stability': round(stability, 4),
			}
		for p in collected_pred:
			label = (p.get('label') or '').lower()
			base_sim = min(1.0, max(0.0, self._safe_float(p.get('base_score', 0.0), 0.0)))
			rcfd = self._safe_float((label_rcfd.get(label) or {}).get('rcfd', 0.0), 0.0)
			stability = self._stability_from_obj_or_components(None, p.get('components') or {})
			rag_score = self._compute_rag_score(base_sim, rcfd, 0.0, stability)
			# 预测项仍做保留系数折减
			p['Rscore'] = round(rag_score * self.predict_loss_rate, 4)
			p['rag_terms'] = {
				'R_sim': round(base_sim, 4),
				'R_cfd': round(rcfd, 4),
				'delta_days': 0.0,
				'stability': round(stability, 4),
			}
		return collected_real, collected_pred

	# ---------------------- LLM 调用 ----------------------
	def _call_llm(self, prompt: str) -> str:
		completion = self.client.chat.completions.create(
			model=self.model,
			messages=[
				{
					"role": "system",
					"content": (
						"You are a precise JSON generator for object relation scoring. "
						"All object labels in output must be English (use existing map labels only). "
						"Return only valid JSON; do not add explanations."
					)
				},
				{"role": "user", "content": prompt},
			],
			temperature=0.2,
			response_format={"type": "json_object"},
		)
		return completion.choices[0].message.content  # type: ignore[index]

	# ---------------------- 解析与后处理 ----------------------
	def _post_process_once(self, raw_json: str, ids_list: List[str]) -> Dict[str, Any]:
		try:
			data = json.loads(raw_json)
		except Exception:
			return {"error": "LLM_NON_JSON", "raw": raw_json}
		# 兼容字段名称 (新版 prompt 输出 real_candidates / predicted_candidates)
		real_arr = data.get("real_candidates") or data.get("candidates") or []
		pred_arr = data.get("predicted_candidates") or data.get("predict_candidates") or []
		ids_set = set(ids_list)
		keep_factor = self.predict_loss_rate
		processed_real = []
		processed_pred = []
		# 处理 real
		for item in real_arr:
			obj_id = item.get("obj_id")
			label = item.get("label") or (self._infer_label_from_id(obj_id) if obj_id else None)
			comps = item.get("components") or item.get("score_components") or {}
			for k in ["FA","SAS","OR","RP","CC"]:
				try:
					comps[k] = float(comps.get(k,0.0))
				except Exception:
					comps[k]=0.0
			base = item.get("base_score")
			if base is None:
				base = round(0.3*comps['FA']+0.2*comps['SAS']+0.2*comps['OR']+0.2*comps['RP']+0.1*comps['CC'],2)
			if obj_id and obj_id not in ids_set:
				# 若 LLM 错误把未知 id 放进 real，降级为 predicted
				processed_pred.append({"label":label,"base_score":base,"components":comps,"reason":item.get("reason",""),"predicted":True})
				continue
			processed_real.append({
				"obj_id": obj_id,
				"label": label,
				"base_score": base,
				"components": comps,
				"reason": item.get("reason",""),
				"predicted": False,
				"Rscore": base
			})
		# 处理 predicted
		for item in pred_arr:
			label = item.get("label")
			comps = item.get("components") or item.get("score_components") or {}
			for k in ["FA","SAS","OR","RP","CC"]:
				try:
					comps[k] = float(comps.get(k,0.0))
				except Exception:
					comps[k]=0.0
			base = item.get("base_score")
			if base is None:
				base = round(0.3*comps['FA']+0.2*comps['SAS']+0.2*comps['OR']+0.2*comps['RP']+0.1*comps['CC'],2)
			processed_pred.append({
				"label": label,
				"base_score": base,
				"components": comps,
				"reason": item.get("reason",""),
				"predicted": True,
				"Rscore": round(base*keep_factor,2)
			})
		return {"real": processed_real, "predicted": processed_pred}

	def _merge_iterations(self, collected_real, collected_pred, target: str, obj_index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
		collected_real, collected_pred = self._rerank_with_rag_formula(collected_real, collected_pred, target, obj_index)
		# 阈值过滤
		if self.Rscore_min is not None:
			collected_real = [r for r in collected_real if r['Rscore'] >= self.Rscore_min]
			collected_pred = [p for p in collected_pred if p['Rscore'] >= self.Rscore_min]
		# 排序及截断单列表
		collected_real.sort(key=lambda x: x['Rscore'], reverse=True)
		collected_pred.sort(key=lambda x: x['Rscore'], reverse=True)
		if self.real_top_k:
			collected_real = collected_real[:self.real_top_k]
		if self.predict_top_k:
			collected_pred = collected_pred[:self.predict_top_k]
		# 聚合标签
		label_scores: Dict[str, List[float]] = {}
		for r in collected_real:
			label_scores.setdefault(r['label'], []).append(r['Rscore'])
		for p in collected_pred:
			label_scores.setdefault(p['label'], []).append(p['Rscore'])
		# 计算融合 (平均 * merge_rate^(n-1))，得到原始 0~1 近似范围的分值(可能略>1 需截断)
		fused_raw: Dict[str, float] = {}
		for label, scores in label_scores.items():
			n = len(scores)
			avg = sum(scores)/n
			factor = (self.merge_rate ** (n-1)) if n > 1 else 1.0
			fused_raw[label] = round(avg * factor, 2)
		# 原始分数排序
		ordered_raw = dict(sorted(fused_raw.items(), key=lambda x: x[1], reverse=True))
		# 将原始分数截断到 [0,1] 后按 RscoreValue_max 线性缩放
		max_val = self.Rscore_value_max if self.Rscore_value_max > 0 else 10.0
		ordered_scaled: Dict[str, float] = {}
		for k, v in ordered_raw.items():
			clamped = min(max(v, 0.0), 1.0)  # 防止 merge_rate 导致略超 1
			ordered_scaled[k] = round(clamped * max_val, 2)
		fused_list = [{"label": k, "Rscore": v} for k, v in ordered_scaled.items()]
		# 详细输出里把最小映射放最前 (插入顺序)
		result: Dict[str, Any] = {
			# 对外主用：已缩放到 [0, RscoreValue_max]
			"Robj_scores": ordered_scaled,
			# 兼容/分析：0~1 原始分数(融合后、缩放前)
			"Robj_scores_0_1": ordered_raw,
			"real_candidates": collected_real,
			"predicted_candidates": collected_pred,
			"Robjlist_with_Rscores": fused_list,
			"merge_rate_used": self.merge_rate,
			"Rscore_max_value_used": max_val,
			"rag_score_params": {
				"beta1": round(self.beta1, 4),
				"beta2": round(self.beta2, 4),
				"time_lambda": self.time_lambda,
				"kappa": self.kappa,
				"epsilon": self.rcfd_epsilon,
				"min_cooccur_count": self.min_cooccur_count,
			},
			"scoring_formula": (
				"R_sim=base_score(from LLM); S_base=beta1*R_sim+beta2*R_cfd; "
				"G_ts=exp(-time_lambda*ΔT/(1+kappa*stability)); S_rag=S_base*G_ts; "
				"predicted *=%.2f (keep); duplicate avg * merge_rate^(n-1); "
				"final label score = clamp(raw,0,1) * RscoreValue_max"
			) % self.predict_loss_rate
		}
		return result

	# ---------------------- 主入口 ----------------------
	def score_related_objects(self, target: str) -> Dict[str, Any]:
		floors = self._load_map()
		ids_list = self._load_ids()
		obj_index = self._index_objects(floors)
		room_inv = self._aggregate_room_inventory(floors)
		need_real_total = self.real_top_k or 0
		need_pred_total = self.predict_top_k or 0
		collected_real: Dict[str, Any] = {}
		collected_pred: Dict[str, Any] = {}
		attempts = 0
		max_attempts = 4
		while attempts < max_attempts:
			attempts += 1
			need_real = max(0, need_real_total - len(collected_real)) if need_real_total else (0 if attempts>1 else 5)
			need_pred = max(0, need_pred_total - len(collected_pred)) if need_pred_total else (0 if attempts>1 else 3)
			if need_real == 0 and need_pred == 0:
				break
			prompt = self._build_prompt(
				target,
				ids_list,
				obj_index,
				room_inv,
				existing_selected=list(collected_real.keys()),
				predicted_selected=list(collected_pred.keys()),
				need_real=need_real,
				need_pred=need_pred,
			)
			raw = self._call_llm(prompt)
			parsed = self._post_process_once(raw, ids_list)
			# 追加 real
			for r in parsed.get('real', []):
				key = r.get('obj_id')
				if not key or key in collected_real:
					continue
				collected_real[key] = r
			# 追加 predicted (按 label 去重)
			for p in parsed.get('predicted', []):
				label = p.get('label')
				if not label or label in collected_pred:
					continue
				collected_pred[label] = p
			# 终止条件
			if (need_real_total and len(collected_real) >= need_real_total) and (need_pred_total and len(collected_pred) >= need_pred_total or not need_pred_total):
				break
		# 合并
		final = self._merge_iterations(list(collected_real.values()), list(collected_pred.values()), target, obj_index)
		# 增补元信息 (始终存在于文件中)
		final['config_ids_path'] = str(self.ids_path)
		final['predict_loss_rate_used'] = self.predict_loss_rate
		final['attempts'] = attempts
		# 保存输出 (配置或外部传入路径)
		self._maybe_save_result(final, target)
		return final

	def _maybe_save_result(self, result: Dict[str, Any], target: str, override_path: Optional[str]=None):
		path_cfg = (override_path or self.output_path)
		if not path_cfg:
			return
		path_cfg = path_cfg.strip()
		if not path_cfg or path_cfg.upper() == 'NONE':
			return
		p = Path(path_cfg)
		# 若包含模板 {target}
		if '{target}' in path_cfg:
			p = Path(path_cfg.format(target=target))
		# 若是目录 (以 / 结尾或者存在且为目录)
		if (path_cfg.endswith('/') or p.is_dir() or (not p.suffix)):
			if not p.exists():
				p.mkdir(parents=True, exist_ok=True)
			p = p / f"{target}_Rscore.json"
		try:
			with open(p, 'w', encoding='utf-8') as f:
				json.dump(result, f, ensure_ascii=False, indent=2)
		except Exception:
			pass

	# ---------------------- 导航集成 ----------------------
	def score_for_navigation(
		self,
		target: str,
		floors_data: List[Dict[str, Any]],
		clip_matched_ids: List[str],
	) -> Dict[str, Any]:
		"""面向导航循环的轻量 Agent 评分 (单次 LLM 调用).

		Args:
			target: 查询目标 (如 "microwave")
			floors_data: 地图数据 (list of floor dicts, 来自 Floor.to_dict())
			clip_matched_ids: CLIP 匹配到的物体 ID 列表

		Returns:
			{
				"obj_scores": {obj_id: float},       # 物体级 S_agent(i)
				"room_priors": {room_id: float},     # 房间级 P_agent(room|q)
				"raw_response": str,                 # LLM 原始响应
			}
		"""
		obj_index = self._index_objects(floors_data)
		room_inv = self._aggregate_room_inventory(floors_data)

		prompt = self._build_nav_prompt(target, clip_matched_ids, obj_index, room_inv)

		try:
			raw = self._call_llm(prompt)
			result = self._parse_nav_response(raw, clip_matched_ids)
			result["raw_response"] = raw
			return result
		except Exception as e:
			return {
				"obj_scores": {},
				"room_priors": {},
				"raw_response": f"ERROR: {e}",
			}

	def _build_nav_prompt(
		self,
		target: str,
		ids_list: List[str],
		obj_index: Dict[str, Dict[str, Any]],
		room_inventory: Dict[str, List[str]],
	) -> str:
		"""构建面向导航的精简 LLM 提示词 (含房间先验)."""
		# 物体简要
		def brief(o: Dict[str, Any]) -> Dict[str, Any]:
			return {
				"obj_id": o.get("obj_id"),
				"label": o.get("label"),
				"room_id": o.get("room_id"),
				"stability": o.get("stability"),
				"cfd": o.get("cfd"),
			}

		inventory = [brief(obj_index[i]) for i in ids_list if i in obj_index]

		# 房间摘要
		room_summary = []
		for rid, labels in room_inventory.items():
			freq: Dict[str, int] = {}
			for lb in labels:
				if lb:
					freq[lb] = freq.get(lb, 0) + 1
			top = sorted(freq.items(), key=lambda x: -x[1])[:10]
			room_summary.append({"room_id": rid, "objects": top})

		prompt_dict = {
			"instruction": (
				f"You are an indoor object localization assistant. Target: '{target}'.\n"
				f"Given CLIP-matched candidate objects and room layouts:\n"
				f"1. Score each candidate object on how likely it is NEAR the target (0.0-1.0).\n"
				f"   Consider: functional affinity, spatial co-occurrence, room semantics.\n"
				f"2. For each room, estimate probability the target is in that room (0.0-1.0, sum≈1).\n"
				f"   Use common sense (e.g., microwave→kitchen, bed→bedroom).\n\n"
				f"Output JSON: {{'obj_scores': {{obj_id: score}}, 'room_priors': {{room_id: prob}}}}\n"
				f"Only use obj_ids from candidates. All scores 0.0-1.0. No extra text."
			),
			"target": target,
			"candidates": inventory[:50],
			"rooms": room_summary,
		}
		return json.dumps(prompt_dict, ensure_ascii=False)

	def _parse_nav_response(
		self,
		raw_json: str,
		valid_ids: List[str],
	) -> Dict[str, Any]:
		"""解析导航 LLM 响应."""
		try:
			data = json.loads(raw_json)
		except Exception:
			return {"obj_scores": {}, "room_priors": {}}

		obj_scores = {}
		raw_obj = data.get("obj_scores", {})
		valid_set = set(valid_ids)
		for oid, score in raw_obj.items():
			if oid in valid_set:
				try:
					obj_scores[oid] = min(1.0, max(0.0, float(score)))
				except (ValueError, TypeError):
					pass

		room_priors = {}
		raw_rooms = data.get("room_priors", {})
		for rid, prob in raw_rooms.items():
			try:
				room_priors[rid] = min(1.0, max(0.0, float(prob)))
			except (ValueError, TypeError):
				pass
		# 归一化房间先验
		total = sum(room_priors.values())
		if total > 0:
			room_priors = {k: v / total for k, v in room_priors.items()}

		return {"obj_scores": obj_scores, "room_priors": room_priors}


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--target", required=True, help="待查找目标物体标签, 如 cup")
	parser.add_argument("--api_key", required=False, help="Qwen / OpenAI 兼容 API KEY")
	parser.add_argument("--model", default="qwen-max", help="模型名称")
	parser.add_argument("--predict_loss_rate", type=float, default=0.4)
	parser.add_argument("--config", type=str, default=str(CONFIG_PATH_DEFAULT), help="配置文件路径 (map.yaml)")
	parser.add_argument("--no_auto_config", action='store_true', help="禁用自动加载配置")
	parser.add_argument("--save", type=str, help="覆盖保存路径(文件或目录/可含{target})")
	args = parser.parse_args()
	agent = RScoreAgent(
		api_key=args.api_key,
		model=args.model,
		predict_loss_rate=args.predict_loss_rate,
		config_path=Path(args.config),
		auto_load_config=not args.no_auto_config,
	)
	output = agent.score_related_objects(target=args.target)
	if args.save:
		agent._maybe_save_result(output, args.target, override_path=args.save)
	# 控制台仅打印最小映射
	if 'Robj_scores' in output:
		print(json.dumps({"Robj_scores": output['Robj_scores']}, ensure_ascii=False, indent=2))
	else:
		print(json.dumps(output, ensure_ascii=False, indent=2))


