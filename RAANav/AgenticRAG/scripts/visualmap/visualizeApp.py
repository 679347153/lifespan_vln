"""Streamlit 语义地图 3D 可视化入口

更新:
1. 2D 平面视图 → 3D 楼层视图 (房间外框上下两层 + 立柱, 对象点, 关系 3D 线)
2. 2D 层级图 → 3D 层次图 (按 level 分层 Z 轴: Floor/Room/Object)
3. 每个节点(楼层/房间/对象)显示名称与关键属性 (hover & 常显文字)
4. 侧栏控制: 视图模式 / Rcfd 过滤 / 标签显示 / 对象点大小

运行: streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events  # 新增用于捕获相机事件
import networkx as nx

from semantic_map import Floor, Room, Object

# ------------------------------- 颜色映射 ------------------------------- #
ROOM_COLOR_PALETTE = [
	'#AEC7E8', '#FFBB78', '#98DF8A', '#FF9896', '#C5B0D5', '#C49C94', '#F7B6D2', '#DBDB8D', '#9EDAE5', '#17BECF',
	'#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd'
]
OBJ_CLASS_COLOR_PALETTE = [
	'#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
	'#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173', '#3182bd', '#e6550d', '#31a354', '#756bb1', '#636363'
]
_room_color_cache: Dict[str, str] = {}
_obj_class_color_cache: Dict[str, str] = {}

def get_room_color(room_id: str) -> str:
	if room_id not in _room_color_cache:
		idx = len(_room_color_cache) % len(ROOM_COLOR_PALETTE)
		_room_color_cache[room_id] = ROOM_COLOR_PALETTE[idx]
	return _room_color_cache[room_id]

def get_obj_class_color(label: str) -> str:
	key = (label or 'unknown').lower()
	if key not in _obj_class_color_cache:
		idx = len(_obj_class_color_cache) % len(OBJ_CLASS_COLOR_PALETTE)
		_obj_class_color_cache[key] = OBJ_CLASS_COLOR_PALETTE[idx]
	return _obj_class_color_cache[key]

def hex_to_rgba(hex_color: str, alpha: float) -> str:
	hex_color = hex_color.lstrip('#')
	if len(hex_color) == 3:
		hex_color = ''.join(c*2 for c in hex_color)
	r = int(hex_color[0:2], 16)
	g = int(hex_color[2:4], 16)
	b = int(hex_color[4:6], 16)
	return f'rgba({r},{g},{b},{alpha})'

# ------------------------------- 数据加载 ------------------------------- #


@st.cache_data(show_spinner=False)
def load_json_map(path: str, _version: int = 0) -> List[Floor]:
	"""加载语义地图 JSON -> List[Floor]

	支持两种顶层结构:
	1. list[FloorLike]: 每个元素含 floor_id 且有 rooms 字段 (标准格式)
	2. list[RoomLike]: 每个元素含 room_id (且通常含 floor_id) 但没有外层 floor 包裹
	   -> 自动按 floor_id 分组构造 Floor 对象

	_version: 强制失效缓存用。
	"""
	p = Path(path)
	if not p.exists():
		raise FileNotFoundError(f"文件不存在: {path}")
	with p.open('r', encoding='utf-8') as f:
		data = json.load(f)
	if not isinstance(data, list):
		raise ValueError("期望顶层为 list")
	if not data:
		return []
	# 判定是否为 floor 列表
	is_floor_list = all(isinstance(item, dict) and 'floor_id' in item and 'rooms' in item for item in data if isinstance(item, dict))
	if is_floor_list:
		return [Floor.from_dict(fr) for fr in data]
	# 判定是否为 room 列表 (允许部分缺失 floor_id, 用 F1 兜底)
	is_room_list = all(isinstance(item, dict) and 'room_id' in item for item in data if isinstance(item, dict))
	if is_room_list:
		# 按 floor_id 分组
		floor_groups: Dict[str, List[Room]] = {}
		for rdict in data:
			if not isinstance(rdict, dict):
				continue
			floor_id = rdict.get('floor_id') or 'F1'
			try:
				room_obj = Room.from_dict(rdict)
			except Exception:
				continue
			room_obj.floor_id = floor_id
			floor_groups.setdefault(floor_id, []).append(room_obj)
		floors: List[Floor] = []
		for fid, rooms in sorted(floor_groups.items(), key=lambda kv: kv[0]):
			floors.append(Floor(floor_id=fid, rooms=rooms))
		return floors
	# 其它情况 -> 尝试宽松解析: 若混合, 将带 rooms 的当 floor, 其余当单房间归入其 floor_id
	floors: List[Floor] = []
	leftover_rooms: List[Dict[str, any]] = []
	for item in data:
		if isinstance(item, dict) and 'floor_id' in item and 'rooms' in item:
			try:
				floors.append(Floor.from_dict(item))
			except Exception:
				pass
		elif isinstance(item, dict) and 'room_id' in item:
			leftover_rooms.append(item)
	if leftover_rooms:
		# 建立 index
		floor_index = {f.floor_id: f for f in floors}
		for rdict in leftover_rooms:
			fid = rdict.get('floor_id') or 'F1'
			if fid not in floor_index:
				floor_index[fid] = Floor(floor_id=fid, rooms=[])
				floors.append(floor_index[fid])
			try:
				r_obj = Room.from_dict(rdict)
				r_obj.floor_id = fid
				floor_index[fid].rooms.append(r_obj)
			except Exception:
				pass
		# 更新 room_ids
		for f in floors:
			f._room_ids = [r.room_id for r in f.rooms]
	if not floors:
		raise ValueError("无法识别的 JSON 结构: 既不是 floor 列表也不是 room 列表")
	return floors

# ---- 处理 room_name / description 为字典 ---- #

def _room_name_text(room: Room) -> str:
	rn = getattr(room, 'room_name', None)
	if isinstance(rn, dict):
		if 'brief' in rn and isinstance(rn['brief'], str):
			return rn['brief']
		numeric = [k for k in rn.keys() if str(k).isdigit()]
		if numeric:
			k_latest = max(numeric, key=lambda x: int(x))
			val = rn.get(k_latest)
			if isinstance(val, str):
				return val
		for v in rn.values():
			if isinstance(v, str):
				return v
		return room.room_id
	if isinstance(rn, str):
		return rn
	return room.room_id

def _room_desc_text(room: Room, limit: int = 80) -> str:
	desc = getattr(room, 'description', None)
	text = ''
	if isinstance(desc, dict):
		if 'brief' in desc and isinstance(desc['brief'], str):
			text = desc['brief']
		else:
			numeric = [k for k in desc.keys() if str(k).isdigit()]
			if numeric:
				latest = max(numeric, key=lambda x: int(x))
				val = desc.get(latest)
				if isinstance(val, str):
					text = val
			if not text:
				for v in desc.values():
					if isinstance(v, str):
						text = v; break
	elif isinstance(desc, str):
		text = desc
	if not text:
		text = '无描述'
	if len(text) > limit:
		return text[:limit] + '…'
	return text


def polygon_centroid(points: List[Dict[str, float]]) -> Tuple[float, float]:
	if not points:
		return (0.0, 0.0)
	# 简单平均 (凸/简单多边形近似即可)
	xs = [p['x'] for p in points]
	ys = [p['y'] for p in points]
	return (sum(xs) / len(xs), sum(ys) / len(ys))


def rcfd_to_color(rcfd: float) -> str:
	# 灰 (#969696) → 红 (#ff0000) 线性插值
	rcfd = max(0.0, min(1.0, rcfd))
	r1, g1, b1 = (150, 150, 150)
	r2, g2, b2 = (255, 0, 0)
	r = int(r1 + (r2 - r1) * rcfd)
	g = int(g1 + (g2 - g1) * rcfd)
	b = int(b1 + (b2 - b1) * rcfd)
	return f"rgb({r},{g},{b})"


############################ 3D 楼层视图 ############################


def build_floor_plan_3d_figure(floor: Floor, show_labels: bool, min_rcfd: float, obj_size: int, obj_diff: Dict[str,str]|None=None) -> go.Figure:
	"""构建 3D 楼层视图: 房间填充 + 对象 footprint + 关系线"""
	z_min = floor.z_range.get('z_min', 0.0) if isinstance(floor.z_range, dict) else 0.0
	z_max = floor.z_range.get('z_max', 4.0) if isinstance(floor.z_range, dict) else 4.0
	height_mid = (z_min + z_max) / 2.0
	fig = go.Figure()

	# 房间填充 (底部 Mesh + 边框 + 顶部) 使用各自唯一颜色
	for room in floor.rooms:
		pts = room.region or room.to_dict().get('Region', [])
		if not pts: continue
		room_color = get_room_color(room.room_id)
		x_coords = [p['x'] for p in pts]
		y_coords = [p['y'] for p in pts]
		if len(pts) >= 3:
			i_list=[]; j_list=[]; k_list=[]
			for i in range(1, len(pts)-1):
				i_list.append(0); j_list.append(i); k_list.append(i+1)
			fig.add_trace(go.Mesh3d(
				x=x_coords, y=y_coords, z=[z_min]*len(pts),
				i=i_list, j=j_list, k=k_list,
				color=room_color, opacity=0.30,
				name=f"Room {room.room_id}", showlegend=False
			))
		xs = x_coords + [pts[0]['x']]; ys = y_coords + [pts[0]['y']]
		room_desc = _room_desc_text(room)
		room_name_txt = _room_name_text(room)
		fig.add_trace(go.Scatter3d(x=xs, y=ys, z=[z_min]*len(xs), mode='lines', line=dict(color=room_color, width=3),
			name=f"Room {room.room_id}", hoverinfo='text', hovertext=f"Room: {room.room_id}<br>Name: {room_name_txt}<br>Objs: {len(room.objects)}<br>Doors: {len(room.door_positions)}<br>N: {room.N if room.N is not None else 'NA'}<br>Desc: {room_desc}", showlegend=False))
		fig.add_trace(go.Scatter3d(x=xs, y=ys, z=[z_max]*len(xs), mode='lines', line=dict(color=hex_to_rgba(room_color,0.45), width=2, dash='dot'), hoverinfo='skip', showlegend=False))
		for (x,y) in zip(xs[:-1], ys[:-1]):
			fig.add_trace(go.Scatter3d(x=[x,x], y=[y,y], z=[z_min,z_max], mode='lines', line=dict(color=hex_to_rgba(room_color,0.35), width=1), hoverinfo='skip', showlegend=False))

	# 对象 footprint + label + hover
	obj_centers: Dict[str, Tuple[float,float,float]] = {}
	for room in floor.rooms:
		for obj in room.objects:
			pts = obj.region or []
			status_color_map = {
				'added':'#2ca02c', 'removed':'#d62728', 'moved':'#ff7f0e', 'changed':'#9467bd'
			}
			diff_status = obj_diff.get(obj.obj_id) if obj_diff else None
			label_color = status_color_map.get(diff_status, get_obj_class_color(obj.label))
			if len(pts) >= 3:
				xc = [p['x'] for p in pts]; yc=[p['y'] for p in pts]
				i_list=[]; j_list=[]; k_list=[]
				for i in range(1, len(pts)-1):
					i_list.append(0); j_list.append(i); k_list.append(i+1)
				fig.add_trace(go.Mesh3d(x=xc,y=yc,z=[height_mid]*len(pts), i=i_list,j=j_list,k=k_list, color=label_color, opacity=0.55, showlegend=False, name=obj.label))
				xs = xc + [pts[0]['x']]; ys = yc + [pts[0]['y']]
				fig.add_trace(go.Scatter3d(x=xs,y=ys,z=[height_mid]*len(xs),mode='lines', line=dict(color=label_color, width=2), hoverinfo='skip', showlegend=False))
			cx, cy = polygon_centroid(pts)
			cz = height_mid
			obj_centers[obj.obj_id] = (cx, cy, cz)
			if show_labels:
				fig.add_trace(go.Scatter3d(x=[cx],y=[cy],z=[cz],mode='text',text=[obj.label],textposition='middle center',showlegend=False,hoverinfo='skip'))
			obj_desc = (obj.description[:60] + '…') if (getattr(obj,'description',None) and len(obj.description)>60) else (getattr(obj,'description',None) or '无描述')
			# Hover 中加入每个关系对象的 Nt/Nr/Rcfd 汇总 (仅前若干避免过长)
			_rel_preview = ''
			if obj.R_objs:
				parts=[]
				for i,(tgt, meta) in enumerate(list(obj.R_objs.items())[:5]):
					nt = meta.get('Nt','-'); nr = meta.get('Nr','-'); rcfd = meta.get('Rcfd','-')
					parts.append(f"{tgt}(Nt={nt},Nr={nr},Rcfd={rcfd})")
				_rel_preview = '<br>' + '<br>'.join(parts)
			fig.add_trace(go.Scatter3d(x=[cx],y=[cy],z=[cz],mode='markers',marker=dict(size=1,color='rgba(0,0,0,0)'),hoverinfo='text',hovertext=f"{obj.label} | {obj.obj_id}<br>room={room.room_id}<br>relations={len(obj.R_objs)}<br>N={obj.N if obj.N is not None else 'NA'}{_rel_preview}<br>Desc: {obj_desc}",showlegend=False))

	# 关系线 (去重)
	drawn = set()
	for room in floor.rooms:
		for obj in room.objects:
			for tgt, meta in (obj.R_objs or {}).items():
				rcfd = float(meta.get('Rcfd', 0.0))
				if rcfd < min_rcfd:
					continue
				key = tuple(sorted([obj.obj_id, tgt]))
				if key in drawn:
					continue
				drawn.add(key)
				if tgt not in obj_centers or obj.obj_id not in obj_centers:
					continue
				x0, y0, z0 = obj_centers[obj.obj_id]
				x1, y1, z1 = obj_centers[tgt]
				meta_self = obj.R_objs.get(tgt, {}) if obj.R_objs else {}
				nt = meta_self.get('Nt','-'); nr = meta_self.get('Nr','-')
				fig.add_trace(go.Scatter3d(
					x=[x0, x1], y=[y0, y1], z=[z0, z1],
					mode='lines',
					line=dict(color=rcfd_to_color(rcfd), width=2 + 4 * rcfd),
					hoverinfo='text', hovertext=f"{obj.obj_id} ↔ {tgt}<br>Rcfd={rcfd:.2f}<br>Nt={nt} Nr={nr}",
					showlegend=False
				))

	fig.update_layout(
		title=f"Floor {floor.floor_id} 3D 楼层视图",
		scene=dict(
			xaxis=dict(title='X'),
			yaxis=dict(title='Y'),
			zaxis=dict(title='Z'),
			aspectmode='data'
		),
		margin=dict(l=10, r=10, t=50, b=10),
		template='plotly_white',
		height=700
	)
	return fig


############################ 多楼层/房间/对象 组合视图 ############################


def build_multi_selection_figure(
	floors: List[Floor],
	selected_floors: set,
	selected_rooms: set,
	selected_objects: set,
	show_labels: bool,
	min_rcfd: float,
	obj_size: int,
	show_doors: bool = True,
	floor_gap: float = 6.0,
	object_plane_ratio: float = 0.5,
	obj_diff: Dict[str,str] | None = None,
) -> go.Figure:
	"""根据多层级选择(楼层/房间/对象)构建统一 3D 图.

	规则:
	- 选中楼层: 显示该楼层全部房间轮廓与全部对象。
	- 选中房间(未选中楼层): 显示该房间轮廓与其对象。
	- 选中对象(未选中其房间/楼层): 仅显示该对象点; 其关系对象作为淡化点与淡化线 (若存在数据)。
	- 关系线跨楼层绘制; Rcfd < min_rcfd 过滤。
	- 多楼层竖直堆叠: 为避免重叠, 按出现顺序给每个逻辑楼层增加 z 偏移 offset = index * gap.
	"""
	fig = go.Figure()
	# 预收集所有对象索引 (id -> (floor, room, object))
	obj_index: Dict[str, Tuple[Floor, Room, Object]] = {}
	for f in floors:
		for r in f.rooms:
			for o in r.objects:
				obj_index[o.obj_id] = (f, r, o)

	# 需要渲染的楼层列表(如果楼层被选或其下有被选房间或对象)
	active_floors: List[Floor] = []
	for f in floors:
		if (f.floor_id in selected_floors) or any(
			(r.room_id in selected_rooms) or any(o.obj_id in selected_objects for o in r.objects)
			for r in f.rooms
		):
			active_floors.append(f)

	# 楼层偏移 (可调楼层间距)
	floor_offsets: Dict[str, float] = {}
	for idx, f in enumerate(active_floors):
		base_height = (f.z_range.get('z_max', 4.0) - f.z_range.get('z_min', 0.0)) if isinstance(f.z_range, dict) else 4.0
		floor_offsets[f.floor_id] = idx * (base_height + floor_gap)

	# 收集点用于后续关系线绘制
	object_positions: Dict[str, Tuple[float, float, float]] = {}
	primary_objects: set = set()  # 直接选择来源 (楼层/房间/对象) 的对象
	ghost_objects: set = set()    # 仅作为关系目标出现

	# 收集门坐标 (统一绘制减少 trace 数)
	door_x: List[float] = []
	door_y: List[float] = []
	door_z: List[float] = []
	door_hover: List[str] = []

	for f in active_floors:
		offset = floor_offsets[f.floor_id]
		z_min = (f.z_range.get('z_min', 0.0) if isinstance(f.z_range, dict) else 0.0) + offset
		z_max = (f.z_range.get('z_max', 4.0) if isinstance(f.z_range, dict) else 4.0) + offset
		# 对象所在平面高度 (可调 0~1)
		ratio = max(0.0, min(1.0, object_plane_ratio))
		room_height_mid = z_min + (z_max - z_min) * ratio
		for r in f.rooms:
			room_selected = (f.floor_id in selected_floors) or (r.room_id in selected_rooms)
			room_has_selected_obj = any(o.obj_id in selected_objects for o in r.objects)
			if room_selected:
				pts = r.region or []
				if pts:
					room_col = get_room_color(r.room_id)
					xc=[p['x'] for p in pts]; yc=[p['y'] for p in pts]
					if len(pts)>=3:
						i_list=[]; j_list=[]; k_list=[]
						for i in range(1,len(pts)-1):
							i_list.append(0); j_list.append(i); k_list.append(i+1)
						fig.add_trace(go.Mesh3d(x=xc,y=yc,z=[z_min]*len(pts),i=i_list,j=j_list,k=k_list,color=room_col,opacity=0.30,showlegend=False))
					xs = xc+[pts[0]['x']]; ys=yc+[pts[0]['y']]
					room_desc = _room_desc_text(r)
					room_name_txt = _room_name_text(r)
					fig.add_trace(go.Scatter3d(x=xs,y=ys,z=[z_min]*len(xs),mode='lines',line=dict(color=room_col,width=3),hoverinfo='text',hovertext=f"Room {r.room_id} ({r.room_name})<br>objs={len(r.objects)}<br>N={r.N if r.N is not None else 'NA'}<br>Desc:{room_desc}",showlegend=False))
					fig.add_trace(go.Scatter3d(x=xs,y=ys,z=[z_max]*len(xs),mode='lines',line=dict(color=hex_to_rgba(room_col,0.45),width=2,dash='dot'),hoverinfo='skip',showlegend=False))
					for (x,y) in zip(xs[:-1], ys[:-1]):
						fig.add_trace(go.Scatter3d(x=[x,x],y=[y,y],z=[z_min,z_max],mode='lines',line=dict(color=hex_to_rgba(room_col,0.35),width=1),hoverinfo='skip',showlegend=False))
			# 对象点
			for o in r.objects:
				if (f.floor_id in selected_floors) or (r.room_id in selected_rooms) or (o.obj_id in selected_objects):
					primary_objects.add(o.obj_id)
				elif room_has_selected_obj:
					ghost_objects.add(o.obj_id)
				cx, cy = polygon_centroid(o.region)
				object_positions[o.obj_id] = (cx, cy, room_height_mid)
				# footprint
				pts = o.region or []
				if len(pts) >= 3:
					col = get_obj_class_color(o.label)
					xc=[p['x'] for p in pts]; yc=[p['y'] for p in pts]
					i_list=[]; j_list=[]; k_list=[]
					for i in range(1,len(pts)-1):
						i_list.append(0); j_list.append(i); k_list.append(i+1)
					fig.add_trace(go.Mesh3d(x=xc,y=yc,z=[room_height_mid]*len(pts),i=i_list,j=j_list,k=k_list,color=col,opacity=0.55 if o.obj_id in primary_objects else 0.25,showlegend=False))
					xs = xc+[pts[0]['x']]; ys=yc+[pts[0]['y']]
					fig.add_trace(go.Scatter3d(x=xs,y=ys,z=[room_height_mid]*len(xs),mode='lines',line=dict(color=col,width=2),hoverinfo='skip',showlegend=False))
					if show_labels and o.obj_id in primary_objects:
						fig.add_trace(go.Scatter3d(x=[cx],y=[cy],z=[room_height_mid],mode='text',text=[o.label],textposition='middle center',showlegend=False,hoverinfo='skip'))
					# 门渲染(在房间级别, 每个房间的门放在 z_min 平面上)
				if show_doors and getattr(r, 'door_positions', None):
					for i, d in enumerate(r.door_positions):
						if not isinstance(d, dict) or 'x' not in d or 'y' not in d: continue
						door_x.append(d['x'])
						door_y.append(d['y'])
						door_z.append(z_min)
						door_hover.append(f"Door | Room={r.room_id} ({i})")

	# 若只有单个或若干独立对象选择 (且其楼层/房间未被选), 需要把它们补充绘制
	for oid in selected_objects:
		if oid not in object_positions and oid in obj_index:
			f, r, o = obj_index[oid]
			if f.floor_id not in floor_offsets:
				# 该楼层未纳入 active_floors (应加)
				active_floors.append(f)
				floor_offsets[f.floor_id] = max(floor_offsets.values()) + 10 if floor_offsets else 0.0
			offset = floor_offsets[f.floor_id]
			z_min = (f.z_range.get('z_min', 0.0) if isinstance(f.z_range, dict) else 0.0) + offset
			z_max = (f.z_range.get('z_max', 4.0) if isinstance(f.z_range, dict) else 4.0) + offset
			room_height_mid = z_min + (z_max - z_min) * max(0.0, min(1.0, object_plane_ratio))
			cx, cy = polygon_centroid(o.region)
			object_positions[oid] = (cx, cy, room_height_mid)
			primary_objects.add(oid)

	# 关系 & 衍生 ghost 对象
	for oid in list(primary_objects):
		f, r, o = obj_index.get(oid, (None, None, None))
		if o is None:
			continue
		for tgt, meta in (o.R_objs or {}).items():
			rcfd = float(meta.get('Rcfd', 0.0))
			if rcfd < min_rcfd:
				continue
			if tgt not in object_positions and tgt in obj_index:
				# 添加 ghost 目标
				_f, _r, _o = obj_index[tgt]
				if _f.floor_id not in floor_offsets:
					floor_offsets[_f.floor_id] = max(floor_offsets.values()) + 10 if floor_offsets else 0.0
					offset = floor_offsets[_f.floor_id]
				else:
					offset = floor_offsets[_f.floor_id]
				z_min = (_f.z_range.get('z_min', 0.0) if isinstance(_f.z_range, dict) else 0.0) + offset
				z_max = (_f.z_range.get('z_max', 4.0) if isinstance(_f.z_range, dict) else 4.0) + offset
				mid = z_min + (z_max - z_min) * max(0.0, min(1.0, object_plane_ratio))
				cx, cy = polygon_centroid(_o.region)
				object_positions[tgt] = (cx, cy, mid)
				ghost_objects.add(tgt)

	# 绘制对象点 (主 / ghost)
	def obj_hover(o: Object, room: Room) -> str:
		obj_desc = (o.description[:60] + '…') if (o.description and len(o.description) > 60) else (o.description or '无描述')
		preview = ''
		if o.R_objs:
			parts=[]
			for tgt, meta in list(o.R_objs.items())[:6]:
				parts.append(f"{tgt}(Nt={meta.get('Nt','-')},Nr={meta.get('Nr','-')},Rcfd={meta.get('Rcfd','-')})")
			preview = '<br>' + '<br>'.join(parts)
		return (
			f"{o.label} | {o.obj_id}<br>room={room.room_id if room else 'NA'}<br>stability={o.stability if o.stability is not None else 'NA'}"
			f"<br>relations={len(o.R_objs)}<br>N={o.N if o.N is not None else 'NA'}{preview}<br>Desc: {obj_desc}"
		)

	primary_x=[]; primary_y=[]; primary_z=[]; primary_text=[]; primary_label=[]
	ghost_x=[]; ghost_y=[]; ghost_z=[]; ghost_text=[]
	for oid, (x,y,z) in object_positions.items():
		f, r, o = obj_index.get(oid, (None, None, None))
		if o is None:
			continue
		htxt = obj_hover(o, r)
		if oid in primary_objects:
			primary_x.append(x); primary_y.append(y); primary_z.append(z)
			primary_text.append(htxt)
			primary_label.append(o.label if show_labels else '')
		else:
			ghost_x.append(x); ghost_y.append(y); ghost_z.append(z)
			ghost_text.append(htxt)

	# 添加门 trace (统一)
	if show_doors and door_x:
		fig.add_trace(go.Scatter3d(
			x=door_x, y=door_y, z=door_z,
			mode='markers',
			marker=dict(size=6, color='#ffcc00', symbol='diamond'),
			hoverinfo='text', hovertext=door_hover,
			name='Doors'
		))

	if ghost_x:
		fig.add_trace(go.Scatter3d(
			x=ghost_x, y=ghost_y, z=ghost_z,
			mode='markers',
			marker=dict(size=max(4, int(obj_size*0.6)), color='rgba(120,120,120,0.35)'),
			hoverinfo='text', hovertext=ghost_text,
			name='Related'
		))
	if primary_x:
		# 根据 obj_diff 为每个 primary object 分配颜色
		local_status_map = {
			'added':'#2ca02c', 'removed':'#d62728', 'moved':'#ff7f0e', 'changed':'#9467bd'
		}
		colors=[]
		for oid in [oid for oid in object_positions.keys() if oid in primary_objects]:
			stt = obj_diff.get(oid) if obj_diff else None
			colors.append(local_status_map.get(stt, '#2ca02c'))
		fig.add_trace(go.Scatter3d(
			x=primary_x, y=primary_y, z=primary_z,
			mode='markers+text' if show_labels else 'markers',
			text=primary_label if show_labels else None,
			textposition='top center',
			marker=dict(size=obj_size, color=colors, opacity=0.95),
			hoverinfo='text', hovertext=primary_text,
			name='Objects'
		))

	# 关系线
	drawn=set()
	for oid in primary_objects.union(ghost_objects):
		f, r, o = obj_index.get(oid, (None, None, None))
		if o is None:
			continue
		for tgt, meta in (o.R_objs or {}).items():
			rcfd = float(meta.get('Rcfd', 0.0))
			if rcfd < min_rcfd:
				continue
			if tgt not in object_positions:
				continue
			key = tuple(sorted([oid, tgt]))
			if key in drawn:
				continue
			drawn.add(key)
			x0,y0,z0 = object_positions[oid]
			x1,y1,z1 = object_positions[tgt]
			alpha = 0.9 if (oid in primary_objects and tgt in primary_objects) else 0.35
			meta_self = o.R_objs.get(tgt, {}) if o.R_objs else {}
			fig.add_trace(go.Scatter3d(
				x=[x0,x1], y=[y0,y1], z=[z0,z1], mode='lines',
				line=dict(color=rcfd_to_color(rcfd), width=2+4*rcfd),
				opacity=alpha,
				hoverinfo='text', hovertext=f"{oid} ↔ {tgt}<br>Rcfd={rcfd:.2f}<br>Nt={meta_self.get('Nt','-')} Nr={meta_self.get('Nr','-')}", showlegend=False
			))

	fig.update_layout(
		title="多层级组合 3D 视图",
		scene=dict(
			xaxis=dict(title='X'),
			yaxis=dict(title='Y'),
			zaxis=dict(title='Z'),
			aspectmode='data'
		),
		margin=dict(l=10, r=10, t=50, b=10),
		template='plotly_white',
		height=800
	)
	return fig


############################ 多楼层 3D 层级视图（共享选择） ############################


def build_multi_hierarchy_3d_figure(
	floors: List[Floor],
	selected_floors: set,
	selected_rooms: set,
	selected_objects: set,
	show_labels: bool,
	obj_size: int,
	min_rcfd: float,
	show_regions: bool = False,
	level_spacing_override: Optional[Dict[str, float]] = None,
	force_multipliers: Optional[Dict[str, float]] = None,
	edge_width: int = 5,
	color_mode: str = 'level',  # 'level' | 'stability'
	enable_animation: bool = False,
	animation_frames: int = 0,
	cooling: float = 1.0,
	snapshot_interval: int = 10,
	obj_diff: Dict[str,str] | None = None,
) -> go.Figure:
	"""构建跨楼层多选择层级图 (Floor→Room→Object) + 对象间关系线.

	节点包含:
	- 被选楼层及其全部房间/对象
	- 被选房间及其对象
	- 被选对象
	- 被选对象的关系目标(ghost)

	关系线:
	- 结构边: Floor->Room, Room->Object
	- 对象关系边: Object<->Object (按 Rcfd 着色宽度, ghost 边淡化)
	"""
	# 收集需要的实体
	# 映射 obj_id -> (floor, room, obj)
	obj_index: Dict[str, Tuple[Floor, Room, Object]] = {
		o.obj_id: (f, r, o)
		for f in floors for r in f.rooms for o in r.objects
	}
	used_floors: set = set()
	used_rooms: set = set()
	primary_objects: set = set()
	ghost_objects: set = set()

	# 楼层级别
	for f in floors:
		if f.floor_id in selected_floors:
			used_floors.add(f.floor_id)
			for r in f.rooms:
				used_rooms.add(r.room_id)
				for o in r.objects:
					primary_objects.add(o.obj_id)
	# 房间
	for f in floors:
		for r in f.rooms:
			if r.room_id in selected_rooms and r.room_id not in used_rooms:
				used_floors.add(f.floor_id)
				used_rooms.add(r.room_id)
				for o in r.objects:
					primary_objects.add(o.obj_id)
	# 对象
	for oid in selected_objects:
		triple = obj_index.get(oid)
		if not triple:
			continue
		f, r, o = triple
		used_floors.add(f.floor_id)
		used_rooms.add(r.room_id)
		primary_objects.add(oid)

	# 关系拓展 ghost
	for oid in list(primary_objects):
		triple = obj_index.get(oid)
		if not triple:
			continue
		_f, _r, _o = triple
		for tgt, meta in (_o.R_objs or {}).items():
			if float(meta.get('Rcfd', 0.0)) < min_rcfd:
				continue
			if tgt not in primary_objects and tgt in obj_index:
				ghost_objects.add(tgt)
				_f2, _r2, _o2 = obj_index[tgt]
				used_floors.add(_f2.floor_id)
				used_rooms.add(_r2.room_id)

	# 构建扩展层级: 增加 Building 根节点
	G = nx.DiGraph()
	BUILDING_ID = "BUILDING"
	G.add_node(BUILDING_ID, level=-1, label='Building', kind='building')
	for f in floors:
		if f.floor_id not in used_floors: continue
		G.add_node(f.floor_id, level=0, label=f.floor_id, kind='floor', floor_obj=f)
		G.add_edge(BUILDING_ID, f.floor_id)
		for r in f.rooms:
			if r.room_id not in used_rooms: continue
			G.add_node(r.room_id, level=1, label=r.room_name, kind='room', room_obj=r)
			G.add_edge(f.floor_id, r.room_id)
			for o in r.objects:
				if o.obj_id not in primary_objects and o.obj_id not in ghost_objects: continue
				G.add_node(o.obj_id, level=2, label=o.label, kind='object', obj=o, room=r.room_id, primary=(o.obj_id in primary_objects))
				G.add_edge(r.room_id, o.obj_id)

	# 构建对象关系边列表 (用于布局+显示)
	relation_edges=[]
	for oid in primary_objects | ghost_objects:
		triple = obj_index.get(oid)
		if not triple: continue
		obj = triple[2]
		for tgt, meta in (obj.R_objs or {}).items():
			if tgt not in G.nodes: continue
			rcfd = float(meta.get('Rcfd', 0.0))
			if rcfd < min_rcfd: continue
			relation_edges.append((oid, tgt, rcfd))

	# ------------------ 动态垂直层级距离计算 ------------------ #
	floor_nodes_list = [n for n,d in G.nodes(data=True) if d.get('kind')=='floor']
	room_nodes_list = [n for n,d in G.nodes(data=True) if d.get('kind')=='room']
	object_nodes_list = [n for n,d in G.nodes(data=True) if d.get('kind')=='object']
	floor_count = len(floor_nodes_list)
	room_count = len(room_nodes_list)
	object_count = len(object_nodes_list)
	# 统计平均子节点数 (用于动态间距放大)
	avg_rooms_per_floor = (room_count / floor_count) if floor_count else 0
	# objects 只统计含对象的房间
	rooms_with_objs = [r for r in room_nodes_list if any(G.nodes[c]['kind']=='object' for c in G.successors(r))]
	objs_in_rooms = sum(len([c for c in G.successors(r) if G.nodes[c]['kind']=='object']) for r in rooms_with_objs)
	avg_objs_per_room = (objs_in_rooms / len(rooms_with_objs)) if rooms_with_objs else 0
	vertical_gap_scale = (force_multipliers or {}).get('vertical_gap_scale', 1.0)
	# 基础动态间距默认值 (受平均子节点数影响，sqrt 缓和)
	default_floor_room_gap = max(1.2, min(4.5, (1.0 + 0.25 * (avg_rooms_per_floor ** 0.5))))
	default_room_object_gap = max(1.4, min(5.5, (1.2 + 0.35 * (avg_objs_per_room ** 0.5))))
	default_building_offset = min(3.5, 0.6 + 0.25 * (floor_count ** 0.5))
	default_floor_room_gap *= vertical_gap_scale
	default_room_object_gap *= vertical_gap_scale
	default_building_offset *= vertical_gap_scale
	if level_spacing_override:
		floor_room_gap = level_spacing_override.get('floor_room_gap', default_floor_room_gap)
		room_object_gap = level_spacing_override.get('room_object_gap', default_room_object_gap)
		building_offset = level_spacing_override.get('building_offset', default_building_offset)
	else:
		floor_room_gap = default_floor_room_gap
		room_object_gap = default_room_object_gap
		building_offset = default_building_offset
	# z 映射
	levels = {
		-1: building_offset,  # Building 在上
		0: 0.0,                # Floors 基准面
		1: -floor_room_gap,    # Rooms 更下
		2: -floor_room_gap - room_object_gap  # Objects 最下
	}

	# --- 初始位置: 分层 + 圆环 (水平) --- #
	pos = {}
	# floors 在一个圆上
	import math, random
	floor_nodes = [n for n,d in G.nodes(data=True) if d.get('kind')=='floor']
	R_floor = 6 + 2*len(floor_nodes)
	for i, n in enumerate(floor_nodes):
		ang = 2*math.pi*i/max(1,len(floor_nodes))
		pos[n] = [R_floor*math.cos(ang), R_floor*math.sin(ang)]
	# rooms 初始在对应 floor 附近的小圆
	for fnode in floor_nodes:
		children = [c for c in G.successors(fnode) if G.nodes[c]['kind']=='room']
		R_room = 2.2 + 0.2*len(children)
		for j, rnode in enumerate(children):
			ang = 2*math.pi*j/max(1,len(children))
			fx, fy = pos[fnode]
			pos[rnode] = [fx + R_room*math.cos(ang), fy + R_room*math.sin(ang)]
	# objects 初始在各自 room 附近更小圆, 并依据关系分群
	# 关系连通分量(限制在该房间范围)
	for rnode,d in G.nodes(data=True):
		if G.nodes[rnode]['kind']!='room': continue
		objs = [c for c in G.successors(rnode) if G.nodes[c]['kind']=='object']
		if not objs: continue
		R_obj = 0.9 + 0.05*len(objs)
		# 简单：按索引分布
		for k, onode in enumerate(objs):
			ang = 2*math.pi*k/len(objs)
			rx, ry = pos[rnode]
			pos[onode] = [rx + R_obj*math.cos(ang), ry + R_obj*math.sin(ang)]
	# building 根
	pos[BUILDING_ID] = [0.0, 0.0]

	# ------------------ 关系簇细化 (修正版): 房间内 Rcfd 过滤 + 连通分量 + 模块度细分 (不再生成单节点簇) ------------------ #
	import networkx as _nxc
	relation_clusters: List[set] = []  # 仅包含>=2对象的真实关系簇
	obj_cluster: Dict[str,int] = {}
	adaptive_thresh = max(min_rcfd, 0.15)
	# 构建对象→房间
	obj_to_room: Dict[str,str] = {}
	for n,d in list(G.nodes(data=True)):
		if d.get('kind')=='object':
			obj_to_room[n] = d.get('room')
	room_objects: Dict[str,List[str]] = {}
	for oid,r in obj_to_room.items():
		room_objects.setdefault(r, []).append(oid)
	all_edges = relation_edges
	# 遍历每个房间构建簇
	for room_id, obj_list in room_objects.items():
		if len(obj_list) < 2:
			continue
		RG = _nxc.Graph(); RG.add_nodes_from(obj_list)
		for a,b,rc in all_edges:
			if rc < adaptive_thresh: continue
			if a in obj_list and b in obj_list:
				RG.add_edge(a,b,rc=rc)
		if RG.number_of_edges()==0:
			# 无任何关系边则不形成簇
			continue
		components = [set(c) for c in _nxc.connected_components(RG)]
		for comp in components:
			if len(comp) < 2:
				continue
			# 大簇尝试模块度细分
			if len(comp) > 6:
				try:
					from networkx.algorithms.community import greedy_modularity_communities
					sub = RG.subgraph(comp)
					coms = list(greedy_modularity_communities(sub))
					valid = [c2 for c2 in coms if len(c2) > 1]
					if len(valid) > 1:
						for sc in valid:
							relation_clusters.append(set(sc))
							cid = len(relation_clusters)-1
							for o in sc: obj_cluster[o]=cid
						continue
				except Exception:
					pass
			relation_clusters.append(comp)
			cid = len(relation_clusters)-1
			for o in comp: obj_cluster[o]=cid
	# 计算每簇平均 Rcfd
	cluster_avg_rcfd: Dict[int,float] = {}
	cluster_edge_sum: Dict[int,float] = {}
	cluster_edge_cnt: Dict[int,int] = {}
	for a,b,rc in all_edges:
		ca = obj_cluster.get(a); cb = obj_cluster.get(b)
		if ca is not None and cb is not None and ca==cb:
			cluster_edge_sum[ca] = cluster_edge_sum.get(ca,0.0)+rc
			cluster_edge_cnt[ca] = cluster_edge_cnt.get(ca,0)+1
	for cid in range(len(relation_clusters)):
		if cluster_edge_cnt.get(cid,0):
			cluster_avg_rcfd[cid] = cluster_edge_sum[cid]/cluster_edge_cnt[cid]
		else:
			cluster_avg_rcfd[cid] = 0.05

	# ------------------ 关系簇 (connected components) ------------------ #
	# ------------------ 自适应参数 + UI 传入倍率 ------------------ #
	n_total = G.number_of_nodes()
	iterations = int(min(300, 90 + 0.65 * n_total))
	scale = (n_total / 55.0) ** 0.5 if n_total else 1.0
	if scale < 1.0: scale = 1.0
	# 默认倍率
	rep_mul = (force_multipliers or {}).get('repulsion_scale', 1.0)
	parent_mul = (force_multipliers or {}).get('parent_pull_scale', 1.0)
	room_attr_mul = (force_multipliers or {}).get('room_attr_scale', 1.0)
	cluster_attr_mul = (force_multipliers or {}).get('cluster_attr_scale', 1.0)
	cluster_rep_mul = (force_multipliers or {}).get('cluster_rep_scale', 1.0)
	global_grav_mul = (force_multipliers or {}).get('global_grav_scale', 1.0)
	# ------------------ 力导向模拟 (增强 + 分层重力) ------------------ #
	snapshots: List[Dict[str,Tuple[float,float]]] = []

	def apply_forces(iterations=iterations, dt=0.02):
		max_step = 3.0 * scale  # 防止单步发散
		temp = 1.0
		collected = 0
		for it in range(iterations):
			disp = {n:[0.0,0.0] for n in G.nodes}
			# 同层节点斥力 (随规模+用户倍率)
			k_rep = 1.3 * scale * rep_mul
			nodes_by_level = {}
			for n,d in G.nodes(data=True):
				lvl = d['level']; nodes_by_level.setdefault(lvl, []).append(n)
			for lvl, nodes_list in nodes_by_level.items():
				for i in range(len(nodes_list)):
					for j in range(i+1, len(nodes_list)):
						n1 = nodes_list[i]; n2 = nodes_list[j]
						x1,y1 = pos[n1]; x2,y2 = pos[n2]
						dx = x1-x2; dy = y1-y2; dist2 = dx*dx+dy*dy+1e-6
						force = k_rep/dist2
						disp[n1][0] += dx*force; disp[n1][1] += dy*force
						disp[n2][0] -= dx*force; disp[n2][1] -= dy*force
			# 分层重力: 使用对子节点逐个矢量累加 (真实叠加)，再随子数加权
			for parent,d in G.nodes(data=True):
				children = [c for c in G.successors(parent)]
				if not children: continue
				px, py = pos[parent]
				fx = 0.0; fy = 0.0
				for c in children:
					cx, cy = pos[c]
					dx = px - cx; dy = py - cy
					dist2 = dx*dx + dy*dy + 1e-6
					# 距离越远吸引越大（线性/平方混合）
					f = (0.04 + 0.015 * len(children)**0.5) * parent_mul * (dist2 ** 0.5)
					fx += dx / (dist2**0.5) * f
					fy += dy / (dist2**0.5) * f
				disp[parent][0] -= fx
				disp[parent][1] -= fy
				# 子节点微弱回拉 (防止聚成一点)
				k_child_back = 0.008 * parent_mul
				for c in children:
					cx, cy = pos[c]
					dx = cx - px; dy = cy - py
					disp[c][0] += dx * k_child_back / len(children)
					disp[c][1] += dy * k_child_back / len(children)
			# 房间内部对象吸引 (cluster 层级较弱)
			k_room_attr = 0.06 * scale * room_attr_mul
			for rnode,d in G.nodes(data=True):
				if d.get('kind')!='room': continue
				children=[c for c in G.successors(rnode) if G.nodes[c]['kind']=='object']
				if len(children)<2: continue
				# centroid
				cx = sum(pos[c][0] for c in children)/len(children)
				cy = sum(pos[c][1] for c in children)/len(children)
				for c in children:
					dx = pos[c][0]-cx; dy = pos[c][1]-cy
					disp[c][0] -= dx*k_room_attr
					disp[c][1] -= dy*k_room_attr
			# 关系簇内部吸引 (热度 + 簇规模权重): 越低层(对象层)越强, 热度越高越强
			base_cluster_attr = 0.15 * scale * cluster_attr_mul
			for cid, comp in enumerate(relation_clusters):
				if len(comp) < 2:
					continue
				cx = sum(pos[o][0] for o in comp)/len(comp)
				cy = sum(pos[o][1] for o in comp)/len(comp)
				heat = cluster_avg_rcfd.get(cid, 0.05)  # 0~1
				k_cluster_attr = base_cluster_attr * (1.0 + 2.2 * heat) * (1 + (len(comp)**0.4))
				for o in comp:
					dx = pos[o][0]-cx; dy = pos[o][1]-cy
					disp[o][0] -= dx * k_cluster_attr
					disp[o][1] -= dy * k_cluster_attr
			# 关系边额外微调吸引 (保持连接)
			k_rel_edge = 0.08 * scale * parent_mul
			for a,b,_rc in relation_edges:
				xa,ya=pos[a]; xb,yb=pos[b]
				dx=xa-xb; dy=ya-yb; dist=math.sqrt(dx*dx+dy*dy)+1e-6
				target=1.2
				force = k_rel_edge*(dist-target)
				disp[a][0] -= dx/dist*force; disp[a][1] -= dy/dist*force
				disp[b][0] += dx/dist*force; disp[b][1] += dy/dist*force
			# 簇间排斥（用簇质心）
			k_cluster_rep = 2.0 * scale * cluster_rep_mul
			centroids=[]
			for comp in relation_clusters:
				if not comp: continue
				cx = sum(pos[o][0] for o in comp)/len(comp)
				cy = sum(pos[o][1] for o in comp)/len(comp)
				centroids.append((comp,cx,cy))
			for i in range(len(centroids)):
				for j in range(i+1,len(centroids)):
					compA,cxA,cyA = centroids[i]
					compB,cxB,cyB = centroids[j]
					dx=cxA-cxB; dy=cyA-cyB; dist2=dx*dx+dy*dy+1e-6
					force=k_cluster_rep/dist2
					# 将力作用分摊到成员
					for o in compA:
						disp[o][0]+=dx*force/len(compA); disp[o][1]+=dy*force/len(compA)
					for o in compB:
						disp[o][0]-=dx*force/len(compB); disp[o][1]-=dy*force/len(compB)
			# 全局重力 toward origin (防止发散) - 随规模降低
			k_grav = max(0.015, 0.04/scale) * global_grav_mul
			for n in G.nodes:
				x,y = pos[n]
				disp[n][0] -= x*k_grav
				disp[n][1] -= y*k_grav
			# 更新(锁定 Building 及少量衰减)
			for n in G.nodes:
				if n == BUILDING_ID: continue
				dx = disp[n][0]*dt*temp; dy = disp[n][1]*dt*temp
				step_len = (dx*dx+dy*dy)**0.5
				if step_len > max_step:
					s = max_step/step_len
					dx*=s; dy*=s
				pos[n][0] += dx
				pos[n][1] += dy
			# 冷却 (退火)
			if 0 < cooling < 1.0:
				temp *= cooling
			# 捕获动画快照
			if enable_animation and animation_frames > 0 and (it % snapshot_interval == 0) and collected < animation_frames:
				snapshots.append({n:(pos[n][0], pos[n][1]) for n in pos})
				collected += 1
	apply_forces()

	# 准备绘制数据
	level_z = { -1: levels[-1], 0: levels[0], 1: levels[1], 2: levels[2] }
	xs=[]; ys=[]; zs=[]; sizes=[]; colors=[]; texts=[]; short_texts=[]
	for node,data in G.nodes(data=True):
		x,y = pos[node]
		lvl = data['level']
		z = level_z.get(lvl, levels[2])
		xs.append(x); ys.append(y); zs.append(z)
		kind = data['kind']
		if kind=='building':
			color='#ff7f0e'; size=38; texts.append('Building Root'); short_texts.append('Building')
		elif kind=='floor':
			color='#1f77b4'; size=30
			fobj: Floor = data.get('floor_obj')
			room_cnt = len([n2 for n2,d2 in G.nodes(data=True) if d2.get('kind')=='room' and any(p==node for p in G.predecessors(n2))])
			obj_cnt = sum(1 for n2,d2 in G.nodes(data=True) if d2.get('kind')=='object')
			texts.append(f"Floor {data['label']}<br>rooms~{room_cnt}<br>objects~{obj_cnt}")
			short_texts.append(data['label'])
		elif kind=='room':
			room_obj: Room = data.get('room_obj')
			color='#7f7f7f'; size=24
			room_desc = _room_desc_text(room_obj, limit=50)
			room_name_txt = _room_name_text(room_obj)
			texts.append(f"Room {room_name_txt} ({room_obj.room_id})<br>objs={len(room_obj.objects)}<br>N={room_obj.N if room_obj.N is not None else 'NA'}<br>{room_desc}")
			short_texts.append(room_name_txt)
		else:
			obj: Object = data.get('obj')
			stb = obj.stability
			status_color_map = {'added':'#2ca02c','removed':'#d62728','moved':'#ff7f0e','changed':'#9467bd'}
			if obj_diff:
				stt = obj_diff.get(obj.obj_id)
			else:
				stt = None
			if stt in status_color_map:
				color = status_color_map[stt]
			else:
				if color_mode == 'stability':
					if stb is None:
						base=(170,170,170)
					else:
						g = int(180*stb+40); r = int(255-200*stb); base=(r,g,60)
				else:  # level (primary vs ghost)
					if data.get('primary'):
						base=(46,204,113)
					else:
						base=(160,160,160)
				alpha = 1.0 if data.get('primary') else 0.32
				color=f'rgba({base[0]},{base[1]},{base[2]},{alpha})'
			if stt in status_color_map:
				alpha = 1.0
			rels = len(obj.R_objs)
			size = max(obj_size, 10 + 5*(rels/5 if rels else 0))
			obj_desc = (obj.description[:50]+'…') if (obj.description and len(obj.description)>50) else (obj.description or '无描述')
			texts.append(f"Object {obj.label} ({obj.obj_id})<br>relations={rels}<br>diff={stt or 'same'}<br>N={obj.N if obj.N is not None else 'NA'}<br>{obj_desc}")
			short_texts.append(obj.label)
		colors.append(color); sizes.append(size)

	# 结构边 (父子) 使用直接直线连接, 支持自定义宽度
	edge_x=[]; edge_y=[]; edge_z=[]; edge_hover=[]
	for u,v in G.edges():
		x0,y0 = pos[u]; x1,y1 = pos[v]
		z0 = level_z.get(G.nodes[u]['level'], levels[2]); z1 = level_z.get(G.nodes[v]['level'], levels[2])
		edge_x += [x0, x1, None]
		edge_y += [y0, y1, None]
		edge_z += [z0, z1, None]
		edge_hover += [f"{u} → {v}", f"{u} → {v}", None]
	edge_trace = go.Scatter3d(x=edge_x,y=edge_y,z=edge_z,mode='lines', line=dict(color='rgba(70,70,70,0.85)', width=max(1,edge_width)), hoverinfo='text', hovertext=edge_hover, showlegend=False, name='结构边')

	# 对象关系边 (性能优化: Rcfd 分桶合并)
	rel_traces=[]; drawn=set(); buckets={}
	for a,b,rc in relation_edges:
		if a not in pos or b not in pos: continue
		k = tuple(sorted([a,b]));
		if k in drawn: continue
		drawn.add(k)
		bucket = int(rc*12)  # 0~12
		primary_pair = G.nodes[a].get('primary') and G.nodes[b].get('primary')
		alpha = 0.9 if primary_pair else 0.3
		buckets.setdefault((bucket,alpha), []).append((a,b,rc))
	for (bucket, alpha), edges in buckets.items():
		xs_e=[]; ys_e=[]; zs_e=[]; hovertexts=[]
		# 统一颜色用平均 rc
		avg_rc = sum(e[2] for e in edges)/len(edges)
		color = rcfd_to_color(avg_rc)
		width = 2+4*avg_rc
		for a,b,rc in edges:
			x0,y0 = pos[a]; x1,y1 = pos[b]
			z_rel = levels[2]
			xs_e += [x0,x1,None]; ys_e += [y0,y1,None]; zs_e += [z_rel,z_rel,None]
			hovertexts.append(f"{a} ↔ {b}<br>Rcfd={rc:.2f}")
		# 关系边 hovertexts 已包含 Rcfd, 为每条补充样本数量桶信息
		rel_traces.append(go.Scatter3d(
			x=xs_e, y=ys_e, z=zs_e, mode='lines',
			line=dict(color=color, width=width), opacity=alpha,
			hoverinfo='text', hovertext=[ht + f"<br>Bucket={bucket}" for ht in hovertexts], showlegend=False
		))

	# 节点散点
	node_trace = go.Scatter3d(
		x=xs, y=ys, z=zs,
		mode='markers+text' if show_labels else 'markers',
		text=short_texts if show_labels else None,
		textposition='top center',
		hoverinfo='text', hovertext=texts,
		marker=dict(size=sizes, color=colors, line=dict(color='#333', width=1), opacity=0.95),
		showlegend=False,
		name='节点'
	)

	fig = go.Figure(data=[edge_trace] + rel_traces + [node_trace])

	# 动画帧 (可选)
	if enable_animation and snapshots:
		frames=[]
		for snap in snapshots:
			xs_anim=[]; ys_anim=[]; zs_anim=[]
			for node,data in G.nodes(data=True):
				x,y = snap[node]
				lvl = data['level']
				z = level_z.get(lvl, levels[2])
				xs_anim.append(x); ys_anim.append(y); zs_anim.append(z)
			frames.append(go.Frame(data=[
				go.Scatter3d(x=edge_x,y=edge_y,z=edge_z,mode='lines', line=dict(color='rgba(70,70,70,0.85)', width=max(1,edge_width)), hoverinfo='skip', showlegend=False),
				*rel_traces,
				go.Scatter3d(x=xs_anim,y=ys_anim,z=zs_anim, mode='markers', marker=dict(size=sizes, color=colors, line=dict(color='#333', width=1), opacity=0.95), hoverinfo='skip')
			]))
		fig.frames = frames
		fig.update_layout(updatemenus=[{
			'type':'buttons',
			'buttons': [
				{'label':'▶ Play','method':'animate','args':[None,{'frame':{'duration':160,'redraw':True},'fromcurrent':True}]},
				{'label':'⏸ Pause','method':'animate','args':[[None],{'frame':{'duration':0},'mode':'immediate'}]},
			],
			'x':0,'y':1.1,'showactive':False
		}])

	# --- 区域蒙板 (房间 / 关系簇) --- #
	if show_regions:
		region_shape = (force_multipliers or {}).get('region_shape','rect')
		region_opacity = float((force_multipliers or {}).get('region_opacity',0.15))
		region_opacity = max(0.02, min(0.6, region_opacity))
		obj_positions = {n:(pos[n][0], pos[n][1]) for n,d in G.nodes(data=True) if d.get('kind')=='object'}
		# padding 动态参考：聚合越强 / 斥力越大 → padding 稍增
		pad_base_room = 0.5 + 0.15 * room_attr_mul + 0.1 * rep_mul
		pad_base_cluster = 0.4 + 0.12 * cluster_attr_mul + 0.1 * rep_mul
		# 收集房间对象点
		room_polys=[]
		for rnode,d in G.nodes(data=True):
			if d.get('kind')!='room': continue
			children=[c for c in G.successors(rnode) if G.nodes[c]['kind']=='object']
			pts=[obj_positions.get(c) for c in children if c in obj_positions]
			pts=[p for p in pts if p]
			if len(pts)<2: continue
			room_polys.append(('room', rnode, pts))
		# 关系簇
		cluster_polys=[]
		for comp in relation_clusters:
			if len(comp)<2: continue
			pts=[obj_positions.get(c) for c in comp if c in obj_positions]
			pts=[p for p in pts if p]
			if len(pts)<2: continue
			cluster_polys.append(('cluster', id(comp), pts))

		def compute_hull(points):
			# Graham scan 简化 (点数不大)
			pts = sorted(set(points))
			if len(pts) <= 2: return pts
			def cross(o,a,b):
				return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
			lower=[]
			for p in pts:
				while len(lower)>=2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()
				lower.append(p)
			upper=[]
			for p in reversed(pts):
				while len(upper)>=2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
				upper.append(p)
			return lower[:-1] + upper[:-1]

		def expand_poly(poly, pad):
			# 改为匀速外扩: 顶点沿质心方向单位向量平移固定 pad, 防止因距离不同导致整体漂移视觉错觉
			if pad <= 1e-6: return poly
			cx = sum(p[0] for p in poly)/len(poly)
			cy = sum(p[1] for p in poly)/len(poly)
			res=[]
			for x,y in poly:
				dx=x-cx; dy=y-cy
				norm=(dx*dx+dy*dy)**0.5
				if norm < 1e-6:
					res.append((x+pad, y))  # 若刚好在中心，随意平移
				else:
					res.append((x + dx/norm*pad, y + dy/norm*pad))
			return res

		# 统一保存: (kind, identifier, z_offset, polygon)
		# kind: 'room' / 'cluster'; identifier: room_id / cluster_id
		regions=[]
		for kind, room_id, pts in room_polys:
			pad = pad_base_room
			if region_shape=='凸包':
				poly = compute_hull(pts)
			else:
				xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
				poly=[(min(xs),min(ys)),(max(xs),min(ys)),(max(xs),max(ys)),(min(xs),max(ys))]
			poly = expand_poly(poly, pad)
			regions.append(('room', room_id, levels[2], poly))
		# 重新按 relation_clusters 顺序重建 cluster_polys, 以确保 identifier = cluster index
		cluster_polys_fixed=[]  # (cid, pts)
		for cid, comp in enumerate(relation_clusters):
			if len(comp) < 2: continue
			pts=[obj_positions.get(c) for c in comp if c in obj_positions]
			pts=[p for p in pts if p]
			if len(pts)<2: continue
			cluster_polys_fixed.append((cid, pts))
		for cid, pts in cluster_polys_fixed:
			pad = pad_base_cluster
			if region_shape=='凸包':
				poly = compute_hull(pts)
			else:
				xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
				poly=[(min(xs),min(ys)),(max(xs),min(ys)),(max(xs),max(ys)),(min(xs),max(ys))]
			poly = expand_poly(poly, pad)
			regions.append(('cluster', cid, levels[2]-0.08, poly))

		# 可选: 区域碰撞解决；默认关闭防止位置漂移
		if (force_multipliers or {}).get('region_no_overlap', False):
			def bbox(poly):
				xs=[p[0] for p in poly]; ys=[p[1] for p in poly]
				return [min(xs),max(xs),min(ys),max(ys)]
			for _ in range(12):
				changed=False
				for i in range(len(regions)):
					for j in range(i+1,len(regions)):
						a=regions[i]; b=regions[j]
						A=bbox(a[3]); B=bbox(b[3])
						if A[1]<B[0] or B[1]<A[0] or A[3]<B[2] or B[3]<A[2]:
							continue
						if a[0]=='room' and b[0]=='room':
							continue
						dx1 = B[1]-A[0]; dx2 = A[1]-B[0]
						dy1 = B[3]-A[2]; dy2 = A[3]-B[2]
						sep_x = dx1 if dx1<dx2 else -dx2
						sep_y = dy1 if dy1<dy2 else -dy2
						if abs(sep_x) < abs(sep_y): move=(sep_x*0.5,0.0)
						else: move=(0.0,sep_y*0.5)
						def shift(poly,m): return [(p[0]+m[0], p[1]+m[1]) for p in poly]
						if a[0]=='cluster' and b[0]=='cluster':
							regions[i]=(a[0],a[1],a[2],shift(a[3],(-move[0],-move[1])))
							regions[j]=(b[0],b[1],b[2],shift(b[3],(move[0],move[1])))
						elif a[0]=='cluster':
							regions[i]=(a[0],a[1],a[2],shift(a[3],(-move[0],-move[1])))
						elif b[0]=='cluster':
							regions[j]=(b[0],b[1],b[2],shift(b[3],(move[0],move[1])))
						changed=True
				if not changed: break

		# 按用户选择过滤区域
		show_room_regions = (force_multipliers or {}).get('show_room_regions', True)
		show_cluster_regions = (force_multipliers or {}).get('show_cluster_regions', True)
		# 绘制区域 (房间区使用房间调色板; 簇区仍使用热度渐变)
		for _rid,(kind, ident, z_offset, poly) in enumerate(regions):
			if kind=='room' and not show_room_regions:
				continue
			if kind=='cluster' and not show_cluster_regions:
				continue
			xs=[p[0] for p in poly]; ys=[p[1] for p in poly]
			xs.append(xs[0]); ys.append(ys[0])  # 封闭
			if kind=='room':
				base_hex = get_room_color(ident)
				color_fill = hex_to_rgba(base_hex, region_opacity)
				color_line = hex_to_rgba(base_hex, min(region_opacity+0.35,0.95))
			else:
				cid = ident  # 直接就是 cluster id
				avg_rc = max(0.0, min(1.0, cluster_avg_rcfd.get(cid,0.0)))
				r1,g1,b1 = (41,128,185)  # 冷色
				r2,g2,b2 = (255,87,34)   # 热色
				r = int(r1 + (r2-r1)*avg_rc)
				g = int(g1 + (g2-g1)*avg_rc)
				b = int(b1 + (b2-b1)*avg_rc)
				color_fill = f'rgba({r},{g},{b},{min(region_opacity+0.05,0.6)})'
				color_line = f'rgba({r},{g},{b},0.85)'
			fig.add_trace(go.Mesh3d(x=xs[:-1], y=ys[:-1], z=[z_offset]*(len(xs)-1), color=color_fill, opacity=region_opacity, showlegend=False))
			fig.add_trace(go.Scatter3d(x=xs, y=ys, z=[z_offset]*len(xs), mode='lines', line=dict(color=color_line, width=2, dash='dot' if kind=='cluster' else 'solid'), hoverinfo='skip', showlegend=False))

	fig.update_layout(
		title="多楼层 3D 层级关系图",
		scene=dict(
			xaxis=dict(title='X', showgrid=False, zeroline=False),
			yaxis=dict(title='Y', showgrid=False, zeroline=False),
			zaxis=dict(title='Level', showgrid=True),
			camera=dict(eye=dict(x=1.9, y=1.9, z=1.3)),
			aspectmode='data'
		),
		margin=dict(l=10, r=10, t=50, b=10),
		template='plotly_white',
		height=850,
		legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0)
	)
	return fig


# ------------------------------- 主 UI ------------------------------- #


def main():
	st.set_page_config(page_title="语义地图可视化", layout="wide")
	st.title("🗺️ RAG_Graph 3D (JSON)")
	default_file = 'RAG_Graph/map_history.json'
	with st.sidebar:
		st.header("配置")
		# ---- 初始化会话状态 ---- #
		if 'scan_refresh' not in st.session_state:
			st.session_state.scan_refresh = 0
		if 'main_version' not in st.session_state:
			st.session_state.main_version = 0
		if 'second_version' not in st.session_state:
			st.session_state.second_version = 0
		if 'loaded_main_path' not in st.session_state:
			st.session_state.loaded_main_path = None
		if 'loaded_second_path' not in st.session_state:
			st.session_state.loaded_second_path = None

		# ---- 扫描 JSON 文件 (递归) ---- #
		recursive = st.checkbox("递归扫描子目录", value=True)
		roots = []
		for cand in ['RAG_Graph', 'test_map']:
			p = Path(cand)
			if p.exists() and p.is_dir():
				roots.append(p)
		found_files: List[str] = []
		try:
			for root in roots:
				iterator = root.rglob('*.json') if recursive else root.glob('*.json')
				for p in iterator:
					if len(found_files) >= 1000:
						break
					found_files.append(p.as_posix())
		except Exception:
			pass
		found_files = sorted(set(found_files))
		col_scan_a, col_scan_b = st.columns([1,1])
		with col_scan_a:
			if st.button("刷新文件列表", use_container_width=True):
				st.session_state.scan_refresh += 1
		with col_scan_b:
			if st.button("清除缓存", use_container_width=True):
				st.session_state.main_version += 1; st.session_state.second_version += 1

		# ---- 主文件 (必须手动选择+加载) ---- #
		st.markdown("**主文件 (History / 基础)**")
		main_choice = st.selectbox("选择主文件", ["(未选择)"] + found_files, index=0, key='main_choice')
		if st.button("加载主文件", use_container_width=True, key='btn_load_main'):
			if main_choice != "(未选择)":
				st.session_state.loaded_main_path = main_choice
				st.session_state.main_version += 1
		# ---- 对比模式 ---- #
		compare_mode = st.checkbox("并排对比 (history vs now)", value=False)
		second_file = None
		if compare_mode:
			st.markdown("**对比文件 (Now)**")
			second_choice = st.selectbox("选择对比文件", ["(未选择)"] + found_files, index=0, key='second_choice')
			if st.button("加载对比文件", use_container_width=True, key='btn_load_second'):
				if second_choice != "(未选择)":
					st.session_state.loaded_second_path = second_choice
					st.session_state.second_version += 1
			move_thresh = st.slider("移动判定阈值(m)", 0.05, 1.0, 0.25, 0.01)
			ghost_removed = st.checkbox("ghost 显示历史删除对象", value=True)
			camera_sync = st.checkbox("相机同步 (试验)", value=False)
		else:
			move_thresh = 0.25
			ghost_removed = False
			camera_sync = False

		# ---- 视图设置 ---- #
		view_mode = st.radio("视图模式", ["3D 平面视图", "3D 层级视图"], index=0)
		show_labels = st.checkbox("显示节点文字", value=True)
		min_rcfd = st.slider("最小 Rcfd (过滤关系线)", 0.0, 1.0, 0.0, 0.01)
		obj_size = st.slider("对象节点大小", 4, 30, 12, 1)
		st.markdown("---")
		st.subheader("分层选择 (楼层/房间/对象)")
		if 'sel_version' not in st.session_state:
			st.session_state.sel_version = 0
		# 兼容旧变量引用
		data_file = st.session_state.loaded_main_path
		second_file = st.session_state.loaded_second_path if compare_mode else None

	floors: List[Floor] = []
	floors2: List[Floor] = []
	if data_file:
		try:
			floors = load_json_map(data_file, _version=st.session_state.get('main_version',0))
		except Exception as e:
			st.error(f"主文件加载失败: {e}")
	if compare_mode and second_file:
		try:
			floors2 = load_json_map(second_file, _version=st.session_state.get('second_version',0))
		except Exception as e:
			st.error(f"对比文件加载失败: {e}")
	# 差异映射: object_id -> status ('added','removed','moved','changed','same')
	obj_diff: Dict[str,str] = {}
	if compare_mode and floors and floors2:
		def index_objs(floors_list: List[Floor]):
			idx={}
			for fl in floors_list:
				for rm in fl.rooms:
					for o in rm.objects:
						idx[o.obj_id]= (o, rm.room_id, fl.floor_id)
			return idx
		idx1 = index_objs(floors)
		idx2 = index_objs(floors2)
		all_ids = set(idx1.keys())|set(idx2.keys())
		for oid in all_ids:
			if oid not in idx1:
				obj_diff[oid]='added'
				continue
			if oid not in idx2:
				obj_diff[oid]='removed'
				# ghost 标记: 将历史对象复制到当前数据集合某个容器 (保持其原楼层/房间) 以便右侧显示半透明
				if ghost_removed:
					(o_hist, r_hist, f_hist) = idx1[oid]
					# 找对应楼层/房间
					for fl2 in floors2:
						if fl2.floor_id == f_hist:
							for rm2 in fl2.rooms:
								if rm2.room_id == r_hist:
									# 克隆一个轻量对象 (不修改原类, 使用 to_dict -> from_dict)
									try:
										ghost_obj = Object.from_dict(o_hist.to_dict())
										ghost_obj.obj_id = o_hist.obj_id  # 保持 id
										ghost_obj.label = f"(del){o_hist.label}"
										if not hasattr(ghost_obj, 'stability') or ghost_obj.stability is None:
											ghost_obj.stability = 0.0
										# 附加一个标记字段
										ghost_obj._deleted = True  # type: ignore
										rm2.objects.append(ghost_obj)
									except Exception:
										pass
				break
				continue
			(o1,r1,f1)=idx1[oid]; (o2,r2,f2)=idx2[oid]
			moved = (r1!=r2) or (f1!=f2)
			changed=False
			c1 = polygon_centroid(o1.region or [])
			c2 = polygon_centroid(o2.region or [])
			if c1 and c2:
				dx=abs(c1[0]-c2[0]); dy=abs(c1[1]-c2[1])
				if dx>move_thresh or dy>move_thresh:
					changed=True
			if moved:
				obj_diff[oid]='moved'
			elif changed:
				obj_diff[oid]='changed'
			else:
				obj_diff[oid]='same'
	if not floors:
		st.warning("尚未加载主文件。请在侧栏选择并点击 '加载主文件'。"); return

	# 侧栏层级选择 UI (仅在组合视图/单楼层增强时显示)
	selected_floors=set(); selected_rooms=set(); selected_objects=set()
	with st.sidebar:
		for f in floors:
			f_checked = st.checkbox(f"楼层 {f.floor_id}", key=f"chk_floor_{f.floor_id}")
			if f_checked:
				selected_floors.add(f.floor_id)
			with st.expander(f"房间 - {f.floor_id}"):
				for r in f.rooms:
					room_name_txt = _room_name_text(r)
					r_checked = st.checkbox(f"{r.room_id} ({room_name_txt})", key=f"chk_room_{r.room_id}")
					if r_checked:
						selected_rooms.add(r.room_id)
						objs = r.objects
						if objs:
							obj_ids = [o.obj_id for o in objs]
							picked = st.multiselect(
								f"对象 {r.room_id}", obj_ids, default=[], key=f"ms_{r.room_id}"
							)
							for oid in picked:
								selected_objects.add(oid)

	# 若未选择任何层级, 默认选中全部楼层便于展示
	if not (selected_floors or selected_rooms or selected_objects):
		selected_floors = {f.floor_id for f in floors}

	# 汇总统计（基于被纳入的楼层集合）
	used_floors = [f for f in floors if f.floor_id in selected_floors]
	all_rooms = [r for f in used_floors for r in f.rooms if (f.floor_id in selected_floors) or (r.room_id in selected_rooms)]
	# 若只是选对象，则也纳入其房间/楼层统计
	if selected_objects:
		for f in floors:
			for r in f.rooms:
				if any(o.obj_id in selected_objects for o in r.objects):
					if r not in all_rooms:
						all_rooms.append(r)
					if f not in used_floors:
						used_floors.append(f)
	all_objects = []
	for r in all_rooms:
		for o in r.objects:
			if (r.room_id in selected_rooms) or (r.floor_id in selected_floors if hasattr(r, 'floor_id') else False) or (o.obj_id in selected_objects):
				all_objects.append(o)
	# 如果只选对象, 仍把那些对象加进去
	for f in floors:
		for r in f.rooms:
			for o in r.objects:
				if o.obj_id in selected_objects and o not in all_objects:
					all_objects.append(o)

	col1, col2, col3, col4 = st.columns(4)
	col1.metric("Floors", len(used_floors))
	col2.metric("Rooms", len({r.room_id for r in all_rooms}))
	col3.metric("Objects", len({o.obj_id for o in all_objects}))
	col4.metric("文件", Path(data_file).name)

	if view_mode == "3D 平面视图":
		# 平面视图附加参数滑杆
		with st.sidebar.expander("平面视图参数", expanded=False):
			floor_gap = st.slider("楼层垂直间距", 1.0, 20.0, 6.0, 0.5, key='flat_floor_gap')
			object_plane_ratio = st.slider("对象平面位置(0底-1顶)", 0.0, 1.0, 0.5, 0.01, key='flat_obj_plane_ratio')
			obj_size = st.slider("对象节点大小", 4, 40, obj_size, 1, key='flat_obj_size')
			show_doors = st.checkbox("显示门 (door_positions)", value=True, key='flat_show_doors')
		if compare_mode and floors2:
			st.subheader("并排对比: 左=历史 / 右=当前")
			c1, c2 = st.columns(2)
			if 'cam_left' not in st.session_state: st.session_state.cam_left=None
			if 'cam_right' not in st.session_state: st.session_state.cam_right=None
			with c1:
				fig1 = build_multi_selection_figure(
					floors,
					selected_floors,
					selected_rooms,
					selected_objects,
					show_labels,
					min_rcfd,
					obj_size,
					show_doors=show_doors,
					floor_gap=floor_gap,
					object_plane_ratio=object_plane_ratio,
					obj_diff=obj_diff,
				)
				if camera_sync:
					if 'auto_cam' not in st.session_state: st.session_state.auto_cam=False
					st.session_state.auto_cam = st.checkbox('自动镜像相机 (左驱动→右)', value=st.session_state.auto_cam, key='auto_cam_left')
					res_left = plotly_events(fig1, override_height=700, key='hist_plot_events', select_event=False)
					st.session_state.cam_left = fig1.layout.scene.camera
				else:
					st.plotly_chart(fig1, use_container_width=True, key='hist_plot')
			with c2:
				fig2 = build_multi_selection_figure(
					floors2,
					selected_floors,
					selected_rooms,
					selected_objects,
					show_labels,
					min_rcfd,
					obj_size,
					show_doors=show_doors,
					floor_gap=floor_gap,
					object_plane_ratio=object_plane_ratio,
					obj_diff=obj_diff,
				)
				if camera_sync:
					if st.session_state.get('cam_left') and st.session_state.get('auto_cam'):
						fig2.update_layout(scene=dict(camera=st.session_state.cam_left))
					plotly_events(fig2, override_height=700, key='now_plot_events', select_event=False)
					st.session_state.cam_right = fig2.layout.scene.camera
				else:
					st.plotly_chart(fig2, use_container_width=True, key='now_plot')
			if camera_sync and st.session_state.cam_right and st.button("同步相机到历史(右→左)", key='sync_cam_rl'):
				fig1.update_layout(scene=dict(camera=st.session_state.cam_right))
		else:
			fig = build_multi_selection_figure(
				floors,
				selected_floors,
				selected_rooms,
				selected_objects,
				show_labels,
				min_rcfd,
				obj_size,
				show_doors=show_doors,
				floor_gap=floor_gap,
				object_plane_ratio=object_plane_ratio,
			)
			st.plotly_chart(fig, use_container_width=True)
	else:  # 3D 层级视图 (多楼层)
		show_regions = st.sidebar.checkbox("标记区分区域", value=False)
		region_shape = 'rect'
		region_opacity = 0.15
		region_types_choice = {'房间区域': True, '关系簇区域': True}
		if show_regions:
			with st.sidebar.expander("区域显示设置", expanded=False):
				region_shape = st.selectbox("区域形状", ["矩形", "凸包"], index=0)
				region_opacity = st.slider("区域不透明度", 0.05, 0.5, 0.15, 0.01)
				room_regions_on = st.checkbox("显示房间区域", value=True, key='show_room_regions')
				cluster_regions_on = st.checkbox("显示关系簇区域", value=True, key='show_cluster_regions')
				region_no_overlap = st.checkbox("启用防重叠微调 (可能引入偏移)", value=False, key='region_no_overlap')
				region_types_choice['房间区域'] = room_regions_on
				region_types_choice['关系簇区域'] = cluster_regions_on
		custom_gap = st.sidebar.checkbox("自定义垂直间距", value=False)
		level_params=None
		if custom_gap:
			with st.sidebar.expander("层级间距调节", expanded=True):
				building_offset_val = st.slider("Building 偏移", 0.0, 12.0, 2.0, 0.1)
				floor_room_gap_val = st.slider("Floor→Room 间距", 0.5, 12.0, 2.0, 0.1)
				room_object_gap_val = st.slider("Room→Object 间距", 0.5, 12.0, 2.5, 0.1)
				level_params = {
					'building_offset': building_offset_val,
					'floor_room_gap': floor_room_gap_val,
					'room_object_gap': room_object_gap_val,
				}
		# 力倍率设置
		force_params=None
		with st.sidebar.expander("力模型倍率", expanded=False):
			st.caption("对数尺度 0.05~100 (滑杆指数映射) + 参数保存 / 动画")
			# 初始化默认 raw slider 值
			if 'force_defaults_raw' not in st.session_state:
				st.session_state.force_defaults_raw = {
					'repulsion_raw': 500,
					'parent_pull_raw': 500,
					'room_attr_raw': 500,
					'cluster_attr_raw': 500,
					'cluster_rep_raw': 500,
					'global_grav_raw': 500,
					'vertical_gap_raw': 500,
					'edge_width': 6,
					'color_mode': 'level'
				}
			if 'force_saved_raw' not in st.session_state:
				st.session_state.force_saved_raw = None
			colA,colB,colC = st.columns(3)
			if colA.button("保存参数"):
				st.session_state.force_saved_raw = {k: st.session_state.get(k,v) for k,v in st.session_state.force_defaults_raw.items()}
			if colB.button("恢复保存") and st.session_state.force_saved_raw:
				for k,v in st.session_state.force_saved_raw.items():
					st.session_state[k]=v
			if colC.button("恢复默认"):
				for k,v in st.session_state.force_defaults_raw.items():
					st.session_state[k]=v
			def log_slider_raw(label, key):
				raw = st.slider(label, 0, 1000, st.session_state.get(key, 500), key=key)
				val = 0.05 * (200 ** (raw/1000))
				st.caption(f"{label}: {val:.3f}")
				return val
			repulsion_scale = log_slider_raw("同层斥力", 'repulsion_raw')
			parent_pull_scale = log_slider_raw("父级被子级拉力", 'parent_pull_raw')
			room_attr_scale = log_slider_raw("房间内对象聚合", 'room_attr_raw')
			cluster_attr_scale = log_slider_raw("关系簇聚合", 'cluster_attr_raw')
			cluster_rep_scale = log_slider_raw("关系簇间斥力", 'cluster_rep_raw')
			global_grav_scale = log_slider_raw("全局重力", 'global_grav_raw')
			vertical_gap_scale = log_slider_raw("垂直间距整体倍率", 'vertical_gap_raw')
			edge_width = st.slider("层级连线粗细", 1, 30, st.session_state.get('edge_width',6), key='edge_width')
			color_mode = st.selectbox("节点配色", ["level","stability"], index=0, key='color_mode')
			st.markdown('---')
			st.caption("动画配置")
			anim_col1, anim_col2 = st.columns(2)
			with anim_col1:
				enable_animation = st.checkbox("启用动画", value=False, key='anim_enable')
				animation_frames = st.slider("帧数", 5, 80, 30, 1, key='anim_frames')
			with anim_col2:
				snapshot_interval = st.slider("抓帧间隔", 1, 30, 5, 1, key='anim_interval')
				cooling = st.slider("冷却系数", 0.85, 1.0, 0.98, 0.01, key='anim_cooling')
			force_params = {
				'repulsion_scale': repulsion_scale,
				'parent_pull_scale': parent_pull_scale,
				'room_attr_scale': room_attr_scale,
				'cluster_attr_scale': cluster_attr_scale,
				'cluster_rep_scale': cluster_rep_scale,
				'global_grav_scale': global_grav_scale,
				'vertical_gap_scale': vertical_gap_scale,
				'edge_width': edge_width,
				'color_mode': color_mode,
				'enable_animation': st.session_state.get('anim_enable', False),
				'animation_frames': st.session_state.get('anim_frames', 0),
				'cooling': st.session_state.get('anim_cooling', 1.0),
				'snapshot_interval': st.session_state.get('anim_interval', 10),
			}
		# 将区域显示选择编码进 force_params 传入 (使用自定义键)
		if compare_mode and floors2:
			c1,c2 = st.columns(2)
			with c1:
				fig_h1 = build_multi_hierarchy_3d_figure(
					floors,
					selected_floors,
					selected_rooms,
					selected_objects,
					show_labels,
					obj_size,
					min_rcfd,
					show_regions and any(region_types_choice.values()),
					level_params,
					{**(force_params or {}), 'region_shape': region_shape, 'region_opacity': region_opacity, 'show_room_regions': region_types_choice['房间区域'], 'show_cluster_regions': region_types_choice['关系簇区域'], 'region_no_overlap': st.session_state.get('region_no_overlap', False)},
					edge_width=force_params.get('edge_width',6) if force_params else 6,
					color_mode=force_params.get('color_mode','level'),
					enable_animation=False,
					animation_frames=0,
					cooling=1.0,
					snapshot_interval=10,
					obj_diff=None,
				)
				if camera_sync:
					if 'auto_cam_hier' not in st.session_state: st.session_state.auto_cam_hier=False
					st.session_state.auto_cam_hier = st.checkbox('自动镜像相机 (层级 左→右)', value=st.session_state.auto_cam_hier, key='auto_cam_hier_left')
					plotly_events(fig_h1, override_height=800, key='hier_hist_events', select_event=False)
					st.session_state.cam_left = fig_h1.layout.scene.camera
				st.plotly_chart(fig_h1, use_container_width=True, key='hier_hist')
			with c2:
				fig_h2 = build_multi_hierarchy_3d_figure(
					floors2,
					selected_floors,
					selected_rooms,
					selected_objects,
					show_labels,
					obj_size,
					min_rcfd,
					show_regions and any(region_types_choice.values()),
					level_params,
					{**(force_params or {}), 'region_shape': region_shape, 'region_opacity': region_opacity, 'show_room_regions': region_types_choice['房间区域'], 'show_cluster_regions': region_types_choice['关系簇区域'], 'region_no_overlap': st.session_state.get('region_no_overlap', False)},
					edge_width=force_params.get('edge_width',6) if force_params else 6,
					color_mode=force_params.get('color_mode','level'),
					enable_animation=False,
					animation_frames=0,
					cooling=1.0,
					snapshot_interval=10,
					obj_diff=obj_diff,
				)
				if camera_sync and st.session_state.get('cam_left') and st.session_state.get('auto_cam_hier'):
					fig_h2.update_layout(scene=dict(camera=st.session_state.cam_left))
				if camera_sync:
					plotly_events(fig_h2, override_height=800, key='hier_now_events', select_event=False)
					st.session_state.cam_right = fig_h2.layout.scene.camera
				st.plotly_chart(fig_h2, use_container_width=True, key='hier_now')
			if camera_sync and st.session_state.cam_right and st.button('同步相机(右→左)', key='sync_hier_rl'):
				fig_h1.update_layout(scene=dict(camera=st.session_state.cam_right))
			st.caption("对象差异颜色: 绿色新增 / 橙色移动 / 紫色变化 / 红色删除")
		else:
			fig = build_multi_hierarchy_3d_figure(
				floors,
				selected_floors,
				selected_rooms,
				selected_objects,
				show_labels,
				obj_size,
				min_rcfd,
				show_regions and any(region_types_choice.values()),
				level_params,
				{**(force_params or {}), 'region_shape': region_shape, 'region_opacity': region_opacity, 'show_room_regions': region_types_choice['房间区域'], 'show_cluster_regions': region_types_choice['关系簇区域'], 'region_no_overlap': st.session_state.get('region_no_overlap', False)},
				edge_width=force_params.get('edge_width',6) if force_params else 6,
				color_mode=force_params.get('color_mode','level'),
				enable_animation=force_params.get('enable_animation', False),
				animation_frames=force_params.get('animation_frames',0),
				cooling=force_params.get('cooling',1.0),
				snapshot_interval=force_params.get('snapshot_interval',10),
				obj_diff=obj_diff,
			)
			st.plotly_chart(fig, use_container_width=True)

	with st.expander("选择摘要"):
		st.write({
			"floors": list(selected_floors),
			"rooms": list(selected_rooms),
			"objects": list(selected_objects)
		})


if __name__ == "__main__":
	main()
