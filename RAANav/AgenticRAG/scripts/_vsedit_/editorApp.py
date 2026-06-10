import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import streamlit as st
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go
import time

# 复用已有数据结构
from semantic_map.floor import Floor
from semantic_map.room import Room
from semantic_map.object import Object

try:
    import visualizeApp  # type: ignore
except Exception:
    visualizeApp = None

st.set_page_config(page_title="语义地图 2D 编辑器", layout="wide")

# ---------------- 序列化 & 状态快照 (撤销/重做) ---------------- #

def serialize_floors(floors: List[Floor]) -> str:
    return json.dumps([f.to_dict() for f in floors], ensure_ascii=False)

def deserialize_floors(data: str) -> List[Floor]:
    arr = json.loads(data)
    return [Floor.from_dict(d) for d in arr]

def push_undo():
    if not st.session_state.floors_edit: return
    snapshot = serialize_floors(st.session_state.floors_edit)
    if not st.session_state.undo_stack or st.session_state.undo_stack[-1] != snapshot:
        st.session_state.undo_stack.append(snapshot)
    # 限制长度
    if len(st.session_state.undo_stack) > 40:
        st.session_state.undo_stack.pop(0)

def apply_state(snapshot: str):
    st.session_state.floors_edit = deserialize_floors(snapshot)

# ---------------- Data IO ---------------- #

def load_map(path: str) -> List[Floor]:
    p = Path(path)
    with p.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON 顶层需为 list[Floor]")
    return [Floor.from_dict(d) for d in data]

def save_map(floors: List[Floor], path: str):
    out = [f.to_dict() for f in floors]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

# ---------------- Helper ---------------- #

def find_room(floors: List[Floor], floor_id: str, room_id: str) -> Optional[Room]:
    for f in floors:
        if f.floor_id == floor_id:
            for r in f.rooms:
                if r.room_id == room_id:
                    return r
    return None

def find_object(floors: List[Floor], obj_id: str) -> Optional[Tuple[Floor, Room, Object]]:
    for f in floors:
        for r in f.rooms:
            for o in r.objects:
                if o.obj_id == obj_id:
                    return f, r, o
    return None

def generate_object_id(room: Room, label: str, existing: set) -> str:
    seq = 1
    base = f"{label}".replace(' ', '_')
    while True:
        oid = f"{base}_{seq}_{room.room_id}"
        if oid not in existing:
            return oid
        seq += 1

def polygon_centroid(pts: List[Dict]) -> Tuple[float, float]:
    if not pts: return 0.0, 0.0
    x = sum(p['x'] for p in pts) / len(pts)
    y = sum(p['y'] for p in pts) / len(pts)
    return x, y

def translate_polygon(pts: List[Dict], dx: float, dy: float) -> List[Dict]:
    return [{"x": p['x'] + dx, "y": p['y'] + dy} for p in pts]

def rect_polygon(cx: float, cy: float, w: float, h: float) -> List[Dict]:
    hw = w / 2.0; hh = h / 2.0
    return [
        {"x": cx - hw, "y": cy - hh},
        {"x": cx + hw, "y": cy - hh},
        {"x": cx + hw, "y": cy + hh},
        {"x": cx - hw, "y": cy + hh},
    ]

# ---------------- Session State 初始化 ---------------- #
for key, default in [
    ('floors_edit', []),
    ('loaded_file', ''),
    ('poly_points', []),
    ('selected_obj_id', None),
    ('new_relations', []),
    ('undo_stack', []),
    ('redo_stack', []),
    ('batch_selected', set()),
    ('view_range', None),  # (xmin,xmax,ymin,ymax)
    ('grid_snap', True),
    ('grid_size', 0.25),
    ('edit_mode', True),          # True=编辑, False=只读
    ('view_scope', 'room'),       # 'room' or 'floor'
    ('keep_aspect', True),
    ('color_map', {}),            # label -> color
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("🛠️ 语义地图 2D 编辑器")

# ---------------- Sidebar: IO & 控制 ---------------- #
with st.sidebar:
    st.header("加载 / 保存")
    data_file = st.text_input("原始 JSON 路径", value=st.session_state.loaded_file or 'RAG_Graph/map_now.json')
    load_col, save_col = st.columns(2)
    with load_col:
        if st.button("加载", type='primary'):
            try:
                floors = load_map(data_file)
                st.session_state.floors_edit = floors
                st.session_state.loaded_file = data_file
                st.session_state.undo_stack = []
                st.session_state.redo_stack = []
                push_undo()  # 初始快照
                st.success("加载成功")
            except Exception as e:
                st.error(f"加载失败: {e}")
    export_name = st.text_input("导出文件名", value=f"RAG_Graph/map_edit_{int(time.time())}.json")
    with save_col:
        if st.button("另存为", type='secondary'):
            if st.session_state.floors_edit:
                save_map(st.session_state.floors_edit, export_name)
                st.success(f"已保存为 {export_name}")

    # 撤销 / 重做
    ucol, rcol = st.columns(2)
    with ucol:
        if st.button("撤销 (Undo)") and len(st.session_state.undo_stack) > 1:
            cur = st.session_state.undo_stack.pop()  # 当前
            st.session_state.redo_stack.append(cur)
            prev = st.session_state.undo_stack[-1]
            apply_state(prev)
            st.experimental_rerun()
    with rcol:
        if st.button("重做 (Redo)") and st.session_state.redo_stack:
            snap = st.session_state.redo_stack.pop()
            st.session_state.undo_stack.append(snap)
            apply_state(snap)
            st.experimental_rerun()

    if st.session_state.floors_edit:
        st.markdown("---")
        st.subheader("楼层 / 房间 选择")
        floor_ids = [f.floor_id for f in st.session_state.floors_edit]
        floor_id = st.selectbox("楼层", floor_ids)
        # 视图范围: 房间或整层
        view_scope = st.radio("查看范围", ["room", "floor"], index=(0 if st.session_state.view_scope=='room' else 1), horizontal=True, key='view_scope')
        rooms_cur = next((f.rooms for f in st.session_state.floors_edit if f.floor_id == floor_id), [])
        room_ids = [r.room_id for r in rooms_cur]
        if view_scope == 'room':
            room_id = st.selectbox("房间", room_ids)
        else:
            # 在整层模式下仍允许选择一个房间作为“当前房间”用于新增对象
            room_id = st.selectbox("当前房间(用于添加对象)", ["(无)"] + room_ids)
            if room_id == "(无)":
                room_id = room_ids[0] if room_ids else None

        # 编辑 / 只读 模式
        st.markdown("---")
        st.subheader("模式")
        edit_mode = st.radio("交互模式", ["编辑", "只读"], index=(0 if st.session_state.edit_mode else 1), horizontal=True)
        st.session_state.edit_mode = (edit_mode == '编辑')
        st.checkbox("保持等比例", value=st.session_state.keep_aspect, key='keep_aspect')

        st.markdown("---")
        st.subheader("对象选择 / 批量")
        cur_room = find_room(st.session_state.floors_edit, floor_id, room_id) if room_id else None
        obj_ids = [o.obj_id for o in (cur_room.objects if cur_room else [])]
        selected_obj = st.selectbox("单对象 (编辑)", ["(无)"] + obj_ids)
        st.session_state.selected_obj_id = None if selected_obj == "(无)" else selected_obj
        batch_multi = st.multiselect("批量选择对象", obj_ids, default=list(st.session_state.batch_selected))
        st.session_state.batch_selected = set(batch_multi)
        colbt1, colbt2 = st.columns(2)
        with colbt1:
            bdx = st.number_input("批量平移 dx", value=0.0, step=0.1)
        with colbt2:
            bdy = st.number_input("批量平移 dy", value=0.0, step=0.1)
        if st.session_state.edit_mode and st.button("应用批量平移") and cur_room and st.session_state.batch_selected:
            push_undo()
            for o in cur_room.objects:
                if o.obj_id in st.session_state.batch_selected:
                    o.region = translate_polygon(o.region or [], bdx, bdy)
            st.success("批量平移完成")

        st.markdown("---")
        st.subheader("对象编辑 (单) / 关系")
        if st.session_state.selected_obj_id:
            triple = find_object(st.session_state.floors_edit, st.session_state.selected_obj_id)
            if triple:
                _, r_sel, o_sel = triple
                st.write(f"Label: {o_sel.label}")
                cx, cy = polygon_centroid(o_sel.region or [])
                st.write(f"质心: ({cx:.3f}, {cy:.3f})")
                if st.session_state.edit_mode:
                    dx = st.number_input("平移 dx", value=0.0, step=0.1, key='single_dx')
                    dy = st.number_input("平移 dy", value=0.0, step=0.1, key='single_dy')
                    if st.button("平移对象"):
                        push_undo()
                        snap = st.session_state.grid_snap
                        g = st.session_state.grid_size
                        ddx, ddy = (round(dx/g)*g if snap else dx, round(dy/g)*g if snap else dy)
                        o_sel.region = translate_polygon(o_sel.region or [], ddx, ddy)
                        st.success("已平移")
                    coldc, colcp = st.columns(2)
                    with coldc:
                        if st.button("删除对象", key='del_obj_btn'):
                            push_undo()
                            r_sel.objects = [oo for oo in r_sel.objects if oo.obj_id != o_sel.obj_id]
                            st.session_state.selected_obj_id = None
                            st.success("已删除")
                            st.experimental_rerun()
                    with colcp:
                        if st.button("复制对象", key='copy_obj_btn'):
                            push_undo()
                            existing = set(o.obj_id for o in r_sel.objects)
                            base_label = o_sel.label + "_copy"
                            new_id = generate_object_id(r_sel, base_label, existing)
                            new_region = translate_polygon(o_sel.region or [], 0.2, 0.2)
                            dup = Object.from_dict({
                                "obj_id": new_id,
                                "label": base_label,
                                "region": new_region,
                                "R_objs": {},
                                "N": o_sel.N,
                                "description": f"copy of {o_sel.obj_id}",
                                "stability": o_sel.stability
                            })
                            r_sel.objects.append(dup)
                            st.session_state.selected_obj_id = new_id
                            st.success(f"已复制为 {new_id}")
                st.markdown("关系 R_objs")
                if o_sel.R_objs:
                    for tgt, meta in list(o_sel.R_objs.items()):
                        cols = st.columns([4,1])
                        with cols[0]:
                            st.write(f"→ {tgt} | Nt={meta.get('Nt')} Nr={meta.get('Nr')} Rcfd={meta.get('Rcfd')}")
                        with cols[1]:
                            if st.button("删", key=f"del_rel_{tgt}"):
                                push_undo()
                                del o_sel.R_objs[tgt]
                                st.experimental_rerun()
                tgt_choice = st.selectbox("目标对象", [oid for oid in obj_ids if oid != o_sel.obj_id])
                col_nt, col_nr, col_rcfd = st.columns(3)
                with col_nt: nt_val = st.number_input("Nt", value=1.0, step=1.0)
                with col_nr: nr_val = st.number_input("Nr", value=1.0, step=1.0)
                with col_rcfd: rcfd_val = st.number_input("Rcfd", value=0.5, step=0.01, format='%.2f')
                if st.button("添加/更新关系"):
                    push_undo()
                    if not o_sel.R_objs: o_sel.R_objs = {}
                    o_sel.R_objs[tgt_choice] = {"Nt": nt_val, "Nr": nr_val, "Rcfd": rcfd_val}
                    st.success("关系已更新")

        st.markdown("---")
        st.subheader("新增对象 / 新房间")
        if st.session_state.edit_mode:
            new_label = st.text_input("对象标签", value="custom")
            shape_mode = st.radio("形状模式", ["矩形", "多边形(点击绘制)"], key='shape_mode_radio')
            st.checkbox("启用网格吸附", value=st.session_state.grid_snap, key='grid_snap')
            st.number_input("网格大小", value=st.session_state.grid_size, step=0.05, key='grid_size')
            if shape_mode == "矩形":
                col1,col2 = st.columns(2)
                with col1:
                    cx = st.number_input("中心 x", value=0.0, step=0.1)
                    w = st.number_input("宽 w", value=1.0, step=0.1)
                with col2:
                    cy = st.number_input("中心 y", value=0.0, step=0.1)
                    h = st.number_input("高 h", value=1.0, step=0.1)
                if st.button("添加矩形对象") and cur_room:
                    push_undo()
                    snap = st.session_state.grid_snap; g = st.session_state.grid_size
                    if snap:
                        cx_s = round(cx/g)*g; cy_s = round(cy/g)*g; w_s = round(w/g)*g; h_s = round(h/g)*g
                    else:
                        cx_s,cy_s,w_s,h_s = cx,cy,w,h
                    existing = set(o.obj_id for o in cur_room.objects)
                    oid = generate_object_id(cur_room, new_label, existing)
                    poly = rect_polygon(cx_s, cy_s, w_s, h_s)
                    new_obj = Object.from_dict({
                        "obj_id": oid,
                        "label": new_label,
                        "region": poly,
                        "R_objs": {},
                        "N": 1,
                        "description": f"added {new_label}",
                        "stability": 1.0
                    })
                    cur_room.objects.append(new_obj)
                    st.success(f"已添加 {oid}")
            else:
                st.info("在右侧 2D 视图点击添加点 (自动吸附)，完成后点“完成多边形”")
                cpoly1, cpoly2 = st.columns(2)
                with cpoly1:
                    if st.button("重置点"):
                        st.session_state.poly_points = []
                with cpoly2:
                    if st.button("完成多边形") and cur_room and len(st.session_state.poly_points) >= 3:
                        push_undo()
                        existing = set(o.obj_id for o in cur_room.objects)
                        oid = generate_object_id(cur_room, new_label, existing)
                        new_obj = Object.from_dict({
                            "obj_id": oid,
                            "label": new_label,
                            "region": st.session_state.poly_points.copy(),
                            "R_objs": {},
                            "N": 1,
                            "description": f"added {new_label}",
                            "stability": 1.0
                        })
                        cur_room.objects.append(new_obj)
                        st.session_state.poly_points = []
                        st.success(f"已添加 {oid}")
            # 新增房间 (仅整层视图)
            if view_scope == 'floor':
                st.markdown("**新增房间**")
                new_room_id = st.text_input("房间ID", value=f"{floor_id}_R{len(rooms_cur)+1}")
                rw_col1, rw_col2 = st.columns(2)
                with rw_col1:
                    rw = st.number_input("房间宽", value=4.0, step=0.5)
                with rw_col2:
                    rh = st.number_input("房间高", value=4.0, step=0.5)
                if st.button("添加房间"):
                    if any(r.room_id == new_room_id for r in rooms_cur):
                        st.error("房间ID已存在")
                    else:
                        # 计算整层当前包围盒, 把新房间放到最右侧
                        all_x=[]; all_y=[]
                        for r in rooms_cur:
                            if r.region:
                                all_x.extend([p['x'] for p in r.region])
                                all_y.extend([p['y'] for p in r.region])
                        if all_x and all_y:
                            xmin,xmax = min(all_x), max(all_x)
                            ymin,ymax = min(all_y), max(all_y)
                        else:
                            xmin=ymin=0.0; xmax=ymax=0.0
                        pad = 1.0
                        cx_new = xmax + pad + rw/2
                        cy_new = (ymin + ymax)/2 if all_y else 0.0
                        region_new = rect_polygon(cx_new, cy_new, rw, rh)
                        new_room = Room.from_dict({
                            "room_id": new_room_id,
                            "label": new_room_id,
                            "description": f"added room {new_room_id}",
                            "region": region_new,
                            "objects": [],
                        })
                        push_undo()
                        for f in st.session_state.floors_edit:
                            if f.floor_id == floor_id:
                                f.rooms.append(new_room)
                                break
                        st.success(f"已添加房间 {new_room_id}")

        st.markdown("---")
        st.subheader("视图范围 / 关系线")
        show_rel = st.checkbox("显示 R_objs 关系线", value=True)
        lock_range = st.checkbox("锁定当前计算范围", value=False, help="锁定后缩放不会自动更新 (只读模式可自由缩放)")
        if st.session_state.view_range:
            vr = st.session_state.view_range
            st.caption(f"当前锁定范围: x[{vr[0]:.2f},{vr[1]:.2f}] y[{vr[2]:.2f},{vr[3]:.2f}]")
        if st.button("清除锁定范围"):
            st.session_state.view_range = None

        # 将关键参数写回共享 (供主区使用)
        st.session_state._editor_ctx = {
            'floor_id': floor_id,
            'room_id': room_id,
            'view_scope': view_scope,
            'shape_mode': st.session_state.get('shape_mode_radio','矩形'),
            'show_rel': show_rel,
            'lock_range': lock_range,
            'edit_mode': st.session_state.edit_mode,
        }

# ---------------- 主区 2D 编辑画布 ---------------- #

floors = st.session_state.floors_edit
if floors and '_editor_ctx' in st.session_state:
    ctx = st.session_state._editor_ctx
    floor_id = ctx['floor_id']; room_id = ctx['room_id']
    view_scope = ctx.get('view_scope','room')
    edit_mode = ctx.get('edit_mode', True)
    # 颜色分配函数
    PALETTE = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf','#393b79','#637939','#8c6d31','#843c39','#7b4173']
    def color_for_label(label:str):
        cm = st.session_state.color_map
        if label not in cm:
            cm[label] = PALETTE[len(cm)%len(PALETTE)]
        return cm[label]

    fig2d = go.Figure()
    centroids = {}
    rooms_drawn = []
    if view_scope == 'room':
        room = find_room(floors, floor_id, room_id)
        if room:
            pts = room.region or []
            if pts:
                xs = [p['x'] for p in pts] + [pts[0]['x']]
                ys = [p['y'] for p in pts] + [pts[0]['y']]
                fig2d.add_trace(go.Scatter(x=xs, y=ys, fill='toself', mode='lines', name=f"Room {room.room_id}", line=dict(color='#1f77b4')))
            for o in room.objects:
                o_pts = o.region or []
                if not o_pts: continue
                ox = [p['x'] for p in o_pts] + [o_pts[0]['x']]
                oy = [p['y'] for p in o_pts] + [o_pts[0]['y']]
                col = color_for_label(o.label)
                fig2d.add_trace(go.Scatter(x=ox, y=oy, mode='lines', line=dict(width=1, color=col), name=o.obj_id, hovertext=o.label, showlegend=False))
                cx, cy = polygon_centroid(o_pts)
                centroids[o.obj_id] = (cx,cy)
                # 仅 marker, 不显示文字
                marker_color = ('#d62728' if o.obj_id == st.session_state.selected_obj_id else ('#9467bd' if o.obj_id in st.session_state.batch_selected else col))
                fig2d.add_trace(go.Scatter(x=[cx], y=[cy], mode='markers', marker=dict(size=6, color=marker_color), name=f"c_{o.obj_id}", hovertext=o.label, showlegend=False))
            rooms_drawn = [room]
        else:
            st.info("未找到房间")
    else:  # floor composite
        floor_obj = next((f for f in floors if f.floor_id == floor_id), None)
        if floor_obj:
            for idx, r in enumerate(floor_obj.rooms):
                pts = r.region or []
                if not pts: continue
                xs = [p['x'] for p in pts] + [pts[0]['x']]
                ys = [p['y'] for p in pts] + [pts[0]['y']]
                fig2d.add_trace(go.Scatter(x=xs, y=ys, fill='toself', mode='lines', name=f"Rm {r.room_id}", line=dict(color='#1f77b4'), opacity=0.25))
                for o in r.objects:
                    o_pts = o.region or []
                    if not o_pts: continue
                    ox = [p['x'] for p in o_pts] + [o_pts[0]['x']]
                    oy = [p['y'] for p in o_pts] + [o_pts[0]['y']]
                    col = color_for_label(o.label)
                    fig2d.add_trace(go.Scatter(x=ox, y=oy, mode='lines', line=dict(width=1, color=col), name=o.obj_id, hovertext=f"{r.room_id}:{o.label}", showlegend=False))
                    cx, cy = polygon_centroid(o_pts)
                    centroids[o.obj_id] = (cx,cy)
                    marker_color = ('#d62728' if o.obj_id == st.session_state.selected_obj_id else ('#9467bd' if o.obj_id in st.session_state.batch_selected else col))
                    fig2d.add_trace(go.Scatter(x=[cx], y=[cy], mode='markers', marker=dict(size=5, color=marker_color), name=f"c_{o.obj_id}", hovertext=f"{o.obj_id}", showlegend=False))
            rooms_drawn = floor_obj.rooms
        else:
            st.info("未找到楼层")
    # 关系线 (仅单房间视图内)
    if ctx['show_rel'] and view_scope=='room' and rooms_drawn:
        room = rooms_drawn[0]
        for o in room.objects:
            if not o.R_objs: continue
            src_c = centroids.get(o.obj_id)
            if not src_c: continue
            for tgt, meta in o.R_objs.items():
                tgt_c = centroids.get(tgt)
                if not tgt_c: continue
                rcfd = float(meta.get('Rcfd',0.0))
                r = int(150 + (255-150)*rcfd)
                g = int(150*(1-rcfd))
                b = int(150*(1-rcfd))
                fig2d.add_trace(go.Scatter(x=[src_c[0], tgt_c[0]], y=[src_c[1], tgt_c[1]], mode='lines', line=dict(color=f'rgb({r},{g},{b})', width=2+3*rcfd), hoverinfo='text', hovertext=f"{o.obj_id}->{tgt} Rcfd={rcfd:.2f}", showlegend=False))
    # 当前绘制多边形
    if edit_mode and ctx['shape_mode'] == '多边形(点击绘制)' and st.session_state.poly_points:
        px = [p['x'] for p in st.session_state.poly_points]
        py = [p['y'] for p in st.session_state.poly_points]
        fig2d.add_trace(go.Scatter(x=px, y=py, mode='markers+lines', line=dict(dash='dot', color='#ff7f0e'), marker=dict(size=6), name='new_poly', showlegend=False))
    # 视图范围 & 自动
    use_lock = ctx['lock_range'] and st.session_state.view_range is not None
    if edit_mode and st.session_state.view_range:
        # 编辑模式：如果有锁定范围始终使用；若未锁定但存储了也使用（保持稳定坐标）
        xr0,xr1,yr0,yr1 = st.session_state.view_range
        fig2d.update_xaxes(range=[xr0,xr1])
        fig2d.update_yaxes(range=[yr0,yr1])
    if (not edit_mode) and use_lock:
        # 只读模式仅在明确锁定时应用锁定范围
        xr0,xr1,yr0,yr1 = st.session_state.view_range
        fig2d.update_xaxes(range=[xr0,xr1])
        fig2d.update_yaxes(range=[yr0,yr1])
    if (edit_mode and st.session_state.view_range is None) or (not edit_mode and (not use_lock)):
        # 需要初始化范围
        all_x=[]; all_y=[]
        for tr in fig2d.data:
            if hasattr(tr,'x') and tr.x is not None:
                all_x.extend(list(tr.x))
            if hasattr(tr,'y') and tr.y is not None:
                all_y.extend(list(tr.y))
        if all_x and all_y:
            xmin,xmax = min(all_x), max(all_x)
            ymin,ymax = min(all_y), max(all_y)
            dx = xmax - xmin; dy = ymax - ymin
            pad = 0.5
            if dx==0: dx=1.0
            if dy==0: dy=1.0
            if st.session_state.keep_aspect:
                side = max(dx, dy)
                cx = (xmin+xmax)/2; cy=(ymin+ymax)/2
                xmin2 = cx - side/2; xmax2 = cx + side/2
                ymin2 = cy - side/2; ymax2 = cy + side/2
                xr = [xmin2-pad, xmax2+pad]; yr = [ymin2-pad, ymax2+pad]
            else:
                xr = [xmin-pad, xmax+pad]; yr = [ymin-pad, ymax+pad]
            fig2d.update_xaxes(range=xr)
            fig2d.update_yaxes(range=yr)
            # 编辑模式下或显式锁定才缓存
            if edit_mode or ctx['lock_range']:
                st.session_state.view_range = (xr[0], xr[1], yr[0], yr[1])
    # 固定/自由缩放
    if edit_mode:
        # 锁定缩放平移
        fig2d.update_xaxes(fixedrange=True)
        fig2d.update_yaxes(fixedrange=True)
    else:
        fig2d.update_xaxes(fixedrange=False)
        fig2d.update_yaxes(fixedrange=False)
    title_extra = "[编辑锁定]" if edit_mode else "[只读自由缩放]"
    fig2d.update_layout(title=((f"房间 {room_id} " if view_scope=='room' else f"楼层 {floor_id} ") + title_extra), xaxis=dict(showgrid=True, scaleanchor=('y' if st.session_state.keep_aspect else None)), yaxis=dict(showgrid=True), legend=dict(orientation='h'))
    # 事件 (只读模式仍允许点击查看信息, 但不写入多边形点)
    click_result = plotly_events(fig2d, click_event=True, hover_event=False, select_event=False, key='edit_plot')
    if click_result and edit_mode and view_scope=='room' and ctx['shape_mode'] == '多边形(点击绘制)':
        pt = click_result[0]
        x_raw = pt.get('x'); y_raw = pt.get('y')
        if x_raw is not None and y_raw is not None:
            g = st.session_state.grid_size; snap = st.session_state.grid_snap
            if snap:
                x_raw = round(x_raw/g)*g; y_raw = round(y_raw/g)*g
            st.session_state.poly_points.append({"x": x_raw, "y": y_raw})
            push_undo()
            st.experimental_rerun()

    # 右侧对象信息面板
    col_fig, col_info = st.columns([4,1])
    with col_fig:
        st.plotly_chart(fig2d, use_container_width=True, config={"scrollZoom": (not edit_mode)})
    with col_info:
        st.markdown("### 对象信息")
        if st.session_state.selected_obj_id:
            triple = find_object(floors, st.session_state.selected_obj_id)
            if triple:
                f_sel, r_sel, o_sel = triple
                cx, cy = polygon_centroid(o_sel.region or [])
                st.write(f"ID: {o_sel.obj_id}")
                st.write(f"Label: {o_sel.label}")
                st.write(f"房间: {r_sel.room_id}")
                st.write(f"质心: ({cx:.2f},{cy:.2f})")
                st.write(f"关系数: {len(o_sel.R_objs or {})}")
                if not edit_mode:
                    st.info("只读模式")
        else:
            st.caption("选择对象以查看信息")

# ---------------- 3D 预览 ---------------- #
if floors and st.sidebar.checkbox('显示简化 3D 预览区', value=False):
    st.subheader("3D 预览 (多楼层组合)")
    if visualizeApp:
        try:
            fig_preview = visualizeApp.build_multi_selection_figure(
                floors,
                selected_floors={f.floor_id for f in floors},
                selected_rooms=set(),
                selected_objects=set(),
                show_labels=True,
                min_rcfd=0.0,
                obj_size=10,
                show_doors=False,
                floor_gap=6.0,
                object_plane_ratio=0.5,
                obj_diff=None,
            )
            st.plotly_chart(fig_preview, use_container_width=True)
        except Exception as e:
            st.error(f"预览失败: {e}")
    else:
        st.info("visualizeApp 未导入")

st.caption("提示: 已支持撤销/重做、批量平移、网格吸附、自定义导出文件名、关系线可视化。2D 视图缩放平移交互后当前版本无法自动捕获新范围(需自定义组件); 可用‘锁定当前计算范围’保持。")
