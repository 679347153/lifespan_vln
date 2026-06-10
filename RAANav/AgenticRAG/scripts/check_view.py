"""诊断: 单视角和360度视角各能看到多少物体, 分别来自几个房间."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from semantic_map_Create.scene_extract import make_simulator

basis_glb = '/home/adminer/agentRAG/experiment_data/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb'
dataset_config = '/home/adminer/agentRAG/experiment_data/hm3d/hm3d_annotated_basis.scene_dataset_config.json'
sim = make_simulator(basis_glb, dataset_config, enable_semantic=True)

agent = sim.get_agent(0)
state = agent.get_state()
pos = state.position
print(f"Agent 位置: {[round(float(x),2) for x in pos]}")

STRUCTURAL = frozenset(['wall','floor','ceiling','void','unknown','misc',
    'door','window','column','beam','railing','stair','stairs','banister','balustrade'])

def analyze_obs(semantic_frames, label_prefix):
    """分析多帧语义图的合并结果."""
    pixel_counter = {}
    for sem in semantic_frames:
        uids, cnts = np.unique(sem, return_counts=True)
        for sid, cnt in zip(uids, cnts):
            sid = int(sid)
            pixel_counter[sid] = pixel_counter.get(sid, 0) + int(cnt)

    results = []
    for sid, cnt in pixel_counter.items():
        if sid == 0:
            continue
        if sid >= len(sim.semantic_scene.objects):
            continue
        obj = sim.semantic_scene.objects[sid]
        if obj is None:
            continue
        cat = obj.category.name('').strip().lower()
        if cat in STRUCTURAL or not cat:
            continue
        if cnt < 100:
            continue
        region = obj.region
        rid = region.id if region else -1
        results.append((cat, cnt, rid))

    results.sort(key=lambda x: -x[1])
    unique_labels = set(x[0] for x in results)
    rooms = set(x[2] for x in results)

    print(f"\n=== {label_prefix} ===")
    print(f"  可见非结构物体 (>100px): {len(results)} 个")
    print(f"  唯一标签: {len(unique_labels)}")
    print(f"  涉及房间数: {len(rooms)}, 房间IDs: {sorted(rooms)}")
    print(f"  物体详情:")
    for cat, px, rid in results:
        print(f"    {cat:30s} px={px:6d}  room={rid}")
    return results, rooms

# --- 1. 单视角 ---
obs = sim.get_sensor_observations()
sem_single = [obs['semantic']]
r1, rooms1 = analyze_obs(sem_single, "单视角 (正前方, 90 FOV)")

# --- 2. 360度环视 (4x90度) ---
import quaternion as qt
original_rot = state.rotation
heading_start = 0
sem_frames_360 = []
for i in range(4):
    heading = heading_start + i * 90
    rad = np.deg2rad(heading)
    new_rot = qt.from_euler_angles([0, rad, 0])
    s = agent.get_state()
    s.rotation = new_rot
    agent.set_state(s)
    obs = sim.get_sensor_observations()
    sem_frames_360.append(obs['semantic'])

r2, rooms2 = analyze_obs(sem_frames_360, "360度环视 (4x90 FOV)")

# --- 3. 分房间统计 ---
print("\n=== 房间隔离分析 ===")
for rid in sorted(rooms2):
    objs_in_room = [(c,p,r) for c,p,r in r2 if r == rid]
    print(f"  房间 {rid}: {len(objs_in_room)} 物体, 标签: {[x[0] for x in objs_in_room]}")

sim.close()
