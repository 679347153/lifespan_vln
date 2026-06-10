import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path('/home/adminer/agentRAG')
out_dir = root / '组会汇报_2026-03-21'
out_dir.mkdir(parents=True, exist_ok=True)

capture_dir = root / 'experiment_data/hm3dsem_walks/val/00824-Dd4bFSTQ8gi'
map_hist = root / 'AgenticRAG/RAG_Graph/map_history.json'
map_merged = root / 'AgenticRAG/RAG_Graph/test_save/map_merged_hovsg_minimal.json'

subs = ['rgb', 'depth', 'semantic', 'pose']
counts = []
for s in subs:
    p = capture_dir / s
    counts.append(len(list(p.glob('*'))) if p.exists() else 0)

plt.figure(figsize=(6, 4))
plt.bar(subs, counts)
plt.title('Habitat采样产物计数 (Scene 00824)')
plt.ylabel('文件数量')
for i, v in enumerate(counts):
    plt.text(i, v + 1, str(v), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(out_dir / '01_采样数据计数.png', dpi=180)
plt.close()


def load_objects(path: Path):
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    objs = []
    for fl in data:
        for rm in fl.get('rooms', []):
            for o in rm.get('objects', []):
                objs.append(o)
    return objs

objs = load_objects(map_hist)
plt.figure(figsize=(7, 6))
for o in objs:
    reg = o.get('region', [])
    if len(reg) < 3:
        continue
    xs = [pt.get('x', 0.0) for pt in reg] + [reg[0].get('x', 0.0)]
    ys = [pt.get('y', 0.0) for pt in reg] + [reg[0].get('y', 0.0)]
    plt.plot(xs, ys, linewidth=1.2)
plt.title('T1 map_history 物体区域(BEV)')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.tight_layout()
plt.savefig(out_dir / '02_T1对象区域可视化.png', dpi=180)
plt.close()

objs_m = load_objects(map_merged)
fields = ['clip_embedding', 'last_update_time', 'exist_prob', 'cooccur_stats', 'R_objs']
present = []
for k in fields:
    c = 0
    for o in objs_m:
        if k in o and o.get(k) not in (None, {}, []):
            c += 1
    present.append(c)

plt.figure(figsize=(7, 4))
plt.bar(fields, present)
plt.title('合并图关键字段覆盖(非空对象数)')
plt.ylabel('对象数量')
plt.xticks(rotation=20)
for i, v in enumerate(present):
    plt.text(i, v + 0.2, str(v), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(out_dir / '03_关键字段覆盖率.png', dpi=180)
plt.close()

summary = {
    'capture_counts': dict(zip(subs, counts)),
    'objects_history': len(objs),
    'objects_merged': len(objs_m),
}
with (out_dir / 'stats.json').open('w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print('done', out_dir)
print(summary)
