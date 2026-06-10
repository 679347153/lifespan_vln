"""构建本地 CLIP 文本索引 (对象描述或标签) 并保存向量库.

用法:
    PYTHONPATH=. python CLIP_RAG/build_clip_index.py \
        --map RAG_Graph/test_save/map_mergedALL.json \
        --out-dir clip_index \
        --field auto \
        --model openai/clip-vit-base-patch32

说明:
    1. 解析语义地图 JSON, 遍历所有对象.
    2. 文本来源 (已更新): 默认优先 label; 若 --field=desc 强制描述则回退 label。
    3. 支持 --field=label / desc / auto 强制选择文本构造.
    4. 输出: out-dir/
        objects.jsonl  (每行: {obj_id, description})  # description: 优先使用 description.brief 其后回退最后一次扫描描述 / label
        embeddings.npy (shape: [N, D])
        ids.json       (obj_id 按顺序列表)
        meta.json      (记录 model_name / field / dim / count)
    5. 可重复运行, 若对象数一致直接覆盖.
"""
from __future__ import annotations
import argparse, json, pathlib, hashlib, datetime, yaml
from typing import List, Dict, Any, Optional

import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# ---------------- Data Extraction ---------------- #

def load_map_objects(map_path: str) -> List[Dict[str, Any]]:
    """展开地图对象列表, 补充 floor_id / room_id 以便索引持久化。"""
    data = json.loads(pathlib.Path(map_path).read_text(encoding='utf-8'))
    objs: List[Dict[str, Any]] = []
    for floor in data:
        fid = floor.get('floor_id')
        for room in floor.get('rooms', []):
            rid = room.get('room_id')
            for obj in room.get('objects', []):
                if 'floor_id' not in obj:
                    obj['floor_id'] = fid
                if 'room_id' not in obj:
                    obj['room_id'] = rid
                objs.append(obj)
    return objs

def pick_object_text(obj: Dict[str, Any], mode: str = 'label') -> str:
    """选择编码文本 (兼容旧参数名), 内部转发到 pick_object_description.
    保持对外 --field 语义不变, 返回用于索引的描述文本。"""
    return pick_object_description(obj, mode)


def pick_object_description(obj: Dict[str, Any], mode: str = 'label') -> str:
    """生成用于索引的描述(description):
    优先顺序:
        1) 若 description 是 dict 且含有键 'brief' 且非空 => 使用 description['brief']
        2) 否则尝试获取最后一次扫描生成的描述: 取 description 中最大的数字键对应的值
        3) 再否则在 description dict 任意一个非空字符串值
        4) 回退 label/name

    mode 影响策略:
        - label: 强制使用 label/name
        - desc: 跳过 label, 按上述 1~3 规则; 若都为空才回退 label
        - auto: label 若存在优先, label 为空则走 desc 规则
    """
    label = (obj.get('label') or '').strip()
    desc_dict = obj.get('description') if isinstance(obj.get('description'), dict) else {}
    brief = ''
    last_scan = ''
    any_desc = ''
    if desc_dict:
        # 1) brief
        b = desc_dict.get('brief')
        if isinstance(b, str) and b.strip():
            brief = b.strip()
        # 2) 最大数字键
        numeric_keys = [k for k in desc_dict.keys() if str(k).isdigit()]
        if numeric_keys:
            try:
                last_scan_candidate = desc_dict[max(numeric_keys, key=lambda x: int(str(x)))]
                if isinstance(last_scan_candidate, str) and last_scan_candidate.strip():
                    last_scan = last_scan_candidate.strip()
            except Exception:
                pass
        # 3) 任意非空
        if not last_scan:
            for v in desc_dict.values():
                if isinstance(v, str) and v.strip():
                    any_desc = v.strip()
                    break

    def choose_desc_no_label():
        return brief or last_scan or any_desc or label

    if mode == 'desc':
        return choose_desc_no_label()
    # label (默认)
    return label

# ---------------- Embedding ---------------- #

def build_embeddings(texts: List[str], model_name: str, device: str = 'cpu', batch_size: int = 32) -> np.ndarray:
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.to(device)
    model.eval()
    all_vecs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = processor(text=batch, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            feats = model.get_text_features(**inputs)  # [B, D]
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        all_vecs.append(feats.cpu().numpy())
    return np.concatenate(all_vecs, axis=0)

# ---------------- Main ---------------- #

def _auto_index_name(map_path: pathlib.Path, model: str) -> str:
    stem = map_path.stem  # e.g. map_mergedALL
    model_tag = model.replace('/', '-').replace(':', '-')
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    # 使用文件内容 hash 片段避免重复 (可选)
    try:
        content = map_path.read_bytes()
        h = hashlib.md5(content).hexdigest()[:8]
    except Exception:
        h = 'na'
    return f"{stem}__{model_tag}__{ts}__{h}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', required=True, help='语义地图 JSON 路径')
    ap.add_argument('--root-dir', default='CLIP_RAG', help='索引根目录 (默认放在 CLIP_RAG/)')
    ap.add_argument('--index-name', default=None, help='自定义索引子目录名称；若不提供自动生成')
    ap.add_argument('--model', default='openai/clip-vit-base-patch32')
    ap.add_argument('--field', default='label', choices=['label','desc'])
    ap.add_argument('--device', default=None, help='设备: cpu|cuda|auto (默认 auto 优先 cuda)')
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--config', default='config/map.yaml', help='可选配置文件 (提供 CLIP_RAG.text_embedding.field_mode / batch_size / device)')
    ap.add_argument('--force', action='store_true', help='若目录已存在则覆盖')
    args = ap.parse_args()

    map_path = pathlib.Path(args.map).expanduser().resolve()
    if not map_path.exists():
        raise SystemExit(f"[ERR] 地图文件不存在: {map_path}")

    # 计算输出目录
    root_dir = pathlib.Path(args.root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    index_name = args.index_name or _auto_index_name(map_path, args.model)
    out_dir = root_dir / index_name
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"[ERR] 目标目录已存在且非空: {out_dir} (使用 --force 覆盖 或指定 --index-name)")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 配置覆盖逻辑
    if args.config and pathlib.Path(args.config).exists():
        try:
            cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8')) or {}
        except Exception:
            cfg = {}
        clip_txt_cfg = (cfg.get('CLIP_RAG') or {}).get('text_embedding') or {}
        # field 覆盖: 仅当未显式传入且配置提供时
        if 'field_mode' in clip_txt_cfg and args.field == 'label' and clip_txt_cfg.get('field_mode') in ('label','desc'):
            args.field = clip_txt_cfg.get('field_mode')
        # batch size 覆盖
        if args.batch_size is None and isinstance(clip_txt_cfg.get('batch_size'), int):
            args.batch_size = int(clip_txt_cfg['batch_size'])
        # device 覆盖
        if args.device is None:# HACK
            dev_cfg = clip_txt_cfg.get('device', 'auto')
            if dev_cfg == 'auto':
                args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                args.device = dev_cfg
    # 默认兜底
    if args.batch_size is None:
        args.batch_size = 32
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    objs = load_map_objects(str(map_path))
    if not objs:
        print('[WARN] 地图中未发现对象')
        return
    ids: List[str] = []
    texts: List[str] = []
    for o in objs:
        oid = o.get('obj_id')
        if not oid:
            continue
        ids.append(oid)
        texts.append(pick_object_description(o, args.field))
    print(f'[INFO] 索引名称: {index_name}')
    print(f'[INFO] 采集对象 {len(ids)} 条, 构建文本向量... (device={args.device}, field_mode={args.field})')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('[WARN] 请求使用 GPU 但当前不可用, 回退 CPU')
        args.device = 'cpu'
    vecs = build_embeddings(texts, args.model, args.device, args.batch_size)
    # Save
    (out_dir / 'ids.json').write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding='utf-8')
    np.save(out_dir / 'embeddings.npy', vecs.astype('float32'))
    with (out_dir / 'objects.jsonl').open('w', encoding='utf-8') as f:
        for o, txt in zip(objs, texts):
            if o.get('obj_id') not in ids:
                continue
            # description 字段写入: brief/last_numeric/fallback_label 按 pick_object_description 逻辑
            rec = {
                'obj_id': o.get('obj_id'),
                'floor_id': o.get('floor_id'),
                'room_id': o.get('room_id'),
                'description': txt
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    meta = {
        'model': args.model,
        'count': len(ids),
        'dim': int(vecs.shape[1]),
        'field_mode': args.field,
        'device': args.device,
        'source_map': str(map_path),
        'index_name': index_name,
        'root_dir': str(root_dir.resolve())
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[OK] 索引完成 -> {out_dir} (N={meta["count"]}, dim={meta["dim"]})')
    print('[HINT] 查询示例:')
    print(f'  PYTHONPATH=. python CLIP_RAG/query_clip_index.py --root-dir {root_dir} --index-name {index_name} --query "植物"')

if __name__ == '__main__':
    main()
