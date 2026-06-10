"""基于内存中语义地图对象直接构建 / 校验 CLIP 向量索引。

场景: 地图对象 (dict 结构) 尚未落盘或落盘 JSON 不含 clip_embedding, 需要
复用对象已存在的内存属性 `clip_embedding`；缺失时按 label/desc 生成。

提供主函数:
    build_runtime_clip_index(floors, *, field_mode='label', model_name='openai/clip-vit-base-patch32',
                             device=None, batch_size=32, ensure_consistent=True) -> dict
返回结构:
    {
        'embeddings': np.ndarray(shape=[N,D], dtype=float32),
        'ids': List[str],
        'texts': List[str],    # 实际用于编码的文本 (description 语义)
        'dim': D,
        'inconsistent_dims': {dim:count,...}  # 若传入对象 clip_embedding 有多种维度
    }

若 ensure_consistent=True 且发现已有 clip_embedding 维度仅一种且 == D, 则优先直接堆叠已有向量；
对缺失或维度不匹配的条目重新编码后补齐；若存在多种维度将统一全部重编码。

依赖: transformers, torch, numpy
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import torch, numpy as np
from transformers import CLIPProcessor, CLIPModel
import pathlib, json, hashlib, datetime

# ---- 文本选择逻辑 (与 build_clip_index 保持一致, 无 auto) ---- #

def _select_text(obj: Dict[str, Any], mode: str) -> str:
    label = (obj.get('label') or obj.get('name') or '').strip()
    if mode == 'label':
        return label
    # desc 模式
    desc_dict = obj.get('description') if isinstance(obj.get('description'), dict) else {}
    brief = ''
    last_scan = ''
    any_desc = ''
    if desc_dict:
        b = desc_dict.get('brief')
        if isinstance(b, str) and b.strip():
            brief = b.strip()
        numeric_keys = [k for k in desc_dict.keys() if str(k).isdigit()]
        if numeric_keys:
            try:
                cand = desc_dict[max(numeric_keys, key=lambda x: int(str(x)))]
                if isinstance(cand, str) and cand.strip():
                    last_scan = cand.strip()
            except Exception:
                pass
        if not last_scan:
            for v in desc_dict.values():
                if isinstance(v, str) and v.strip():
                    any_desc = v.strip(); break
    return brief or last_scan or any_desc or label

# ---- 主函数 ---- #

def build_runtime_clip_index(floors: List[Dict[str, Any]], *, field_mode: str = 'label',
                              model_name: str = 'openai/clip-vit-base-patch32', device: str | None = None,
                              batch_size: int = 32, ensure_consistent: bool = True) -> Dict[str, Any]:
    assert field_mode in ('label','desc'), 'field_mode 仅支持 label|desc'
    if device is None or device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 收集对象
    objs: List[Dict[str, Any]] = []
    for f in floors:
        for r in f.get('rooms', []):
            for o in r.get('objects', []):
                if o.get('obj_id'):
                    objs.append(o)
    ids: List[str] = []
    texts: List[str] = []
    existing_vecs: List[Tuple[int, np.ndarray]] = []  # (index, vec)
    dim_counter: Dict[int,int] = {}
    for o in objs:
        oid = o['obj_id']
        ids.append(oid)
        texts.append(_select_text(o, field_mode))
    # 检查已有向量
    for idx, o in enumerate(objs):
        emb = o.get('clip_embedding')
        if isinstance(emb, list) and emb:
            try:
                arr = np.asarray(emb, dtype='float32')
                d = arr.shape[0]
                dim_counter[d] = dim_counter.get(d,0)+1
                existing_vecs.append((idx, arr))
            except Exception:
                pass
    inconsistent_dims = dim_counter.copy()
    multi_dim = len(dim_counter) > 1
    # 编码所需文本索引
    need_encode_indices = []
    if not existing_vecs or multi_dim or not ensure_consistent:
        # 统一全部重编码 (multi_dim) 或 无现有向量
        need_encode_indices = list(range(len(objs)))
    else:
        # 仅对缺失向量编码
        present = {i for i,_ in existing_vecs}
        need_encode_indices = [i for i in range(len(objs)) if i not in present]
    # 执行编码
    if need_encode_indices:
        model = CLIPModel.from_pretrained(model_name)
        proc = CLIPProcessor.from_pretrained(model_name)
        model.to(device); model.eval()
        encode_texts = [texts[i] for i in need_encode_indices]
        all_vecs: List[np.ndarray] = []
        for i in range(0, len(encode_texts), batch_size):
            batch = encode_texts[i:i+batch_size]
            inputs = proc(text=batch, return_tensors='pt', padding=True, truncation=True).to(device)
            with torch.no_grad():
                feats = model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            all_vecs.append(feats.cpu().numpy())
        new_mat = np.concatenate(all_vecs, axis=0) if all_vecs else np.zeros((0,512), dtype=np.float32)
        # 写回对象 (保持原顺序)
        for i, vec in zip(need_encode_indices, new_mat):
            objs[i]['clip_embedding'] = vec.round(6).tolist()
        final_dim = new_mat.shape[1] if new_mat.size else (existing_vecs[0][1].shape[0] if existing_vecs else 0)
    else:
        final_dim = existing_vecs[0][1].shape[0] if existing_vecs else 0
    # 构造最终 embeddings 按 ids 顺序
    emb_rows = []
    for o in objs:
        v = o.get('clip_embedding')
        if isinstance(v, list) and v:
            emb_rows.append(np.asarray(v, dtype='float32'))
        else:
            emb_rows.append(np.zeros((final_dim,), dtype='float32'))
    embeddings = np.stack(emb_rows, axis=0) if emb_rows else np.zeros((0, final_dim), dtype='float32')
    return {
        'embeddings': embeddings,
        'ids': ids,
        'texts': texts,
        'dim': final_dim,
        'inconsistent_dims': inconsistent_dims,
        'reencoded_all': bool(need_encode_indices and (not existing_vecs or multi_dim)),
    }

__all__ = ['build_runtime_clip_index']

# ---- 持久化为与 build_clip_index 一致的索引目录 ---- #

def _gen_index_name(tag: str, model_name: str) -> str:
    model_tag = model_name.replace('/', '-').replace(':', '-')
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    h = hashlib.md5(tag.encode('utf-8')).hexdigest()[:8]
    return f"{tag}__{model_tag}__{ts}__{h}"

def persist_runtime_clip_index(
    floors: List[Dict[str, Any]],
    *,
    root_dir: str = 'CLIP_RAG',
    index_name: str | None = None,
    map_tag: str = 'runtime',
    field_mode: str = 'label',
    model_name: str = 'openai/clip-vit-base-patch32',
    device: str | None = None,
    batch_size: int = 32,
    force: bool = True,
) -> str:
    """基于内存对象构建并保存索引目录。

    返回: 索引目录路径 (str)
    目录结构: ids.json / embeddings.npy / objects.jsonl / meta.json
    与 build_clip_index.py 输出保持一致。
    """
    res = build_runtime_clip_index(
        floors,
        field_mode=field_mode,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        ensure_consistent=True,
    )
    root = pathlib.Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    name = index_name or _gen_index_name(map_tag, model_name)
    out_dir = root / name
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise FileExistsError(f"目标目录已存在且非空: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # 写入
    (out_dir / 'ids.json').write_text(json.dumps(res['ids'], ensure_ascii=False, indent=2), encoding='utf-8')
    np.save(out_dir / 'embeddings.npy', res['embeddings'].astype('float32'))
    # objects.jsonl
    with (out_dir / 'objects.jsonl').open('w', encoding='utf-8') as f:
        for oid, txt in zip(res['ids'], res['texts']):
            # 尝试提取 floor_id / room_id
            floor_id = None; room_id = None
            # 遍历原 floors 查找 (少量开销, 可优化建索引)
            for fl in floors:
                for rm in fl.get('rooms', []):
                    for o in rm.get('objects', []):
                        if o.get('obj_id') == oid:
                            floor_id = fl.get('floor_id')
                            room_id = rm.get('room_id')
                            break
                    if floor_id is not None:
                        break
                if floor_id is not None:
                    break
            rec = {
                'obj_id': oid,
                'floor_id': floor_id,
                'room_id': room_id,
                'description': txt
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    meta = {
        'model': model_name,
        'count': len(res['ids']),
        'dim': int(res['dim']),
        'field_mode': field_mode,
        'device': device or ('cuda' if torch.cuda.is_available() else 'cpu'),
        'source_map': map_tag,
        'index_name': name,
        'root_dir': str(root.resolve()),
        'runtime': True,
        'inconsistent_dims': res['inconsistent_dims']
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return out_dir.as_posix()

__all__.append('persist_runtime_clip_index')
