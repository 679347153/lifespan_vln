"""查询本地 CLIP 文本向量索引。

用法:
    PYTHONPATH=. python CLIP_RAG/query_clip_index.py \
        --index-dir clip_index \
        --query "一个绿色的植物" \
        --top-k 5

流程:
    1. 读取 meta.json / ids.json / embeddings.npy
    2. 对查询文本编码 (与索引同模型) 并归一化
    3. 支持 top-k 或相似度阈值 (--min-score) 过滤；若同时给出则先阈值再截取 top-k
    4. 输出 JSON 行 (rank, obj_id, consine_similarity, description, floor_id, room_id)
"""
from __future__ import annotations
import argparse, json, pathlib, yaml, re, difflib, sys
from typing import List
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel


def _resolve_index_dir(root_dir: pathlib.Path, index_name: str | None) -> pathlib.Path:
    if index_name:
        target = root_dir / index_name
        if not target.exists():
            candidates = [d.name for d in root_dir.glob('*') if d.is_dir() and (d / 'meta.json').exists()]
            hint = ''
            if candidates:
                sim = difflib.get_close_matches(index_name, candidates, n=3, cutoff=0.3)
                hint_lines = ["可用索引列表:"] + [f"  - {c}" for c in candidates]
                if sim:
                    hint_lines.append("相似名称建议: " + ", ".join(sim))
                hint = "\n" + "\n".join(hint_lines)
            raise SystemExit(f"[ERR] 指定索引不存在: {target}{hint}")
        return target
    # 自动挑选最近修改的索引目录
    candidates = [d for d in root_dir.glob('*') if d.is_dir() and (d / 'meta.json').exists()]
    if not candidates:
        raise SystemExit(f"[ERR] 未发现任何索引目录于 {root_dir}")
    candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return candidates[0]


def load_index(index_dir: pathlib.Path):
    p = index_dir
    ids = json.loads((p / 'ids.json').read_text(encoding='utf-8'))
    vecs = np.load(p / 'embeddings.npy')  # [N, D]
    # texts (可选)
    texts = {}
    obj_jsonl = p / 'objects.jsonl'
    if obj_jsonl.exists():
        for line in obj_jsonl.read_text(encoding='utf-8').splitlines():
            try:
                rec = json.loads(line)
                # 兼容旧字段 'text', 新字段 'description'
                txt_val = rec.get('description') if rec.get('description') is not None else rec.get('text','')
                texts[rec['obj_id']] = txt_val
            except Exception:
                continue
    meta = json.loads((p / 'meta.json').read_text(encoding='utf-8'))
    return ids, vecs, texts, meta


def encode_query(texts: List[str], model_name: str, device: str = 'cpu'):
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.to(device)
    model.eval()
    inputs = processor(text=texts, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return feats.cpu().numpy()


def search(vecs: np.ndarray, ids: List[str], query_vec: np.ndarray, top_k: int | None, min_score: float | None):
    sims = (vecs @ query_vec.T).reshape(-1)
    order = np.argsort(-sims)
    pairs = [(i, float(sims[i])) for i in order]
    if min_score is not None:
        pairs = [p for p in pairs if p[1] >= min_score]
    if top_k is not None:
        pairs = pairs[:top_k]
    results = []
    for rank, (idx, sc) in enumerate(pairs, start=1):
        results.append((rank, ids[idx], sc, idx))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root_dir', default='CLIP_RAG', help='索引根目录 (build 时指定)')
    ap.add_argument('--index-name', default=None, help='索引子目录名；不提供则自动选择最新')
    ap.add_argument('--query', required=True)
    ap.add_argument('--top-k', type=int, default=None)
    ap.add_argument('--min-score', type=float, default=None, help='相似度过滤阈值 (0~1). 若未提供尝试从 map.yaml 的 CLIP_RAG.consine_similarity_filmin 读取')
    ap.add_argument('--config', default='config/map.yaml', help='可选配置文件 (用于默认阈值)')
    ap.add_argument('--no-threshold', action='store_true', help='忽略配置文件中的默认阈值 (即使存在 consine_similarity_filmin)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    root_dir = pathlib.Path(args.root_dir)
    if not root_dir.exists():
        raise SystemExit(f"[ERR] 索引根目录不存在: {root_dir}")
    # 若 root_dir 本身就是一个索引目录 (含 meta.json / ids.json), 直接使用
    if args.index_name is None and (root_dir / 'meta.json').exists() and (root_dir / 'ids.json').exists():
        index_dir = root_dir
    else:
        index_dir = _resolve_index_dir(root_dir, args.index_name)
    ids, vecs, texts, meta = load_index(index_dir)
    min_score = args.min_score
    if not args.no_threshold and min_score is None and args.config and pathlib.Path(args.config).exists():
        try:
            with open(args.config, 'r', encoding='utf-8') as cf:
                cfg = yaml.safe_load(cf)
            clip_cfg = cfg.get('CLIP_RAG', {}) if isinstance(cfg, dict) else {}
            val = clip_cfg.get('consine_similarity_filmin')  # 保持原拼写
            if isinstance(val, (int, float)):
                min_score = float(val)
        except Exception:
            pass
    # 若用户仅给 top-k 而没有阈值且未从配置加载到阈值 -> 不做阈值过滤
    # (当前逻辑已满足, 此处只是显式注释)
    qv = encode_query([args.query], meta['model'], device=args.device)
    results = search(vecs, ids, qv, args.top_k, min_score)
    if not results:
        # 计算最高分用于提示
        sims_all = (vecs @ qv.T).reshape(-1)
        best_idx = int(np.argmax(sims_all)) if sims_all.size else -1
        best_score = float(sims_all[best_idx]) if best_idx >= 0 else 0.0
        msg = ["[INFO] 未命中任何结果"]
        if min_score is not None:
            msg.append(f"  使用阈值 min-score={min_score} 过滤后为空")
            msg.append(f"  最高相似度={best_score:.4f} (< 阈值) 对象ID={ids[best_idx] if best_idx>=0 else 'N/A'}")
            msg.append("  建议: ")
            msg.append("    1) 降低 --min-score 或使用 --no-threshold")
            msg.append("    2) 检查构建索引时文本来源 (--field) 是否与查询语义匹配")
            msg.append("    3) 使用英文/同义词 或 更具体/更短的关键词")
        else:
            msg.append("  (无阈值情况下为空，可能索引为空或编码失败)")
        print("\n".join(msg))
        return
    # 为输出 floor_id / room_id, 从 objects.jsonl 读取缓存
    objects_path = index_dir / 'objects.jsonl'
    fr_map = {}
    if objects_path.exists():
        for line in objects_path.read_text(encoding='utf-8').splitlines():
            try:
                rec = json.loads(line)
                fr_map[rec.get('obj_id')] = (rec.get('floor_id'), rec.get('room_id'))
            except Exception:
                continue
    for rank, oid, score, _raw_idx in results:
        floor_id, room_id = fr_map.get(oid, (None, None))
        print(json.dumps({
            'rank': rank,
            'obj_id': oid,
            'consine_similarity': round(score, 6),
            'description': texts.get(oid, ''),
            'floor_id': floor_id,
            'room_id': room_id
        }, ensure_ascii=False))

if __name__ == '__main__':
    main()
