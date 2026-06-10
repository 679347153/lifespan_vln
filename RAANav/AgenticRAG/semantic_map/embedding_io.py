"""Embedding 存取辅助：将对象 clip_embedding 按 obj_id 单独存文件或集中存储。

提供两套方式：
1. save_embedding_npy(dir_path, obj_id, vector)  -> dir_path/obj_id.npy
   load_embedding_npy(dir_path, obj_id) -> List[float]
2. append_embedding_jsonl(file_path, obj_id, vector)
   load_embedding_jsonl(file_path, obj_id)

默认推荐 .npy 逐文件方式，便于局部更新与延迟加载。
"""
from __future__ import annotations
import json, os, pathlib, math
from typing import List, Optional

import struct

try:
    import numpy as np  # type: ignore
except Exception:  # 允许无 numpy 环境 (降级 json 存储)
    np = None  # type: ignore


def _ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

# ---------- NPY 逐文件 ---------- #

def save_embedding_npy(dir_path: str, obj_id: str, vector: List[float]) -> str:
    pdir = pathlib.Path(dir_path)
    _ensure_dir(pdir)
    if np is None:
        # 回退为 .json
        out = pdir / f"{obj_id}.json"
        out.write_text(json.dumps(vector), encoding='utf-8')
        return out.as_posix()
    arr = np.asarray(vector, dtype='float32')
    out = pdir / f"{obj_id}.npy"
    np.save(out, arr, allow_pickle=False)
    return out.as_posix()

def load_embedding_npy(dir_path: str, obj_id: str) -> Optional[List[float]]:
    pdir = pathlib.Path(dir_path)
    if np is not None:
        f = pdir / f"{obj_id}.npy"
        if f.exists():
            try:
                arr = np.load(f, allow_pickle=False)
                return arr.astype('float32').tolist()
            except Exception:
                return None
    # 尝试 json 回退
    f_json = pdir / f"{obj_id}.json"
    if f_json.exists():
        try:
            return json.loads(f_json.read_text(encoding='utf-8'))
        except Exception:
            return None
    return None

# ---------- JSONL 集中 ---------- #

def append_embedding_jsonl(file_path: str, obj_id: str, vector: List[float]) -> None:
    line = json.dumps({"id": obj_id, "vec": vector}, ensure_ascii=False)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def load_embedding_jsonl(file_path: str, obj_id: str) -> Optional[List[float]]:
    p = pathlib.Path(file_path)
    if not p.exists():
        return None
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get('id') == obj_id:
                vec = rec.get('vec')
                if isinstance(vec, list):
                    return [float(x) for x in vec]
    return None

# ---------- 统一接口 ---------- #

def save_embedding(obj_id: str, vector: List[float], base_dir: str = 'embeddings', mode: str = 'npy') -> str:
    if mode == 'jsonl':
        path = pathlib.Path(base_dir)
        _ensure_dir(path.parent if path.suffix else path)
        fp = path / 'embeddings.jsonl' if path.is_dir() else path
        append_embedding_jsonl(fp.as_posix(), obj_id, vector)
        return fp.as_posix()
    else:
        return save_embedding_npy(base_dir, obj_id, vector)

def load_embedding(obj_id: str, base_dir: str = 'embeddings', mode: str = 'npy') -> Optional[List[float]]:
    if mode == 'jsonl':
        path = pathlib.Path(base_dir)
        fp = path / 'embeddings.jsonl' if path.is_dir() else path
        return load_embedding_jsonl(fp.as_posix(), obj_id)
    else:
        return load_embedding_npy(base_dir, obj_id)

__all__ = [
    'save_embedding_npy','load_embedding_npy','append_embedding_jsonl','load_embedding_jsonl',
    'save_embedding','load_embedding'
]
