import json, argparse, math
from pathlib import Path
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageDraw, ImageFont

CONFIG_DEFAULT = Path('config/map.yaml')
MAP_DEFAULT = Path('RAG_Graph/test_save/map_mergedALL.json')

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

PALETTE = [
    (66, 135, 245), (245, 127, 66), (66, 245, 138), (199, 66, 245), (245, 218, 66),
    (66, 245, 233), (245, 66, 179), (140, 140, 140)
]

def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists() or yaml is None:
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def load_map(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return []

def load_prob(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))

def room_bbox(objs: List[Dict[str, Any]]):
    xs, ys = [], []
    for o in objs:
        region = o.get('region') or []
        for pt in region:
            xs.append(pt.get('x'))
            ys.append(pt.get('y'))
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))

def collect_room_bboxes(floors) -> Dict[str, Tuple[float,float,float,float]]:
    bboxes = {}
    for fl in floors:
        for rm in fl.get('rooms', []):
            rid = rm.get('room_id')
            bb = room_bbox(rm.get('objects', []))
            if bb:
                bboxes[rid] = bb
    return bboxes

def normalize_coords(bboxes: Dict[str, Tuple[float,float,float,float]], pad=20):
    xs, ys = [], []
    for bb in bboxes.values():
        xs += [bb[0], bb[2]]
        ys += [bb[1], bb[3]]
    if not xs or not ys:
        return {}, (0,0,1,1)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    scale = 800 / max(maxx - minx, maxy - miny, 1e-6)
    norm = {}
    for rid, (x1,y1,x2,y2) in bboxes.items():
        nx1 = pad + (x1 - minx) * scale
        ny1 = pad + (y1 - miny) * scale
        nx2 = pad + (x2 - minx) * scale
        ny2 = pad + (y2 - miny) * scale
        norm[rid] = (nx1, ny1, nx2, ny2)
    W = int((maxx - minx) * scale + pad*2)
    H = int((maxy - miny) * scale + pad*2)
    return norm, (W,H)

def color_for(idx: int):
    return PALETTE[idx % len(PALETTE)]

def render(prob: Dict[str,float], norm_bboxes: Dict[str,Tuple[float,float,float,float]], size, out: Path, title: str):
    W,H = size
    img = Image.new('RGBA', (W,H), (255,255,255,255))
    draw = ImageDraw.Draw(img)
    # attempt font
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 14)
    except Exception:
        font = ImageFont.load_default()
    max_p = max(prob.values()) if prob else 1.0
    for idx,(rid, bb) in enumerate(norm_bboxes.items()):
        p = prob.get(rid, 0.0)
        alpha = int(40 + 200 * (p / max_p if max_p>0 else 0))
        fill = (*color_for(idx), alpha)
        draw.rectangle(bb, fill=fill, outline=(0,0,0,120), width=2)
        x1,y1,x2,y2 = bb
        draw.text((x1+3, y1+3), f"{rid}:{p:.2f}", fill=(0,0,0), font=font)
    draw.text((10, H-20), title, fill=(0,0,0), font=font)
    img.save(out)


def main():
    ap = argparse.ArgumentParser(description='Visualize GMM room probability map')
    ap.add_argument('--gmm-json', required=True, help='compute_room_probabilities 输出的 JSON 文件')
    ap.add_argument('--map', default=str(MAP_DEFAULT))
    ap.add_argument('--output-dir', default='GMM_vis')
    ap.add_argument('--prefix', default='gmm')
    args = ap.parse_args()
    prob_path = Path(args.gmm_json)
    gmm = load_prob(prob_path)
    floors = load_map(Path(args.map))
    bboxes = collect_room_bboxes(floors)
    norm, size = normalize_coords(bboxes)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with_w = gmm.get('with_walls') or {}
    no_w = gmm.get('no_walls') or {}
    render(with_w, norm, size, out_dir / f"{args.prefix}_with_walls.png", f"{gmm.get('target','?')} with_walls")
    if no_w:
        render(no_w, norm, size, out_dir / f"{args.prefix}_no_walls.png", f"{gmm.get('target','?')} no_walls")
    # 保存一个合成概览
    summary = {
        'target': gmm.get('target'),
        'rooms': {rid: {'with_walls': with_w.get(rid,0.0), 'no_walls': no_w.get(rid,0.0)} for rid in bboxes.keys()},
        'image_with_walls': str((out_dir / f"{args.prefix}_with_walls.png").resolve()),
        'image_no_walls': str((out_dir / f"{args.prefix}_no_walls.png").resolve()) if no_w else None
    }
    (out_dir / f"{args.prefix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
