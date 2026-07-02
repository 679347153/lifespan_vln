from __future__ import annotations

import argparse
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .common import as_path, is_noise_detection_label, sanitize_detection_label, write_json
from .diagnose_temporal_run import diagnose


def _read_json_optional(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path, *, max_records: int = 0) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if max_records > 0 and len(rows) >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _pose_xz(value: Any) -> Optional[List[float]]:
    if isinstance(value, dict):
        if "x" in value and "z" in value:
            x = _safe_float(value.get("x"))
            z = _safe_float(value.get("z"))
            if x is not None and z is not None:
                return [x, z]
        pos = value.get("position")
        if isinstance(pos, list) and len(pos) >= 3:
            x = _safe_float(pos[0])
            z = _safe_float(pos[2])
            if x is not None and z is not None:
                return [x, z]
    if isinstance(value, list) and len(value) >= 2:
        x = _safe_float(value[0])
        z = _safe_float(value[1])
        if x is not None and z is not None:
            return [x, z]
    return None


def _obs_xz(record: Dict[str, Any]) -> Optional[List[float]]:
    for key in ("pos_2d", "pos_3d"):
        value = record.get(key)
        if isinstance(value, list):
            if key == "pos_2d" and len(value) >= 2:
                x = _safe_float(value[0])
                z = _safe_float(value[1])
                if x is not None and z is not None:
                    return [x, z]
            if key == "pos_3d" and len(value) >= 3:
                x = _safe_float(value[0])
                z = _safe_float(value[2])
                if x is not None and z is not None:
                    return [x, z]
    return None


def _count_by(records: Iterable[Dict[str, Any]], key: str, *, limit: int = 50) -> List[Dict[str, Any]]:
    counts: Counter[str] = Counter()
    for record in records:
        value = record.get(key)
        if key == "label":
            value = sanitize_detection_label(value)
            if is_noise_detection_label(value):
                continue
        if value is not None:
            counts[str(value)] += 1
    return [{"key": key_, "count": count} for key_, count in counts.most_common(limit)]


def _candidate_backend_counts(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: Counter[str] = Counter()
    for record in records:
        selected = record.get("selected_candidate")
        if isinstance(selected, dict):
            counts[str(selected.get("sim_backend") or "unknown")] += 1
    return [{"key": key, "count": count} for key, count in counts.most_common()]


def _points_from_records(
    run_rows: List[Dict[str, Any]],
    diagnostics: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    observations: List[Dict[str, Any]],
) -> Dict[str, List[List[float]]]:
    points: Dict[str, List[List[float]]] = defaultdict(list)
    for row in run_rows:
        for step in row.get("steps", []) or []:
            pos = _pose_xz((step or {}).get("position"))
            if pos:
                points["trajectory"].append(pos)
        for trace in row.get("subtask_traces", []) or []:
            pos = _pose_xz((trace or {}).get("final_pose"))
            if pos:
                points["final"].append(pos)
    for record in diagnostics:
        pos = _pose_xz(record.get("target_position"))
        if pos:
            points["target"].append(pos)
        for peak in record.get("peaks", []) or []:
            if isinstance(peak, dict):
                x = _safe_float(peak.get("world_x"))
                z = _safe_float(peak.get("world_z"))
                if x is not None and z is not None:
                    points["peak"].append([x, z])
    for record in candidates:
        for key in ("planned_pose", "final_pose"):
            pos = _pose_xz(record.get(key))
            if pos:
                points[key].append(pos)
        selected = record.get("selected_candidate")
        if isinstance(selected, dict):
            x = _safe_float(selected.get("world_x"))
            z = _safe_float(selected.get("world_z"))
            if x is not None and z is not None:
                points["selected_candidate"].append([x, z])
    for record in observations:
        pos = _obs_xz(record)
        if pos:
            points["observation"].append(pos)
    return dict(points)


def _group_by(records: Iterable[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        out[str(record.get(key, ""))].append(record)
    return dict(out)


def _clean_observation_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for record in records:
        label = sanitize_detection_label(record.get("label"))
        if is_noise_detection_label(label):
            continue
        clean = dict(record)
        raw_label = str(record.get("raw_label") or record.get("label") or "").strip()
        if raw_label and raw_label != label:
            clean["raw_label"] = raw_label
        clean["label"] = label
        out.append(clean)
    return out


def _scene_memory_payload(output_dir: Path) -> Dict[str, Any]:
    scene_root = output_dir / "scene_memory"
    scenes: List[Dict[str, Any]] = []
    room_counts: Counter[str] = Counter()
    if not scene_root.exists():
        return {"scenes": [], "room_counts": []}
    for path in sorted(scene_root.rglob("scene_memory_final.json")):
        payload = _read_json_optional(path) or {}
        rooms = payload.get("rooms") if isinstance(payload.get("rooms"), dict) else {}
        tracks = payload.get("tracks") if isinstance(payload.get("tracks"), list) else []
        compact_tracks: List[Dict[str, Any]] = []
        for track in tracks:
            if not isinstance(track, dict):
                continue
            stats = track.get("cooccur_stats") if isinstance(track.get("cooccur_stats"), dict) else {}
            compact_tracks.append(
                {
                    "obj_id": track.get("obj_id"),
                    "label": track.get("label"),
                    "room_id": track.get("room_id"),
                    "pos_2d": track.get("pos_2d"),
                    "pos_3d": track.get("pos_3d"),
                    "exist_prob": track.get("exist_prob"),
                    "last_update_time": track.get("last_update_time"),
                    "cooccur_stats": {
                        "current_room_id": stats.get("current_room_id"),
                        "room_belief": stats.get("room_belief") or {},
                        "room_history": (stats.get("room_history") or [])[-20:],
                        "seen_count": stats.get("seen_count"),
                        "miss_count": stats.get("miss_count"),
                        "negative_feedback_count": stats.get("negative_feedback_count"),
                    },
                }
            )
        compact_rooms: Dict[str, Any] = {}
        for room_id, room in rooms.items():
            if not isinstance(room, dict):
                continue
            refs = room.get("object_refs")
            if isinstance(refs, dict):
                room_counts[str(room_id)] += len(refs)
                compact_refs: Dict[str, Any] = {}
                for obj_id, ref in refs.items():
                    if not isinstance(ref, dict):
                        continue
                    compact = dict(ref)
                    if isinstance(compact.get("positions_2d"), list):
                        compact["positions_2d"] = compact["positions_2d"][-10:]
                    compact_refs[str(obj_id)] = compact
                compact_rooms[str(room_id)] = {
                    "room_id": room.get("room_id", room_id),
                    "room_label": room.get("room_label", room_id),
                    "track_count": room.get("track_count", len(compact_refs)),
                    "object_refs": compact_refs,
                }
        scenes.append(
            {
                "scene_name": payload.get("scene_name") or path.parent.name,
                "track_count": payload.get("track_count", len(compact_tracks)),
                "room_assignment": payload.get("room_assignment") or {},
                "rooms": compact_rooms,
                "tracks": compact_tracks,
            }
        )
    return {
        "scenes": scenes,
        "room_counts": [{"key": key, "count": count} for key, count in room_counts.most_common(30)],
    }


def _scene_geometry_payload(output_dir: Path) -> Dict[str, Any]:
    geom_root = output_dir / "scene_geometry"
    scenes: Dict[str, Any] = {}
    if not geom_root.exists():
        return {"scenes": {}, "scene_count": 0}
    for path in sorted(geom_root.glob("*.json")):
        payload = _read_json_optional(path) or {}
        if not isinstance(payload, dict):
            continue
        scene_name = str(payload.get("scene_name") or path.stem)
        scenes[scene_name] = {
            "scene_name": scene_name,
            "source": payload.get("source") or {},
            "bounds": payload.get("bounds") or {},
            "grid_resolution": payload.get("grid_resolution"),
            "navmesh_contours": payload.get("navmesh_contours") or [],
            "navigable_points": payload.get("navigable_points") or [],
            "semantic_regions": payload.get("semantic_regions") or [],
            "layout_objects": payload.get("layout_objects") or [],
            "room_overlays": payload.get("room_overlays") or [],
            "notes": payload.get("notes") or [],
        }
    return {"scenes": scenes, "scene_count": len(scenes)}


def build_visual_payload(output_dir: Path, *, max_observations: int = 20000) -> Dict[str, Any]:
    output_dir = as_path(output_dir)
    run_rows = _read_jsonl(output_dir / "run.jsonl")
    diagnostics = _read_jsonl(output_dir / "memory_diagnostics.jsonl")
    candidates = _read_jsonl(output_dir / "candidate_traces.jsonl")
    observations = _clean_observation_records(
        _read_jsonl(output_dir / "observation_traces.jsonl", max_records=max_observations)
    )
    memory_updates = _read_jsonl(output_dir / "memory_updates.jsonl")
    negative_feedback = _read_jsonl(output_dir / "negative_feedback.jsonl")
    layout_transitions = _read_jsonl(output_dir / "layout_transitions.jsonl")
    summary = _read_json_optional(output_dir / "summary.json") or {}
    temporal_summary = _read_json_optional(output_dir / "temporal_summary.json") or {}
    perception_summary = _read_json_optional(output_dir / "perception_summary.json") or {}
    by_task_type = _read_json_optional(output_dir / "by_task_type.json") or {}
    diagnosis = diagnose(output_dir)
    scene_memory = _scene_memory_payload(output_dir)
    scene_geometry = _scene_geometry_payload(output_dir)

    episode_rows = []
    for row in run_rows:
        traces = row.get("subtask_traces", []) or []
        episode_rows.append(
            {
                "episode_id": row.get("episode_id"),
                "scene_name": row.get("scene_name"),
                "layout_id": row.get("layout_id"),
                "seen_layout_count_before": row.get("seen_layout_count_before"),
                "elapsed_seconds": row.get("elapsed_seconds"),
                "subtasks": len(traces),
                "reported_success": sum(1 for t in traces if bool((t or {}).get("reported_success"))),
                "path_length": round(
                    sum(float((t or {}).get("path_length", 0.0) or 0.0) for t in traces),
                    6,
                ),
                "steps": sum(int((t or {}).get("steps", 0) or 0) for t in traces),
            }
        )

    points = _points_from_records(run_rows, diagnostics, candidates, observations)
    payload = {
        "output_dir": str(output_dir),
        "summary": summary,
        "temporal_summary": temporal_summary,
        "perception_summary": perception_summary,
        "by_task_type": by_task_type,
        "diagnosis": diagnosis,
        "episodes": episode_rows,
        "run_rows": run_rows,
        "diagnostics": diagnostics,
        "candidates": candidates,
        "observations": observations,
        "memory_updates": memory_updates,
        "negative_feedback": negative_feedback,
        "layout_transitions": layout_transitions,
        "scene_memory": scene_memory,
        "scene_geometry": scene_geometry,
        "counts": {
            "observation_labels": _count_by(observations, "label", limit=30),
            "memory_events": _count_by(memory_updates, "event", limit=20),
            "negative_feedback_labels": _count_by(negative_feedback, "label", limit=30),
            "candidate_backends": _candidate_backend_counts(candidates),
            "targets": _count_by(diagnostics, "target_object", limit=30),
            "task_types": _count_by(diagnostics, "task_type", limit=10),
            "rooms": scene_memory.get("room_counts", []),
        },
        "groups": {
            "diagnostics_by_episode": _group_by(diagnostics, "episode_id"),
            "candidates_by_subtask": _group_by(candidates, "subtask_id"),
            "observations_by_layout": _group_by(observations, "layout_id"),
            "memory_by_layout": _group_by(memory_updates, "layout_id"),
            "feedback_by_subtask": _group_by(negative_feedback, "track_id"),
        },
        "points": points,
        "truncated": {
            "observations_max_records": max_observations,
            "observations_loaded": len(observations),
        },
    }
    return payload


def _html_template(payload_json: str, title: str) -> str:
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1f2933;
      --muted: #64748b;
      --line: #d7dde5;
      --blue: #2563eb;
      --green: #059669;
      --red: #dc2626;
      --amber: #b45309;
      --purple: #7c3aed;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 18px 24px 12px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    h1 {{ margin: 0 0 6px; font-size: 22px; }}
    h2 {{ margin: 0 0 10px; font-size: 17px; }}
    h3 {{ margin: 14px 0 8px; font-size: 14px; }}
    .subtle {{ color: var(--muted); font-size: 13px; }}
    main {{ padding: 18px 24px 28px; }}
    .grid {{ display: grid; gap: 14px; }}
    .metrics {{ grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }}
    .two {{ grid-template-columns: minmax(0, 1.1fr) minmax(360px, 0.9fr); }}
    .three {{ grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }}
    section, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      overflow: hidden;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .metric .value {{ font-size: 22px; font-weight: 650; }}
    .metric .label {{ color: var(--muted); font-size: 12px; margin-top: 3px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid #e8edf2; padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ color: #334155; background: #f8fafc; position: sticky; top: 0; }}
    tbody tr:hover {{ background: #f8fbff; }}
    .table-scroll {{ max-height: 420px; overflow: auto; border: 1px solid #e8edf2; border-radius: 6px; }}
    select, input {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 8px;
      background: #fff;
      color: var(--text);
      min-width: 220px;
    }}
    .controls {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-bottom: 12px; }}
    .badge {{ display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 11px; background: #eef2ff; color: #3730a3; }}
    .ok {{ color: var(--green); }}
    .bad {{ color: var(--red); }}
    .warn {{ color: var(--amber); }}
    svg {{ width: 100%; height: auto; display: block; }}
    .legend {{ display: flex; gap: 12px; flex-wrap: wrap; font-size: 12px; color: var(--muted); margin-top: 8px; }}
    .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: -1px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #f8fafc; border: 1px solid #e8edf2; border-radius: 6px; padding: 10px; font-size: 12px; max-height: 320px; overflow: auto; }}
    .bar-row {{ display: grid; grid-template-columns: minmax(80px, 170px) 1fr 44px; gap: 8px; align-items: center; margin: 5px 0; font-size: 12px; }}
    .bar-bg {{ height: 9px; background: #edf2f7; border-radius: 999px; overflow: hidden; }}
    .bar {{ height: 100%; background: var(--blue); }}
    .room-grid {{ display: grid; grid-template-columns: minmax(260px, 0.9fr) minmax(360px, 1.1fr); gap: 14px; }}
    .room-node {{ border: 1px solid #e8edf2; border-radius: 6px; margin: 8px 0; padding: 9px; background: #fbfdff; }}
    .room-node h3 {{ margin: 0 0 7px; }}
    .obj-ref {{ display: flex; justify-content: space-between; gap: 10px; padding: 4px 0; border-top: 1px dashed #e2e8f0; font-size: 12px; }}
    .belief {{ color: var(--green); font-variant-numeric: tabular-nums; }}
    @media (max-width: 960px) {{ .two {{ grid-template-columns: 1fr; }} main {{ padding: 12px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{safe_title}</h1>
    <div class="subtle" id="outputDir"></div>
  </header>
  <main class="grid">
    <div class="grid metrics" id="metrics"></div>
    <section>
      <h2>长期趋势</h2>
      <div id="trend"></div>
      <div class="legend">
        <span><i class="swatch" style="background:var(--blue)"></i>found_rate</span>
        <span><i class="swatch" style="background:var(--green)"></i>GHR@5</span>
        <span><i class="swatch" style="background:var(--amber)"></i>avg_min_dist</span>
      </div>
    </section>
    <div class="grid two">
      <section>
        <h2>Episode / Subtask 2D 调试图</h2>
        <div class="controls">
          <label>Episode <select id="episodeSelect"></select></label>
          <label>Subtask <select id="subtaskSelect"></select></label>
        </div>
        <div id="map"></div>
        <div class="legend">
          <span><i class="swatch" style="background:#111827"></i>trajectory/final</span>
          <span><i class="swatch" style="background:var(--red)"></i>target</span>
          <span><i class="swatch" style="background:var(--blue)"></i>selected/planned</span>
          <span><i class="swatch" style="background:var(--green)"></i>top candidates</span>
          <span><i class="swatch" style="background:var(--purple)"></i>observations</span>
        </div>
      </section>
      <section>
        <h2>当前 Subtask 详情</h2>
        <div id="subtaskDetail"></div>
      </section>
    </div>
    <div class="grid three">
      <section><h2>观测标签 Top</h2><div id="obsLabels"></div></section>
      <section><h2>候选后端</h2><div id="candidateBackends"></div></section>
      <section><h2>Memory 事件</h2><div id="memoryEvents"></div></section>
    </div>
    <section>
      <h2>Room Memory 树</h2>
      <div class="subtle" id="roomMode"></div>
      <div class="room-grid">
        <div id="roomTree"></div>
        <div>
          <h3>Object Room Belief</h3>
          <div class="table-scroll"><table id="roomObjectTable"></table></div>
        </div>
      </div>
    </section>
    <section>
      <h2>Subtask 列表</h2>
      <div class="table-scroll"><table id="subtaskTable"></table></div>
    </section>
    <section>
      <h2>候选轮次</h2>
      <div class="table-scroll"><table id="candidateTable"></table></div>
    </section>
    <section>
      <h2>诊断 JSON</h2>
      <pre id="diagnosis"></pre>
    </section>
  </main>
  <script id="payload" type="application/json">{payload_json}</script>
  <script>
    const data = JSON.parse(document.getElementById('payload').textContent);
    const $ = (id) => document.getElementById(id);
    const fmt = (v, d=3) => (v === null || v === undefined || Number.isNaN(Number(v))) ? 'NA' : Number(v).toFixed(d);
    const esc = (s) => String(s ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
    function metric(label, value, cls='') {{
      return `<div class="metric"><div class="value ${{cls}}">${{esc(value)}}</div><div class="label">${{esc(label)}}</div></div>`;
    }}
    function renderMetrics() {{
      const s = data.summary || {{}};
      const d = data.diagnosis?.diagnostics || {{}};
      const c = data.diagnosis?.candidates || {{}};
      const p = data.perception_summary || data.diagnosis?.observations?.perception_summary || {{}};
      $('outputDir').textContent = data.output_dir || '';
      $('metrics').innerHTML = [
        metric('Episodes', s.episodes ?? data.episodes.length),
        metric('Success Rate', fmt(s.success_rate), (s.success_rate || 0) > 0 ? 'ok' : 'bad'),
        metric('SPL', fmt(s.spl)),
        metric('Dynamic Memory Acc.', fmt(s.dynamic_memory_accuracy)),
        metric('Found / Subtasks', `${{d.found ?? 0}} / ${{d.subtasks ?? 0}}`),
        metric('GHR@5', fmt(d.ghr5_rate ?? (d.subtasks ? d.ghr5 / d.subtasks : null))),
        metric('Candidate Recall', fmt(c.candidate_recall_rate)),
        metric('Detections', p.detections ?? data.diagnosis?.observations?.detections ?? 0),
        metric('Failure Bucket', data.diagnosis?.failure_bucket || 'NA', 'warn')
      ].join('');
    }}
    function seriesFromTemporal() {{
      const bySeen = data.temporal_summary?.by_seen_layout_count_before || {{}};
      return Object.keys(bySeen).map(k => ({{seen: Number(k), ...bySeen[k]}})).sort((a,b) => a.seen - b.seen);
    }}
    function renderTrend() {{
      const rows = seriesFromTemporal();
      if (!rows.length) {{ $('trend').innerHTML = '<div class="subtle">No temporal_summary found.</div>'; return; }}
      const w = 900, h = 260, pad = 36;
      const maxDist = Math.max(1, ...rows.map(r => Number(r.avg_min_dist || 0)));
      const x = (i) => pad + (w - pad*2) * (rows.length <= 1 ? 0 : i / (rows.length - 1));
      const yRate = (v) => h - pad - (h - pad*2) * Math.max(0, Math.min(1, Number(v || 0)));
      const yDist = (v) => h - pad - (h - pad*2) * Math.max(0, Math.min(1, Number(v || 0) / maxDist));
      const path = (getter, yfn) => rows.map((r,i) => `${{i?'L':'M'}}${{x(i)}},${{yfn(getter(r))}}`).join(' ');
      const ticks = rows.map((r,i) => `<text x="${{x(i)}}" y="${{h-10}}" font-size="11" text-anchor="middle">${{r.seen}}</text>`).join('');
      $('trend').innerHTML = `<svg viewBox="0 0 ${{w}} ${{h}}" role="img">
        <rect x="0" y="0" width="${{w}}" height="${{h}}" fill="#fff"/>
        <line x1="${{pad}}" y1="${{h-pad}}" x2="${{w-pad}}" y2="${{h-pad}}" stroke="#cbd5e1"/>
        <line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{h-pad}}" stroke="#cbd5e1"/>
        <path d="${{path(r=>r.found_rate, yRate)}}" fill="none" stroke="var(--blue)" stroke-width="2.5"/>
        <path d="${{path(r=>r.ghr5, yRate)}}" fill="none" stroke="var(--green)" stroke-width="2.5"/>
        <path d="${{path(r=>r.avg_min_dist, yDist)}}" fill="none" stroke="var(--amber)" stroke-width="2.5"/>
        ${{rows.map((r,i)=>`<circle cx="${{x(i)}}" cy="${{yRate(r.found_rate)}}" r="3" fill="var(--blue)"><title>seen=${{r.seen}} found=${{r.found_rate}}</title></circle>`).join('')}}
        ${{rows.map((r,i)=>`<circle cx="${{x(i)}}" cy="${{yRate(r.ghr5)}}" r="3" fill="var(--green)"><title>seen=${{r.seen}} ghr5=${{r.ghr5}}</title></circle>`).join('')}}
        ${{rows.map((r,i)=>`<circle cx="${{x(i)}}" cy="${{yDist(r.avg_min_dist)}}" r="3" fill="var(--amber)"><title>seen=${{r.seen}} avg_min_dist=${{r.avg_min_dist}}</title></circle>`).join('')}}
        ${{ticks}}
        <text x="${{pad}}" y="18" font-size="12" fill="#64748b">rate / normalized distance</text>
        <text x="${{w-pad}}" y="${{h-10}}" font-size="12" fill="#64748b" text-anchor="end">seen_layout_count_before</text>
      </svg>`;
    }}
    function barChart(id, rows, color='var(--blue)') {{
      const max = Math.max(1, ...rows.map(r => r.count || 0));
      $(id).innerHTML = rows.map(r => `<div class="bar-row"><div title="${{esc(r.key)}}">${{esc(r.key)}}</div><div class="bar-bg"><div class="bar" style="width:${{100*(r.count||0)/max}}%;background:${{color}}"></div></div><div>${{r.count}}</div></div>`).join('') || '<div class="subtle">No data.</div>';
    }}
    function renderRoomMemory() {{
      const scenes = data.scene_memory?.scenes || [];
      if (!scenes.length) {{
        $('roomMode').textContent = 'No scene_memory_final.json found.';
        $('roomTree').innerHTML = '<div class="subtle">No room memory available.</div>';
        $('roomObjectTable').innerHTML = '';
        return;
      }}
      const scene = scenes[0];
      const mode = scene.room_assignment?.mode || 'unknown';
      const grid = scene.room_assignment?.grid_size_m;
      $('roomMode').textContent = `scene=${{scene.scene_name}} tracks=${{scene.track_count}} room_assignment=${{mode}}${{grid ? ' grid=' + grid + 'm' : ''}}`;
      const rooms = scene.rooms || {{}};
      const tracks = scene.tracks || [];
      const trackById = Object.fromEntries(tracks.map(t => [t.obj_id, t]));
      const roomRows = Object.entries(rooms).sort((a,b) => {{
        const ac = Object.keys(a[1]?.object_refs || {{}}).length;
        const bc = Object.keys(b[1]?.object_refs || {{}}).length;
        return bc - ac || a[0].localeCompare(b[0]);
      }});
      $('roomTree').innerHTML = roomRows.slice(0, 40).map(([roomId, room]) => {{
        const refs = room.object_refs || {{}};
        const refRows = Object.entries(refs).sort((a,b) => Number(b[1]?.room_exist_prob || 0) - Number(a[1]?.room_exist_prob || 0)).slice(0, 12);
        return `<div class="room-node">
          <h3>${{esc(roomId)}} <span class="badge">${{Object.keys(refs).length}} objects</span></h3>
          ${{refRows.map(([objId, ref]) => {{
            const tr = trackById[objId] || {{}};
            return `<div class="obj-ref"><span>${{esc(tr.label || objId)}}<br><span class="subtle">${{esc(objId)}}</span></span><span class="belief">P=${{fmt(ref.room_exist_prob,2)}} seen=${{ref.seen_count || 0}} neg=${{ref.negative_count || 0}}</span></div>`;
          }}).join('') || '<div class="subtle">No object refs.</div>'}}
        </div>`;
      }}).join('') || '<div class="subtle">No rooms.</div>';
      const objectRows = tracks.map(t => {{
        const stats = t.cooccur_stats || {{}};
        const belief = stats.room_belief || {{}};
        const topBelief = Object.entries(belief).sort((a,b) => Number(b[1]) - Number(a[1])).slice(0, 4)
          .map(([rid, p]) => `${{rid}}:${{fmt(p,2)}}`).join(' | ');
        return {{
          obj_id: t.obj_id,
          label: t.label,
          current_room_id: stats.current_room_id || t.room_id || '',
          top_belief: topBelief,
          history_count: Array.isArray(stats.room_history) ? stats.room_history.length : 0,
        }};
      }}).filter(r => r.current_room_id || r.top_belief);
      $('roomObjectTable').innerHTML = `<thead><tr><th>label</th><th>current_room</th><th>room_belief top</th><th>history</th></tr></thead><tbody>` +
        objectRows.sort((a,b) => String(a.current_room_id).localeCompare(String(b.current_room_id)) || String(a.label).localeCompare(String(b.label)))
          .slice(0, 300)
          .map(r => `<tr><td>${{esc(r.label)}}<br><span class="subtle">${{esc(r.obj_id)}}</span></td><td>${{esc(r.current_room_id)}}</td><td>${{esc(r.top_belief)}}</td><td>${{r.history_count}}</td></tr>`).join('') +
        '</tbody>';
    }}
    function fillSelectors() {{
      const epSel = $('episodeSelect');
      if (!data.episodes.length) {{
        epSel.innerHTML = '';
        $('subtaskSelect').innerHTML = '';
        return;
      }}
      epSel.innerHTML = data.episodes.map((e,i) => `<option value="${{esc(e.episode_id)}}">${{i+1}}. state=${{e.seen_layout_count_before}} ${{esc(e.layout_id)}}</option>`).join('');
      epSel.addEventListener('change', () => {{ fillSubtaskSelect(); renderSelected(); }});
      $('subtaskSelect').addEventListener('change', renderSelected);
      fillSubtaskSelect();
    }}
    function fillSubtaskSelect() {{
      const ep = $('episodeSelect').value || data.episodes[0]?.episode_id;
      const rows = (data.groups.diagnostics_by_episode[ep] || []);
      $('subtaskSelect').innerHTML = rows.map((r,i) => `<option value="${{esc(r.subtask_id)}}">${{i+1}}. ${{esc(r.task_type)}} / ${{esc(r.target_object)}} / ${{r.found?'found':'miss'}}</option>`).join('');
    }}
    function allPointsFor(epId, subId) {{
      const pts = [];
      const run = data.run_rows.find(r => r.episode_id === epId);
      (run?.steps || []).forEach(s => {{ const p = s.position; if (p) pts.push({{kind:'trajectory', x:p.x, z:p.z, label:s.action}}); }});
      (run?.subtask_traces || []).forEach(t => {{ if (!subId || t.subtask_id === subId) {{ const p=t.final_pose; if(p) pts.push({{kind:'final', x:p.x, z:p.z, label:'final'}}); }} }});
      const diagRows = data.groups.diagnostics_by_episode[epId] || [];
      diagRows.filter(r => !subId || r.subtask_id === subId).forEach(r => {{
        if (Array.isArray(r.target_position)) pts.push({{kind:'target', x:r.target_position[0], z:r.target_position[1], label:r.target_object}});
        (r.peaks || []).slice(0, 8).forEach((p, i) => pts.push({{kind:'peak', x:p.world_x, z:p.world_z, label:`${{i+1}} ${{p.label}} ${{p.S_final ?? p.score ?? ''}}`}}));
      }});
      (data.groups.candidates_by_subtask[subId] || []).forEach(r => {{
        const sp = r.selected_candidate; if (sp) pts.push({{kind:'selected', x:sp.world_x, z:sp.world_z, label:sp.label}});
        const pp = r.planned_pose; if (pp) pts.push({{kind:'planned', x:pp.x, z:pp.z, label:'planned'}});
        const fp = r.final_pose; if (fp) pts.push({{kind:'final', x:fp.x, z:fp.z, label:'round final'}});
      }});
      const layout = data.episodes.find(e => e.episode_id === epId)?.layout_id;
      (data.groups.observations_by_layout[layout] || []).slice(0, 700).forEach(o => {{
        let p = o.pos_2d ? {{x:o.pos_2d[0], z:o.pos_2d[1]}} : (o.pos_3d ? {{x:o.pos_3d[0], z:o.pos_3d[2]}} : null);
        if (p) pts.push({{kind:'obs', x:p.x, z:p.z, label:o.label}});
      }});
      return pts.filter(p => Number.isFinite(Number(p.x)) && Number.isFinite(Number(p.z)));
    }}
    function renderMap(points) {{
      if (!points.length) {{ $('map').innerHTML = '<div class="subtle">No coordinates for this selection.</div>'; return; }}
      const xs = points.map(p => Number(p.x)), zs = points.map(p => Number(p.z));
      let minX = Math.min(...xs), maxX = Math.max(...xs), minZ = Math.min(...zs), maxZ = Math.max(...zs);
      const margin = 1.0; minX -= margin; maxX += margin; minZ -= margin; maxZ += margin;
      const w = 760, h = 520, pad = 28;
      const sx = (x) => pad + (w-pad*2) * ((Number(x)-minX) / Math.max(0.001, maxX-minX));
      const sy = (z) => h - pad - (h-pad*2) * ((Number(z)-minZ) / Math.max(0.001, maxZ-minZ));
      const color = {{trajectory:'#111827', final:'#111827', target:'var(--red)', peak:'var(--green)', selected:'var(--blue)', planned:'var(--blue)', obs:'var(--purple)'}};
      const radius = {{trajectory:3, final:5, target:7, peak:4, selected:6, planned:5, obs:2}};
      const traj = points.filter(p => p.kind === 'trajectory' || p.kind === 'final').map(p => `${{sx(p.x)}},${{sy(p.z)}}`).join(' ');
      $('map').innerHTML = `<svg viewBox="0 0 ${{w}} ${{h}}">
        <rect x="0" y="0" width="${{w}}" height="${{h}}" fill="#fff"/>
        <rect x="${{pad}}" y="${{pad}}" width="${{w-pad*2}}" height="${{h-pad*2}}" fill="#fbfdff" stroke="#d7dde5"/>
        ${{traj ? `<polyline points="${{traj}}" fill="none" stroke="#111827" stroke-width="1.8" opacity="0.65"/>` : ''}}
        ${{points.map(p => `<circle cx="${{sx(p.x)}}" cy="${{sy(p.z)}}" r="${{radius[p.kind] || 3}}" fill="${{color[p.kind] || '#64748b'}}" opacity="${{p.kind==='obs' ? 0.28 : 0.88}}"><title>${{esc(p.kind)}} ${{esc(p.label)}} (${{fmt(p.x,2)}}, ${{fmt(p.z,2)}})</title></circle>`).join('')}}
        <text x="${{pad}}" y="18" fill="#64748b" font-size="12">x/z world coordinates</text>
      </svg>`;
    }}
    function renderTables(epId, subId) {{
      const diag = (data.groups.diagnostics_by_episode[epId] || []);
      $('subtaskTable').innerHTML = `<thead><tr><th>#</th><th>target</th><th>type</th><th>alias/query</th><th>found</th><th>mra</th><th>ghr5</th><th>min/final dist</th><th>backend</th></tr></thead><tbody>` +
        diag.map((r,i) => `<tr><td>${{i+1}}</td><td>${{esc(r.target_object)}}</td><td>${{esc(r.task_type)}}</td><td>${{esc(r.query_label)}}<br><span class="subtle">${{esc((r.alias_labels||[]).join(', '))}}</span></td><td class="${{r.found?'ok':'bad'}}">${{r.found}}</td><td>${{r.mra}}</td><td>${{r.ghr5}}</td><td>${{fmt(r.min_dist)}} / ${{fmt(r.final_dist)}}</td><td>${{esc(r.top_sim_backend || '')}}</td></tr>`).join('') + '</tbody>';
      const cands = data.groups.candidates_by_subtask[subId] || [];
      $('candidateTable').innerHTML = `<thead><tr><th>round</th><th>selected</th><th>score</th><th>backend</th><th>planned</th><th>final</th><th>top candidates</th></tr></thead><tbody>` +
        cands.map(r => {{ const s = r.selected_candidate || {{}}; return `<tr><td>${{r.round}}</td><td>${{esc(s.label)}}</td><td>${{fmt(s.S_final ?? s.score)}}</td><td>${{esc(s.sim_backend)}}</td><td>${{r.planned_pose ? fmt(r.planned_pose.x,2)+','+fmt(r.planned_pose.z,2) : ''}}</td><td>${{r.final_pose ? fmt(r.final_pose.x,2)+','+fmt(r.final_pose.z,2) : ''}}</td><td>${{esc((r.top_candidates||[]).slice(0,5).map(c=>`${{c.label}}:${{fmt(c.S_final ?? c.score,2)}}`).join(' | '))}}</td></tr>`; }}).join('') + '</tbody>';
    }}
    function renderSelected() {{
      if (!data.episodes.length) {{
        $('map').innerHTML = '<div class="subtle">No episode rows found.</div>';
        $('subtaskDetail').innerHTML = '<div class="subtle">No subtask diagnostics found.</div>';
        $('subtaskTable').innerHTML = '';
        $('candidateTable').innerHTML = '';
        return;
      }}
      const epId = $('episodeSelect').value || data.episodes[0]?.episode_id;
      const subId = $('subtaskSelect').value;
      const diag = (data.groups.diagnostics_by_episode[epId] || []).find(r => r.subtask_id === subId) || {{}};
      const cands = data.groups.candidates_by_subtask[subId] || [];
      const points = allPointsFor(epId, subId);
      renderMap(points);
      renderTables(epId, subId);
      $('subtaskDetail').innerHTML = `
        <h3>${{esc(diag.target_object || 'No subtask')}}</h3>
        <div class="subtle">${{esc(subId || '')}}</div>
        <p><span class="badge">${{esc(diag.task_type)}}</span> <span class="badge">${{esc(diag.query_modality)}}</span></p>
        <table><tbody>
          <tr><th>query_label</th><td>${{esc(diag.query_label)}}</td></tr>
          <tr><th>alias_labels</th><td>${{esc((diag.alias_labels || []).join(', '))}}</td></tr>
          <tr><th>found / perception_found</th><td>${{diag.found}} / ${{diag.perception_found}}</td></tr>
          <tr><th>mra / ghr3 / ghr5</th><td>${{diag.mra}} / ${{diag.ghr3}} / ${{diag.ghr5}}</td></tr>
          <tr><th>min_dist / final_dist</th><td>${{fmt(diag.min_dist)}} / ${{fmt(diag.final_dist)}}</td></tr>
          <tr><th>fallback_count</th><td>${{diag.fallback_count ?? 0}}</td></tr>
          <tr><th>candidate rounds</th><td>${{cands.length}}</td></tr>
        </tbody></table>
        <h3>Failure feedback / notes</h3>
        <pre>${{esc(JSON.stringify({{failure_bucket: data.diagnosis?.failure_bucket, flags: data.diagnosis?.failure_flags}}, null, 2))}}</pre>
      `;
    }}
    function init() {{
      renderMetrics();
      renderTrend();
      barChart('obsLabels', data.counts.observation_labels || [], 'var(--purple)');
      barChart('candidateBackends', data.counts.candidate_backends || [], 'var(--blue)');
      barChart('memoryEvents', data.counts.memory_events || [], 'var(--green)');
      renderRoomMemory();
      fillSelectors();
      renderSelected();
      $('diagnosis').textContent = JSON.stringify(data.diagnosis || {{}}, null, 2);
    }}
    init();
  </script>
</body>
</html>"""


def _html_template(payload_json: str, title: str) -> str:
    safe_title = html.escape(title)
    template = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root { --bg:#f6f7f9; --panel:#fff; --text:#1f2933; --muted:#64748b; --line:#d7dde5; --blue:#2563eb; --green:#059669; --red:#dc2626; --amber:#b45309; --purple:#7c3aed; --cyan:#0891b2; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif; line-height:1.45; }
    header { padding:18px 24px 12px; border-bottom:1px solid var(--line); background:#fff; position:sticky; top:0; z-index:10; }
    h1 { margin:0 0 6px; font-size:22px; } h2 { margin:0 0 10px; font-size:17px; } h3 { margin:14px 0 8px; font-size:14px; }
    main { padding:18px 24px 28px; } .grid { display:grid; gap:14px; } .metrics { grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); } .two { grid-template-columns:minmax(0,1.08fr) minmax(360px,0.92fr); } .three { grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); }
    section,.panel,.metric { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; overflow:hidden; } .metric { padding:12px; }
    .metric .value { font-size:22px; font-weight:650; } .metric .label,.subtle { color:var(--muted); font-size:12px; } .subtle { font-size:13px; }
    table { width:100%; border-collapse:collapse; font-size:12px; } th,td { border-bottom:1px solid #e8edf2; padding:7px 8px; text-align:left; vertical-align:top; } th { color:#334155; background:#f8fafc; position:sticky; top:0; } tbody tr:hover { background:#f8fbff; }
    .table-scroll { max-height:420px; overflow:auto; border:1px solid #e8edf2; border-radius:6px; } select,input,button { border:1px solid var(--line); border-radius:6px; padding:7px 8px; background:#fff; color:var(--text); } select { min-width:220px; } button { cursor:pointer; }
    .controls,.layer-controls,.legend { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-bottom:12px; } .layer-controls,.legend { gap:12px; font-size:12px; color:var(--muted); margin:8px 0 12px; } .layer-controls input { vertical-align:-2px; }
    .badge { display:inline-block; border-radius:999px; padding:2px 8px; font-size:11px; background:#eef2ff; color:#3730a3; } .ok { color:var(--green); } .bad { color:var(--red); } .warn { color:var(--amber); }
    svg { width:100%; height:auto; display:block; } .swatch { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:4px; vertical-align:-1px; }
    pre { white-space:pre-wrap; word-break:break-word; background:#f8fafc; border:1px solid #e8edf2; border-radius:6px; padding:10px; font-size:12px; max-height:320px; overflow:auto; }
    .bar-row { display:grid; grid-template-columns:minmax(80px,170px) 1fr 44px; gap:8px; align-items:center; margin:5px 0; font-size:12px; } .bar-bg { height:9px; background:#edf2f7; border-radius:999px; overflow:hidden; } .bar { height:100%; background:var(--blue); }
    .room-grid { display:grid; grid-template-columns:minmax(260px,0.9fr) minmax(360px,1.1fr); gap:14px; } .room-node { border:1px solid #e8edf2; border-radius:6px; margin:8px 0; padding:9px; background:#fbfdff; } .room-node h3 { margin:0 0 7px; } .obj-ref { display:flex; justify-content:space-between; gap:10px; padding:4px 0; border-top:1px dashed #e2e8f0; font-size:12px; } .belief { color:var(--green); font-variant-numeric:tabular-nums; }
    #view3d { width:100%; height:520px; border:1px solid #d7dde5; border-radius:6px; background:#f8fafc; display:block; cursor:grab; }
    @media (max-width:960px) { .two { grid-template-columns:1fr; } main { padding:12px; } }
  </style>
</head>
<body>
  <header><h1>__TITLE__</h1><div class="subtle" id="outputDir"></div></header>
  <main class="grid">
    <div class="grid metrics" id="metrics"></div>
    <section><h2>长期趋势</h2><div id="trend"></div><div class="legend"><span><i class="swatch" style="background:var(--blue)"></i>found_rate</span><span><i class="swatch" style="background:var(--green)"></i>GHR@5</span><span><i class="swatch" style="background:var(--amber)"></i>avg_min_dist</span></div></section>
    <div class="grid two">
      <section>
        <h2>Episode / Subtask 2D 调试图</h2>
        <div class="controls"><label>Episode <select id="episodeSelect"></select></label><label>Subtask <select id="subtaskSelect"></select></label></div>
        <div class="subtle" id="geometryStatus"></div>
        <div class="layer-controls" id="layerControls">
          <label><input type="checkbox" data-layer="navmesh" checked> navmesh</label><label><input type="checkbox" data-layer="regions" checked> regions</label><label><input type="checkbox" data-layer="layout" checked> layout objects</label><label><input type="checkbox" data-layer="trajectory" checked> trajectory</label><label><input type="checkbox" data-layer="candidates" checked> candidates</label><label><input type="checkbox" data-layer="observations" checked> observations</label><label><input type="checkbox" data-layer="memory" checked> memory tracks</label>
        </div>
        <div id="map"></div>
        <div class="legend"><span><i class="swatch" style="background:#111827"></i>trajectory/final</span><span><i class="swatch" style="background:var(--red)"></i>target</span><span><i class="swatch" style="background:var(--blue)"></i>selected/planned</span><span><i class="swatch" style="background:var(--green)"></i>top candidates</span><span><i class="swatch" style="background:var(--purple)"></i>observations</span><span><i class="swatch" style="background:var(--cyan)"></i>memory/layout</span></div>
      </section>
      <section><h2>当前 Subtask 详情</h2><div id="subtaskDetail"></div></section>
    </div>
    <section><h2>3D Debug</h2><div class="controls"><button id="reset3d" type="button">重置视角</button><span class="subtle">拖拽旋转，滚轮缩放。地面层=navmesh/region，中间层=轨迹/候选/观测，上层=Scene→Room→Object 记忆树。</span></div><canvas id="view3d" width="1100" height="520"></canvas></section>
    <div class="grid three"><section><h2>观测标签 Top</h2><div id="obsLabels"></div></section><section><h2>候选后端</h2><div id="candidateBackends"></div></section><section><h2>Memory 事件</h2><div id="memoryEvents"></div></section></div>
    <section><h2>Room Memory 树</h2><div class="subtle" id="roomMode"></div><div class="room-grid"><div id="roomTree"></div><div><h3>Object Room Belief</h3><div class="table-scroll"><table id="roomObjectTable"></table></div></div></div></section>
    <section><h2>Subtask 列表</h2><div class="table-scroll"><table id="subtaskTable"></table></div></section>
    <section><h2>候选轮次</h2><div class="table-scroll"><table id="candidateTable"></table></div></section>
    <section><h2>诊断 JSON</h2><pre id="diagnosis"></pre></section>
  </main>
  <script id="payload" type="application/json">__PAYLOAD__</script>
  <script>
    const data = JSON.parse(document.getElementById('payload').textContent);
    const $ = id => document.getElementById(id);
    const fmt = (v,d=3) => (v===null||v===undefined||Number.isNaN(Number(v))) ? 'NA' : Number(v).toFixed(d);
    const esc = s => String(s ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    const layerOn = name => document.querySelector(`input[data-layer="${name}"]`)?.checked !== false;
    const colors = {trajectory:'#111827', final:'#111827', target:'#dc2626', peak:'#059669', selected:'#2563eb', planned:'#2563eb', obs:'#7c3aed', memory:'#0891b2', layout:'#0f766e'};
    let view3d = {yaw:-0.65, pitch:0.72, scale:26, dragging:false, lastX:0, lastY:0};
    function metric(label,value,cls='') { return `<div class="metric"><div class="value ${cls}">${esc(value)}</div><div class="label">${esc(label)}</div></div>`; }
    function renderMetrics() { const s=data.summary||{}, d=data.diagnosis?.diagnostics||{}, c=data.diagnosis?.candidates||{}, p=data.perception_summary||data.diagnosis?.observations?.perception_summary||{}; $('outputDir').textContent=data.output_dir||''; $('metrics').innerHTML=[metric('Episodes',s.episodes??data.episodes.length),metric('Success Rate',fmt(s.success_rate),(s.success_rate||0)>0?'ok':'bad'),metric('SPL',fmt(s.spl)),metric('Dynamic Memory Acc.',fmt(s.dynamic_memory_accuracy)),metric('Found / Subtasks',`${d.found??0} / ${d.subtasks??0}`),metric('GHR@5',fmt(d.ghr5_rate??(d.subtasks?d.ghr5/d.subtasks:null))),metric('Candidate Recall',fmt(c.candidate_recall_rate)),metric('Detections',p.detections??data.diagnosis?.observations?.detections??0),metric('Failure Bucket',data.diagnosis?.failure_bucket||'NA','warn')].join(''); }
    function renderTrend() { const bySeen=data.temporal_summary?.by_seen_layout_count_before||{}; const rows=Object.keys(bySeen).map(k=>({seen:Number(k),...bySeen[k]})).sort((a,b)=>a.seen-b.seen); if(!rows.length){$('trend').innerHTML='<div class="subtle">No temporal_summary found.</div>';return;} const w=900,h=260,pad=36,maxDist=Math.max(1,...rows.map(r=>Number(r.avg_min_dist||0))); const x=i=>pad+(w-pad*2)*(rows.length<=1?0:i/(rows.length-1)); const yRate=v=>h-pad-(h-pad*2)*Math.max(0,Math.min(1,Number(v||0))); const yDist=v=>h-pad-(h-pad*2)*Math.max(0,Math.min(1,Number(v||0)/maxDist)); const path=(getter,yfn)=>rows.map((r,i)=>`${i?'L':'M'}${x(i)},${yfn(getter(r))}`).join(' '); $('trend').innerHTML=`<svg viewBox="0 0 ${w} ${h}"><rect width="${w}" height="${h}" fill="#fff"/><line x1="${pad}" y1="${h-pad}" x2="${w-pad}" y2="${h-pad}" stroke="#cbd5e1"/><line x1="${pad}" y1="${pad}" x2="${pad}" y2="${h-pad}" stroke="#cbd5e1"/><path d="${path(r=>r.found_rate,yRate)}" fill="none" stroke="var(--blue)" stroke-width="2.5"/><path d="${path(r=>r.ghr5,yRate)}" fill="none" stroke="var(--green)" stroke-width="2.5"/><path d="${path(r=>r.avg_min_dist,yDist)}" fill="none" stroke="var(--amber)" stroke-width="2.5"/>${rows.map((r,i)=>`<text x="${x(i)}" y="${h-10}" font-size="11" text-anchor="middle">${r.seen}</text>`).join('')}</svg>`; }
    function barChart(id,rows,color='var(--blue)'){const max=Math.max(1,...rows.map(r=>r.count||0));$(id).innerHTML=rows.map(r=>`<div class="bar-row"><div title="${esc(r.key)}">${esc(r.key)}</div><div class="bar-bg"><div class="bar" style="width:${100*(r.count||0)/max}%;background:${color}"></div></div><div>${r.count}</div></div>`).join('')||'<div class="subtle">No data.</div>';}
    function renderRoomMemory(){const scenes=data.scene_memory?.scenes||[];if(!scenes.length){$('roomMode').textContent='No scene_memory_final.json found.';$('roomTree').innerHTML='<div class="subtle">No room memory available.</div>';$('roomObjectTable').innerHTML='';return;}const scene=scenes[0],rooms=scene.rooms||{},tracks=scene.tracks||[],trackById=Object.fromEntries(tracks.map(t=>[t.obj_id,t]));$('roomMode').textContent=`scene=${scene.scene_name} tracks=${scene.track_count} room_assignment=${scene.room_assignment?.mode||'unknown'}`;const roomRows=Object.entries(rooms).sort((a,b)=>Object.keys(b[1]?.object_refs||{}).length-Object.keys(a[1]?.object_refs||{}).length);$('roomTree').innerHTML=roomRows.slice(0,40).map(([roomId,room])=>{const refs=room.object_refs||{};return `<div class="room-node"><h3>${esc(roomId)} <span class="badge">${Object.keys(refs).length} objects</span></h3>${Object.entries(refs).slice(0,12).map(([objId,ref])=>{const tr=trackById[objId]||{};return `<div class="obj-ref"><span>${esc(tr.label||objId)}<br><span class="subtle">${esc(objId)}</span></span><span class="belief">P=${fmt(ref.room_exist_prob,2)} seen=${ref.seen_count||0} neg=${ref.negative_count||0}</span></div>`;}).join('')||'<div class="subtle">No object refs.</div>'}</div>`;}).join('')||'<div class="subtle">No rooms.</div>';const rows=tracks.map(t=>{const stats=t.cooccur_stats||{},belief=stats.room_belief||{};return {obj_id:t.obj_id,label:t.label,current_room_id:stats.current_room_id||t.room_id||'',top_belief:Object.entries(belief).sort((a,b)=>Number(b[1])-Number(a[1])).slice(0,4).map(([rid,p])=>`${rid}:${fmt(p,2)}`).join(' | '),history_count:Array.isArray(stats.room_history)?stats.room_history.length:0};}).filter(r=>r.current_room_id||r.top_belief);$('roomObjectTable').innerHTML='<thead><tr><th>label</th><th>current_room</th><th>room_belief top</th><th>history</th></tr></thead><tbody>'+rows.slice(0,300).map(r=>`<tr><td>${esc(r.label)}<br><span class="subtle">${esc(r.obj_id)}</span></td><td>${esc(r.current_room_id)}</td><td>${esc(r.top_belief)}</td><td>${r.history_count}</td></tr>`).join('')+'</tbody>';}
    function fillSelectors(){const epSel=$('episodeSelect');if(!data.episodes.length){epSel.innerHTML='';$('subtaskSelect').innerHTML='';return;}epSel.innerHTML=data.episodes.map((e,i)=>`<option value="${esc(e.episode_id)}">${i+1}. state=${e.seen_layout_count_before} ${esc(e.layout_id)}</option>`).join('');epSel.addEventListener('change',()=>{fillSubtaskSelect();renderSelected();});$('subtaskSelect').addEventListener('change',renderSelected);fillSubtaskSelect();}
    function fillSubtaskSelect(){const ep=$('episodeSelect').value||data.episodes[0]?.episode_id;const rows=data.groups.diagnostics_by_episode[ep]||[];$('subtaskSelect').innerHTML=rows.map((r,i)=>`<option value="${esc(r.subtask_id)}">${i+1}. ${esc(r.task_type)} / ${esc(r.target_object)} / ${r.found?'found':'miss'}</option>`).join('');}
    function sceneMemoryFor(sceneName){return (data.scene_memory?.scenes||[]).find(s=>s.scene_name===sceneName)||(data.scene_memory?.scenes||[])[0]||null;} function geometryFor(sceneName){return data.scene_geometry?.scenes?.[sceneName]||null;}
    function allPointsFor(epId,subId){
      const pts=[],run=data.run_rows.find(r=>r.episode_id===epId),ep=data.episodes.find(e=>e.episode_id===epId)||{};
      let seq=0;
      const pushPose=(pose,kind,label,extra={})=>{if(pose&&Number.isFinite(Number(pose.x))&&Number.isFinite(Number(pose.z)))pts.push({kind,x:Number(pose.x),y:Number(pose.y||0),z:Number(pose.z),label,seq:seq++,...extra});};
      const start=(run?.steps||[])[0]?.position;
      if(start)pushPose(start,'trajectory','start',{trajectory_role:'start'});
      const candsForSub=data.groups.candidates_by_subtask[subId]||[];
      let hasDetailedPath=false;
      candsForSub.forEach(r=>{
        const path=Array.isArray(r.path)?r.path:[];
        if(path.length){
          hasDetailedPath=true;
          path.forEach((p,i)=>pushPose(p,'trajectory',`round ${r.round} step ${i+1}`,{round:r.round,path_index:i,trajectory_role:'path'}));
        }else if(r.final_pose){
          pushPose(r.final_pose,'trajectory',`round ${r.round} final`,{round:r.round,trajectory_role:'round_final'});
        }
      });
      if(!hasDetailedPath&&!candsForSub.length){
        (run?.steps||[]).slice(1).forEach(s=>pushPose(s.position,'trajectory',s.action||'step',{trajectory_role:'run_step'}));
      }
      (run?.subtask_traces||[]).forEach(t=>{if(!subId||t.subtask_id===subId){const p=t.final_pose;if(p)pts.push({kind:'final',x:p.x,y:p.y||0,z:p.z,label:'final'});}});
      (data.groups.diagnostics_by_episode[epId]||[]).filter(r=>!subId||r.subtask_id===subId).forEach(r=>{if(Array.isArray(r.target_position))pts.push({kind:'target',x:r.target_position[0],y:0,z:r.target_position[1],label:r.target_object});(r.peaks||[]).slice(0,8).forEach((p,i)=>pts.push({kind:'peak',x:p.world_x,y:0.12+i*0.03,z:p.world_z,label:`${i+1} ${p.label}`}));});
      candsForSub.forEach(r=>{const sp=r.selected_candidate;if(sp)pts.push({kind:'selected',x:sp.world_x,y:0.2,z:sp.world_z,label:sp.label});(r.top_candidates||[]).slice(0,5).forEach((c,i)=>pts.push({kind:'peak',x:c.world_x,y:0.1+i*0.03,z:c.world_z,label:`top${i+1} ${c.label}`}));const pp=r.planned_pose;if(pp)pts.push({kind:'planned',x:pp.x,y:pp.y||0,z:pp.z,label:'planned'});const fp=r.final_pose;if(fp)pts.push({kind:'final',x:fp.x,y:fp.y||0,z:fp.z,label:'round final'});});
      (data.groups.observations_by_layout[ep.layout_id]||[]).slice(0,900).forEach(o=>{const p=o.pos_2d?{x:o.pos_2d[0],y:0.08,z:o.pos_2d[1]}:(o.pos_3d?{x:o.pos_3d[0],y:o.pos_3d[1]||0.08,z:o.pos_3d[2]}:null);if(p)pts.push({kind:'obs',...p,label:o.label});});
      (sceneMemoryFor(ep.scene_name)?.tracks||[]).forEach(t=>{const p=Array.isArray(t.pos_2d)?{x:t.pos_2d[0],y:0.16,z:t.pos_2d[1]}:(Array.isArray(t.pos_3d)?{x:t.pos_3d[0],y:t.pos_3d[1]||0.16,z:t.pos_3d[2]}:null);if(p)pts.push({kind:'memory',...p,label:t.label});});
      return pts.filter(p=>Number.isFinite(Number(p.x))&&Number.isFinite(Number(p.z)));
    }
    function boundsFor(points,geom){const b=geom?.bounds;if(b&&Number.isFinite(Number(b.min_x)))return{minX:Number(b.min_x),maxX:Number(b.max_x),minZ:Number(b.min_z),maxZ:Number(b.max_z)};const xs=points.map(p=>Number(p.x)),zs=points.map(p=>Number(p.z));if(!xs.length)return{minX:-5,maxX:5,minZ:-5,maxZ:5};return{minX:Math.min(...xs),maxX:Math.max(...xs),minZ:Math.min(...zs),maxZ:Math.max(...zs)};}
    function renderMap(points,geom){if(!points.length&&!geom){$('map').innerHTML='<div class="subtle">No coordinates for this selection.</div>';return;}let {minX,maxX,minZ,maxZ}=boundsFor(points,geom);minX-=1;maxX+=1;minZ-=1;maxZ+=1;const w=760,h=520,pad=28,sx=x=>pad+(w-pad*2)*((Number(x)-minX)/Math.max(0.001,maxX-minX)),sy=z=>h-pad-(h-pad*2)*((Number(z)-minZ)/Math.max(0.001,maxZ-minZ));const contours=geom?.navmesh_contours||[],regions=geom?.semantic_regions||[],layoutObjects=geom?.layout_objects||[],navPts=geom?.navigable_points||[];$('geometryStatus').textContent=geom?`geometry: regions=${regions.length}, navmesh_segments=${contours.length}, navigable_points=${navPts.length}`:'No scene_geometry found; using point-only fallback.';const navSvg=layerOn('navmesh')?navPts.slice(0,1200).map(p=>`<circle cx="${sx(p[0])}" cy="${sy(p[2])}" r="1" fill="#cbd5e1" opacity="0.28"/>`).join('')+contours.slice(0,12000).map(seg=>`<line x1="${sx(seg[0][0])}" y1="${sy(seg[0][1])}" x2="${sx(seg[1][0])}" y2="${sy(seg[1][1])}" stroke="#94a3b8" stroke-width="0.9" opacity="0.55"/>`).join(''):'';const regionSvg=layerOn('regions')?regions.map(r=>{const pts=(r.polygon||[]).map(p=>`${sx(p[0])},${sy(p[1])}`).join(' '),c=r.center||[];return `<g><polygon points="${pts}" fill="#bfdbfe" opacity="0.11" stroke="#2563eb" stroke-width="1.1" stroke-dasharray="4 3"><title>${esc(r.room_id||r.region_id)} source=${esc(r.source)}</title></polygon>${c.length>=3?`<text x="${sx(c[0])}" y="${sy(c[2])}" font-size="10" fill="#1d4ed8">R${esc(r.region_id)}</text>`:''}</g>`;}).join(''):'';const layoutSvg=layerOn('layout')?layoutObjects.map(o=>`<rect x="${sx(o.position[0])-3}" y="${sy(o.position[2])-3}" width="6" height="6" fill="${colors.layout}" opacity="0.75"><title>layout ${esc(o.name)} region=${esc(o.sampled_region_id)}</title></rect>`).join(''):'';const drawPts=points.filter(p=>p.kind==='obs'?layerOn('observations'):p.kind==='memory'?layerOn('memory'):['peak','selected','planned'].includes(p.kind)?layerOn('candidates'):['trajectory','final'].includes(p.kind)?layerOn('trajectory'):true);const trajPts=points.filter(p=>p.kind==='trajectory').sort((a,b)=>(a.seq??0)-(b.seq??0));const trajSegments=layerOn('trajectory')?trajPts.slice(1).map((p,i)=>{const a=trajPts[i],len=Math.hypot(sx(p.x)-sx(a.x),sy(p.z)-sy(a.z));if(len<2)return'';return `<line x1="${sx(a.x)}" y1="${sy(a.z)}" x2="${sx(p.x)}" y2="${sy(p.z)}" stroke="#111827" stroke-width="1.9" opacity="0.72" marker-end="url(#trajArrow)"><title>${esc(a.label)} → ${esc(p.label)}</title></line>`;}).join(''):'';const trajLabels=layerOn('trajectory')&&trajPts.length?`<circle cx="${sx(trajPts[0].x)}" cy="${sy(trajPts[0].z)}" r="5" fill="#f59e0b" stroke="#111827"><title>trajectory start</title></circle><text x="${sx(trajPts[0].x)+7}" y="${sy(trajPts[0].z)-7}" font-size="10" fill="#92400e">start</text><circle cx="${sx(trajPts[trajPts.length-1].x)}" cy="${sy(trajPts[trajPts.length-1].z)}" r="5" fill="#111827"><title>trajectory end</title></circle><text x="${sx(trajPts[trajPts.length-1].x)+7}" y="${sy(trajPts[trajPts.length-1].z)-7}" font-size="10" fill="#111827">end</text>`:'';const radius={trajectory:2.4,final:5,target:7,peak:4,selected:6,planned:5,obs:2,memory:4};$('map').innerHTML=`<svg viewBox="0 0 ${w} ${h}"><defs><marker id="trajArrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 Z" fill="#111827"/></marker></defs><rect width="${w}" height="${h}" fill="#fff"/><rect x="${pad}" y="${pad}" width="${w-pad*2}" height="${h-pad*2}" fill="#fbfdff" stroke="#d7dde5"/>${navSvg}${regionSvg}${layoutSvg}${trajSegments}${trajLabels}${drawPts.map(p=>`<circle cx="${sx(p.x)}" cy="${sy(p.z)}" r="${radius[p.kind]||3}" fill="${colors[p.kind]||'#64748b'}" opacity="${p.kind==='obs'?0.28:p.kind==='trajectory'?0.58:0.9}"><title>${esc(p.kind)} ${esc(p.label)} (${fmt(p.x,2)}, ${fmt(p.z,2)})</title></circle>`).join('')}<text x="${pad}" y="18" fill="#64748b" font-size="12">x/z world coordinates; arrows show chronological trajectory direction</text></svg>`;}
    function renderTables(epId,subId){const diag=data.groups.diagnostics_by_episode[epId]||[];$('subtaskTable').innerHTML='<thead><tr><th>#</th><th>target</th><th>type</th><th>alias/query</th><th>found</th><th>mra</th><th>ghr5</th><th>min/final dist</th><th>backend</th></tr></thead><tbody>'+diag.map((r,i)=>`<tr><td>${i+1}</td><td>${esc(r.target_object)}</td><td>${esc(r.task_type)}</td><td>${esc(r.query_label)}<br><span class="subtle">${esc((r.alias_labels||[]).join(', '))}</span></td><td class="${r.found?'ok':'bad'}">${r.found}</td><td>${r.mra}</td><td>${r.ghr5}</td><td>${fmt(r.min_dist)} / ${fmt(r.final_dist)}</td><td>${esc(r.top_sim_backend||'')}</td></tr>`).join('')+'</tbody>';const cands=data.groups.candidates_by_subtask[subId]||[];$('candidateTable').innerHTML='<thead><tr><th>round</th><th>selected</th><th>score</th><th>backend</th><th>planned</th><th>final</th><th>fallback</th><th>top candidates</th></tr></thead><tbody>'+cands.map(r=>{const s=r.selected_candidate||{};return `<tr><td>${r.round}</td><td>${esc(s.label)}</td><td>${fmt(s.S_final??s.score)}</td><td>${esc(s.sim_backend)}</td><td>${r.planned_pose?fmt(r.planned_pose.x,2)+','+fmt(r.planned_pose.z,2):''}</td><td>${r.final_pose?fmt(r.final_pose.x,2)+','+fmt(r.final_pose.z,2):''}</td><td>${esc(r.fallback_reason||'')}</td><td>${esc((r.top_candidates||[]).slice(0,5).map(c=>`${c.label}:${fmt(c.S_final??c.score,2)}`).join(' | '))}</td></tr>`;}).join('')+'</tbody>';}
    function selectedContext(){const epId=$('episodeSelect').value||data.episodes[0]?.episode_id,subId=$('subtaskSelect').value,ep=data.episodes.find(e=>e.episode_id===epId)||{},geom=geometryFor(ep.scene_name),points=allPointsFor(epId,subId);return{epId,subId,ep,geom,points};}
    function project3d(p,center,canvas){const x0=Number(p.x)-center.x,y0=Number(p.y||0)-center.y,z0=Number(p.z)-center.z,cy=Math.cos(view3d.yaw),sy=Math.sin(view3d.yaw),cp=Math.cos(view3d.pitch),sp=Math.sin(view3d.pitch),x1=x0*cy-z0*sy,z1=x0*sy+z0*cy,y1=y0*cp-z1*sp,depth=y0*sp+z1*cp,scale=view3d.scale*(1/Math.max(0.25,1+depth*0.025));return{x:canvas.width/2+x1*scale,y:canvas.height*0.58-y1*scale,depth};}
    function draw3Line(ctx,a,b,center,canvas,stroke,width=1,alpha=1){const pa=project3d({x:a[0],y:a[1]||0,z:a[2]},center,canvas),pb=project3d({x:b[0],y:b[1]||0,z:b[2]},center,canvas);ctx.globalAlpha=alpha;ctx.strokeStyle=stroke;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(pa.x,pa.y);ctx.lineTo(pb.x,pb.y);ctx.stroke();ctx.globalAlpha=1;}
    function draw3Arrow(ctx,a,b,center,canvas,stroke,width=1,alpha=1){const pa=project3d({x:a[0],y:a[1]||0,z:a[2]},center,canvas),pb=project3d({x:b[0],y:b[1]||0,z:b[2]},center,canvas),dx=pb.x-pa.x,dy=pb.y-pa.y,len=Math.hypot(dx,dy);if(len<3)return;draw3Line(ctx,a,b,center,canvas,stroke,width,alpha);const ux=dx/len,uy=dy/len,size=7*(window.devicePixelRatio||1);ctx.globalAlpha=alpha;ctx.fillStyle=stroke;ctx.beginPath();ctx.moveTo(pb.x,pb.y);ctx.lineTo(pb.x-ux*size-uy*size*0.55,pb.y-uy*size+ux*size*0.55);ctx.lineTo(pb.x-ux*size+uy*size*0.55,pb.y-uy*size-ux*size*0.55);ctx.closePath();ctx.fill();ctx.globalAlpha=1;}
    function render3D(points,geom){const canvas=$('view3d');if(!canvas)return;const dpr=window.devicePixelRatio||1,rect=canvas.getBoundingClientRect();if(rect.width&&rect.height&&(canvas.width!==Math.round(rect.width*dpr)||canvas.height!==Math.round(rect.height*dpr))){canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);}const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#f8fafc';ctx.fillRect(0,0,canvas.width,canvas.height);const b=boundsFor(points,geom),center={x:(b.minX+b.maxX)/2,y:1.35,z:(b.minZ+b.maxZ)/2};const L={ground:0,regionTop:0.55,traj:0.28,obs:0.78,cand:1.18,memory:1.58,room:2.45,scene:3.35};const pt=(x,y,z)=>({x:x,y:y,z:z});const drawLabel=(text,p,fill='#475569')=>{const q=project3d(p,center,canvas);ctx.globalAlpha=0.9;ctx.fillStyle=fill;ctx.font=`${10*dpr}px sans-serif`;ctx.fillText(String(text).slice(0,22),q.x+5*dpr,q.y-4*dpr);ctx.globalAlpha=1;};const drawDot=(p,fill,r=4,alpha=0.9)=>{const q=project3d(p,center,canvas);ctx.globalAlpha=alpha;ctx.fillStyle=fill;ctx.beginPath();ctx.arc(q.x,q.y,r*dpr,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;};if(layerOn('navmesh'))(geom?.navmesh_contours||[]).slice(0,12000).forEach(seg=>draw3Line(ctx,[seg[0][0],L.ground,seg[0][1]],[seg[1][0],L.ground,seg[1][1]],center,canvas,'#94a3b8',1,0.35));const regions=(geom?.semantic_regions||[]).slice(0,60);const regionNodes=[];if(layerOn('regions'))regions.forEach(r=>{const poly=r.polygon||[],rid=`R${r.region_id}`;if(poly.length){for(let i=0;i<poly.length;i++){const a=poly[i],bb=poly[(i+1)%poly.length];draw3Line(ctx,[a[0],L.ground,a[1]],[bb[0],L.ground,bb[1]],center,canvas,'#2563eb',1.1,0.62);draw3Line(ctx,[a[0],L.regionTop,a[1]],[bb[0],L.regionTop,bb[1]],center,canvas,'#60a5fa',1,0.35);draw3Line(ctx,[a[0],L.ground,a[1]],[a[0],L.regionTop,a[1]],center,canvas,'#93c5fd',0.8,0.28);}}const c=r.center&&r.center.length>=3?pt(r.center[0],L.room,r.center[2]):(poly.length?pt(poly.reduce((s,p)=>s+p[0],0)/poly.length,L.room,poly.reduce((s,p)=>s+p[1],0)/poly.length):null);if(c){regionNodes.push({id:rid,label:rid,pos:c});draw3Line(ctx,[c.x,L.regionTop,c.z],[c.x,L.room,c.z],center,canvas,'#2563eb',1,0.38);drawDot(c,'#2563eb',5,0.86);drawLabel(rid,c,'#1d4ed8');}});const sceneNode=pt(center.x,L.scene,center.z);drawDot(sceneNode,'#111827',7,0.9);drawLabel('Scene',sceneNode,'#111827');regionNodes.slice(0,36).forEach(n=>draw3Line(ctx,[sceneNode.x,sceneNode.y,sceneNode.z],[n.pos.x,n.pos.y,n.pos.z],center,canvas,'#64748b',1,0.28));if(layerOn('layout'))(geom?.layout_objects||[]).slice(0,160).forEach(o=>{const p=o.position||[];if(p.length>=3){const top=pt(p[0],L.obs,p[2]);draw3Line(ctx,[p[0],L.ground,p[2]],[top.x,top.y,top.z],center,canvas,colors.layout,1,0.42);drawDot(top,colors.layout,3,0.72);}});const drawPts=points.filter(p=>p.kind==='obs'?layerOn('observations'):p.kind==='memory'?layerOn('memory'):['peak','selected','planned'].includes(p.kind)?layerOn('candidates'):['trajectory','final'].includes(p.kind)?layerOn('trajectory'):true).map(p=>({...p,y:p.kind==='obs'?L.obs:p.kind==='memory'?L.memory:['peak','selected','planned'].includes(p.kind)?L.cand:p.kind==='target'?L.cand+0.35:L.traj}));const traj=drawPts.filter(p=>p.kind==='trajectory'||p.kind==='final');for(let i=1;i<traj.length;i++)draw3Line(ctx,[traj[i-1].x,traj[i-1].y,traj[i-1].z],[traj[i].x,traj[i].y,traj[i].z],center,canvas,'#111827',2,0.72);drawPts.sort((a,b)=>project3d(b,center,canvas).depth-project3d(a,center,canvas).depth).forEach(p=>{drawDot(p,colors[p.kind]||'#64748b',p.kind==='target'?7:p.kind==='obs'?2.2:p.kind==='memory'?4.5:4.5,p.kind==='obs'?0.34:0.9);if(p.kind==='target'||p.kind==='selected')drawLabel(p.label,p,colors[p.kind]||'#475569');});const memoryPts=drawPts.filter(p=>p.kind==='memory').slice(0,160);memoryPts.forEach((p,i)=>{let nearest=null,best=Infinity;regionNodes.forEach(r=>{const d=(r.pos.x-p.x)*(r.pos.x-p.x)+(r.pos.z-p.z)*(r.pos.z-p.z);if(d<best){best=d;nearest=r;}});if(nearest){draw3Line(ctx,[nearest.pos.x,nearest.pos.y,nearest.pos.z],[p.x,p.y,p.z],center,canvas,'#0891b2',0.9,0.26);}if(i<45)drawLabel(p.label,p,'#0e7490');});ctx.fillStyle='#64748b';ctx.font=`${12*dpr}px sans-serif`;ctx.fillText('layered 3D: ground(navmesh/regions) → evidence(trajectory/candidates/observations) → memory tree(Scene→Room→Object)',12*dpr,20*dpr);}
    function render3D(points,geom){
      const canvas=$('view3d');if(!canvas)return;
      const dpr=window.devicePixelRatio||1,rect=canvas.getBoundingClientRect();
      if(rect.width&&rect.height&&(canvas.width!==Math.round(rect.width*dpr)||canvas.height!==Math.round(rect.height*dpr))){canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);}
      const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#f8fafc';ctx.fillRect(0,0,canvas.width,canvas.height);
      const b=boundsFor(points,geom),center={x:(b.minX+b.maxX)/2,y:1.35,z:(b.minZ+b.maxZ)/2};
      const L={ground:0,regionTop:0.55,traj:0.28,obs:0.78,cand:1.18,memory:1.58,room:2.45,scene:3.35};
      const pt=(x,y,z)=>({x:x,y:y,z:z});
      const drawLabel=(text,p,fill='#475569')=>{const q=project3d(p,center,canvas);ctx.globalAlpha=0.9;ctx.fillStyle=fill;ctx.font=`${10*dpr}px sans-serif`;ctx.fillText(String(text).slice(0,22),q.x+5*dpr,q.y-4*dpr);ctx.globalAlpha=1;};
      const drawDot=(p,fill,r=4,alpha=0.9)=>{const q=project3d(p,center,canvas);ctx.globalAlpha=alpha;ctx.fillStyle=fill;ctx.beginPath();ctx.arc(q.x,q.y,r*dpr,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;};
      if(layerOn('navmesh'))(geom?.navmesh_contours||[]).slice(0,12000).forEach(seg=>draw3Line(ctx,[seg[0][0],L.ground,seg[0][1]],[seg[1][0],L.ground,seg[1][1]],center,canvas,'#94a3b8',1,0.35));
      const regions=(geom?.semantic_regions||[]).slice(0,60),regionNodes=[];
      if(layerOn('regions'))regions.forEach(r=>{const poly=r.polygon||[],rid=`R${r.region_id}`;if(poly.length){for(let i=0;i<poly.length;i++){const a=poly[i],bb=poly[(i+1)%poly.length];draw3Line(ctx,[a[0],L.ground,a[1]],[bb[0],L.ground,bb[1]],center,canvas,'#2563eb',1.1,0.62);draw3Line(ctx,[a[0],L.regionTop,a[1]],[bb[0],L.regionTop,bb[1]],center,canvas,'#60a5fa',1,0.35);draw3Line(ctx,[a[0],L.ground,a[1]],[a[0],L.regionTop,a[1]],center,canvas,'#93c5fd',0.8,0.28);}}const c=r.center&&r.center.length>=3?pt(r.center[0],L.room,r.center[2]):(poly.length?pt(poly.reduce((s,p)=>s+p[0],0)/poly.length,L.room,poly.reduce((s,p)=>s+p[1],0)/poly.length):null);if(c){regionNodes.push({id:rid,label:rid,pos:c});draw3Line(ctx,[c.x,L.regionTop,c.z],[c.x,L.room,c.z],center,canvas,'#2563eb',1,0.38);drawDot(c,'#2563eb',5,0.86);drawLabel(rid,c,'#1d4ed8');}});
      const sceneNode=pt(center.x,L.scene,center.z);drawDot(sceneNode,'#111827',7,0.9);drawLabel('Scene',sceneNode,'#111827');regionNodes.slice(0,36).forEach(n=>draw3Line(ctx,[sceneNode.x,sceneNode.y,sceneNode.z],[n.pos.x,n.pos.y,n.pos.z],center,canvas,'#64748b',1,0.28));
      if(layerOn('layout'))(geom?.layout_objects||[]).slice(0,160).forEach(o=>{const p=o.position||[];if(p.length>=3){const top=pt(p[0],L.obs,p[2]);draw3Line(ctx,[p[0],L.ground,p[2]],[top.x,top.y,top.z],center,canvas,colors.layout,1,0.42);drawDot(top,colors.layout,3,0.72);}});
      const drawPts=points.filter(p=>p.kind==='obs'?layerOn('observations'):p.kind==='memory'?layerOn('memory'):['peak','selected','planned'].includes(p.kind)?layerOn('candidates'):['trajectory','final'].includes(p.kind)?layerOn('trajectory'):true).map(p=>({...p,y:p.kind==='obs'?L.obs:p.kind==='memory'?L.memory:['peak','selected','planned'].includes(p.kind)?L.cand:p.kind==='target'?L.cand+0.35:L.traj}));
      const traj=drawPts.filter(p=>p.kind==='trajectory').sort((a,b)=>(a.seq??0)-(b.seq??0));
      for(let i=1;i<traj.length;i++)draw3Arrow(ctx,[traj[i-1].x,traj[i-1].y,traj[i-1].z],[traj[i].x,traj[i].y,traj[i].z],center,canvas,'#111827',2,0.72);
      drawPts.sort((a,b)=>project3d(b,center,canvas).depth-project3d(a,center,canvas).depth).forEach(p=>{drawDot(p,colors[p.kind]||'#64748b',p.kind==='target'?7:p.kind==='obs'?2.2:p.kind==='memory'?4.5:4.5,p.kind==='obs'?0.34:p.kind==='trajectory'?0.58:0.9);if(p.kind==='target'||p.kind==='selected')drawLabel(p.label,p,colors[p.kind]||'#475569');});
      if(traj.length){drawLabel('start',traj[0],'#92400e');drawLabel('end',traj[traj.length-1],'#111827');}
      const memoryPts=drawPts.filter(p=>p.kind==='memory').slice(0,160);
      memoryPts.forEach((p,i)=>{let nearest=null,best=Infinity;regionNodes.forEach(r=>{const d=(r.pos.x-p.x)*(r.pos.x-p.x)+(r.pos.z-p.z)*(r.pos.z-p.z);if(d<best){best=d;nearest=r;}});if(nearest){draw3Line(ctx,[nearest.pos.x,nearest.pos.y,nearest.pos.z],[p.x,p.y,p.z],center,canvas,'#0891b2',0.9,0.26);}if(i<45)drawLabel(p.label,p,'#0e7490');});
      ctx.fillStyle='#64748b';ctx.font=`${12*dpr}px sans-serif`;ctx.fillText('layered 3D: ground(navmesh/regions) → evidence(trajectory arrows/candidates/observations) → memory tree(Scene→Room→Object)',12*dpr,20*dpr);
    }
    function renderSelected(){if(!data.episodes.length){$('map').innerHTML='<div class="subtle">No episode rows found.</div>';$('subtaskDetail').innerHTML='<div class="subtle">No subtask diagnostics found.</div>';$('subtaskTable').innerHTML='';$('candidateTable').innerHTML='';return;}const {epId,subId,geom,points}=selectedContext(),diag=(data.groups.diagnostics_by_episode[epId]||[]).find(r=>r.subtask_id===subId)||{},cands=data.groups.candidates_by_subtask[subId]||[];renderMap(points,geom);render3D(points,geom);renderTables(epId,subId);const fallback=cands.map(c=>c.fallback_reason).filter(Boolean).join(', ')||'none';$('subtaskDetail').innerHTML=`<h3>${esc(diag.target_object||'No subtask')}</h3><div class="subtle">${esc(subId||'')}</div><p><span class="badge">${esc(diag.task_type)}</span> <span class="badge">${esc(diag.query_modality)}</span></p><table><tbody><tr><th>query_label</th><td>${esc(diag.query_label)}</td></tr><tr><th>alias_labels</th><td>${esc((diag.alias_labels||[]).join(', '))}</td></tr><tr><th>found / perception_found</th><td>${diag.found} / ${diag.perception_found}</td></tr><tr><th>mra / ghr3 / ghr5</th><td>${diag.mra} / ${diag.ghr3} / ${diag.ghr5}</td></tr><tr><th>min_dist / final_dist</th><td>${fmt(diag.min_dist)} / ${fmt(diag.final_dist)}</td></tr><tr><th>fallback_count</th><td>${diag.fallback_count??0}</td></tr><tr><th>fallback_reason</th><td>${esc(fallback)}</td></tr><tr><th>candidate rounds</th><td>${cands.length}</td></tr></tbody></table><h3>Failure feedback / notes</h3><pre>${esc(JSON.stringify({failure_bucket:data.diagnosis?.failure_bucket,flags:data.diagnosis?.failure_flags},null,2))}</pre>`;}
    function install3DControls(){const canvas=$('view3d');if(!canvas)return;canvas.addEventListener('mousedown',e=>{view3d.dragging=true;view3d.lastX=e.clientX;view3d.lastY=e.clientY;canvas.style.cursor='grabbing';});window.addEventListener('mouseup',()=>{view3d.dragging=false;canvas.style.cursor='grab';});window.addEventListener('mousemove',e=>{if(!view3d.dragging)return;view3d.yaw+=(e.clientX-view3d.lastX)*0.008;view3d.pitch=Math.max(0.15,Math.min(1.35,view3d.pitch+(e.clientY-view3d.lastY)*0.006));view3d.lastX=e.clientX;view3d.lastY=e.clientY;const {points,geom}=selectedContext();render3D(points,geom);});canvas.addEventListener('wheel',e=>{e.preventDefault();view3d.scale=Math.max(5,Math.min(120,view3d.scale*(e.deltaY>0?0.9:1.1)));const {points,geom}=selectedContext();render3D(points,geom);},{passive:false});$('reset3d').addEventListener('click',()=>{view3d={yaw:-0.65,pitch:0.72,scale:26,dragging:false,lastX:0,lastY:0};const {points,geom}=selectedContext();render3D(points,geom);});}
    function init(){renderMetrics();renderTrend();barChart('obsLabels',data.counts.observation_labels||[],'var(--purple)');barChart('candidateBackends',data.counts.candidate_backends||[],'var(--blue)');barChart('memoryEvents',data.counts.memory_events||[],'var(--green)');renderRoomMemory();fillSelectors();document.querySelectorAll('#layerControls input').forEach(el=>el.addEventListener('change',renderSelected));install3DControls();renderSelected();$('diagnosis').textContent=JSON.stringify(data.diagnosis||{},null,2);}
    init();
  </script>
</body>
</html>"""
    return template.replace("__TITLE__", safe_title).replace("__PAYLOAD__", payload_json)


def write_visualization(payload: Dict[str, Any], output_path: Path) -> None:
    title = "Temporal Habitat Loop Visual Debug"
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_html_template(payload_json, title), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an HTML visual debugger for run_temporal_habitat_loop outputs.")
    parser.add_argument("--output-dir", required=True, help="Directory produced by run_temporal_habitat_loop.")
    parser.add_argument("--html", default="", help="Output HTML path. Defaults to <output-dir>/visual_debug/index.html.")
    parser.add_argument("--write-json", default="", help="Optional compact JSON payload path.")
    parser.add_argument("--max-observations", type=int, default=20000, help="Maximum observation records embedded in the HTML.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = as_path(args.output_dir)
    payload = build_visual_payload(output_dir, max_observations=int(args.max_observations))
    html_path = as_path(args.html) if args.html else output_dir / "visual_debug" / "index.html"
    write_visualization(payload, html_path)
    json_path = as_path(args.write_json) if args.write_json else output_dir / "visual_debug" / "visual_payload.json"
    write_json(json_path, payload)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "html": str(html_path),
                "json": str(json_path),
                "episodes": len(payload.get("episodes", [])),
                "subtasks": len(payload.get("diagnostics", [])),
                "candidate_rounds": len(payload.get("candidates", [])),
                "observations_embedded": len(payload.get("observations", [])),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
