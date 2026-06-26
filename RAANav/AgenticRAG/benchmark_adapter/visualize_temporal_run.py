from __future__ import annotations

import argparse
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .common import as_path, write_json
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


def build_visual_payload(output_dir: Path, *, max_observations: int = 20000) -> Dict[str, Any]:
    output_dir = as_path(output_dir)
    run_rows = _read_jsonl(output_dir / "run.jsonl")
    diagnostics = _read_jsonl(output_dir / "memory_diagnostics.jsonl")
    candidates = _read_jsonl(output_dir / "candidate_traces.jsonl")
    observations = _read_jsonl(output_dir / "observation_traces.jsonl", max_records=max_observations)
    memory_updates = _read_jsonl(output_dir / "memory_updates.jsonl")
    negative_feedback = _read_jsonl(output_dir / "negative_feedback.jsonl")
    layout_transitions = _read_jsonl(output_dir / "layout_transitions.jsonl")
    summary = _read_json_optional(output_dir / "summary.json") or {}
    temporal_summary = _read_json_optional(output_dir / "temporal_summary.json") or {}
    perception_summary = _read_json_optional(output_dir / "perception_summary.json") or {}
    by_task_type = _read_json_optional(output_dir / "by_task_type.json") or {}
    diagnosis = diagnose(output_dir)

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
        "counts": {
            "observation_labels": _count_by(observations, "label", limit=30),
            "memory_events": _count_by(memory_updates, "event", limit=20),
            "negative_feedback_labels": _count_by(negative_feedback, "label", limit=30),
            "candidate_backends": _candidate_backend_counts(candidates),
            "targets": _count_by(diagnostics, "target_object", limit=30),
            "task_types": _count_by(diagnostics, "task_type", limit=10),
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
    function fillSelectors() {{
      const epSel = $('episodeSelect');
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
      fillSelectors();
      renderSelected();
      $('diagnosis').textContent = JSON.stringify(data.diagnosis || {{}}, null, 2);
    }}
    init();
  </script>
</body>
</html>"""


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
