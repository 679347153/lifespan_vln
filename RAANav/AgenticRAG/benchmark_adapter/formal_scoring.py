from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

from RAANav.AgenticRAG.semantic_map import Floor, Object

from .common import euclidean_2d
from .episode_to_queries import QuerySpec, query_label_matches, query_label_similarity
from .image_goal_index import ImageGoalIndex


@dataclass
class FusionParams:
    beta1: float = 0.65
    beta2: float = 0.35
    time_lambda: float = 0.231
    kappa: float = 1.0
    agent_k: float = 0.35
    agent_w_min: float = 0.15
    epsilon: float = 1e-8
    neg_gamma_min: float = 0.1
    neg_gamma_span: float = 0.25
    use_agent: bool = True
    use_time_decay: bool = True
    use_cooccur: bool = True
    use_clip: bool = True
    label_only: bool = False


@dataclass
class CandidateScore:
    obj_id: str
    label: str
    world_x: float
    world_z: float
    R_sim: float
    R_cfd: float
    G_ts: float
    S_rag: float
    S_agent: float
    w_agent: float
    w_rag: float
    S_final: float
    stability: float
    exist_prob: float
    negative_feedback_count: int
    stability_bucket: str
    query_modality: str
    sim_backend: str
    reason: str


def iter_objects(floors: Iterable[Floor]) -> Iterable[Object]:
    for floor in floors:
        for room in floor.rooms:
            for obj in room.objects:
                yield obj


def _parse_time(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        except Exception:
            return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None
    return None


def query_time_for_state(state_index: int) -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=int(state_index))


def delta_days(obj: Object, state_index: int) -> float:
    last = _parse_time(obj.last_update_time)
    if last is None:
        return 0.0
    return max(0.0, (query_time_for_state(state_index) - last).total_seconds() / 86400.0)


def relation_confidence(obj: Object, query: QuerySpec) -> float:
    if obj.obj_id == query.target_object_id:
        return 1.0
    rel = (obj.R_objs or {}).get(query.target_object_id)
    if isinstance(rel, dict):
        try:
            return max(0.0, min(1.0, float(rel.get("Rcfd", 0.0))))
        except Exception:
            pass
    stats = obj.cooccur_stats or {}
    score = 0.0
    if query.sampled_region_id is not None and str(stats.get("sampled_region_id")) == str(query.sampled_region_id):
        score += 0.25
    if query.target_instance_id is not None and str(stats.get("target_instance_id")) == str(query.target_instance_id):
        score += 0.35
    if query.target_instance_id is not None and str(stats.get("assigned_target_instance_id")) == str(query.target_instance_id):
        score += 0.45
    return max(0.0, min(1.0, score))


def agent_prior(obj: Object, query: QuerySpec) -> float:
    score = 0.0
    if obj.obj_id == query.target_object_id:
        score += 0.8
    if query_label_matches(obj.label, query):
        score += 0.65
    overlap = query_label_similarity(obj.label, query)
    score += 0.2 * overlap
    stats = obj.cooccur_stats or {}
    if query.sampled_region_id is not None and str(stats.get("sampled_region_id")) == str(query.sampled_region_id):
        score += 0.35
    if query.target_instance_id is not None and str(stats.get("target_instance_id")) == str(query.target_instance_id):
        score += 0.45
    if query.language_prompt:
        # Language goals explicitly mention receptacle/region, so the metadata
        # prior is a meaningful Agent-channel signal.
        score += 0.15
    return max(0.0, min(1.0, score))


def stability_bucket(value: float) -> str:
    if value >= 0.67:
        return "high"
    if value >= 0.34:
        return "medium"
    return "low"


def _label_similarity(obj: Object, query: QuerySpec) -> float:
    if obj.obj_id == query.target_object_id:
        return 1.0
    if query_label_matches(obj.label, query):
        return 0.95
    return query_label_similarity(obj.label, query)


def compute_candidate_score(
    obj: Object,
    query: QuerySpec,
    state_index: int,
    exploration_round: int,
    image_index: ImageGoalIndex,
    params: FusionParams,
) -> Optional[CandidateScore]:
    if not obj.pos_2d:
        return None
    try:
        x, z = float(obj.pos_2d[0]), float(obj.pos_2d[1])
    except Exception:
        return None

    if params.label_only or not params.use_clip:
        r_sim = _label_similarity(obj, query)
        sim = type("_Sim", (), {"score": r_sim, "backend": "label_only", "reason": "ablation_label_only"})()
    else:
        sim = image_index.similarity(query, obj)
        r_sim = sim.score
    if obj.obj_id == query.target_object_id:
        r_sim = max(r_sim, 1.0)
        sim_reason = f"{sim.reason},same_obj_id"
    elif query_label_matches(obj.label, query):
        r_sim = max(r_sim, 0.95)
        sim_reason = f"{sim.reason},same_label"
    else:
        sim_reason = sim.reason

    r_cfd = relation_confidence(obj, query) if params.use_cooccur else 0.0
    stability = max(0.0, min(1.0, float(obj.stability if obj.stability is not None else 0.5)))
    d_days = delta_days(obj, state_index)
    g_ts = 1.0 if not params.use_time_decay else math.exp(
        -(params.time_lambda * d_days) / (1.0 + params.kappa * stability)
    )
    base = params.beta1 * r_sim + params.beta2 * r_cfd
    exist_prob = max(0.0, min(1.0, float(obj.exist_prob if obj.exist_prob is not None else 1.0)))
    s_rag = max(0.0, min(1.0, base * g_ts * exist_prob))
    s_agent = agent_prior(obj, query) if params.use_agent else 0.0
    w_agent = 0.0 if not params.use_agent else max(params.agent_w_min, math.exp(-params.agent_k * max(0, exploration_round)))
    w_rag = 1.0 - w_agent
    s_final = w_rag * s_rag + w_agent * s_agent
    if s_final <= 0:
        return None
    return CandidateScore(
        obj_id=obj.obj_id,
        label=obj.label,
        world_x=x,
        world_z=z,
        R_sim=round(r_sim, 6),
        R_cfd=round(r_cfd, 6),
        G_ts=round(g_ts, 6),
        S_rag=round(s_rag, 6),
        S_agent=round(s_agent, 6),
        w_agent=round(w_agent, 6),
        w_rag=round(w_rag, 6),
        S_final=round(s_final, 6),
        stability=round(stability, 6),
        exist_prob=round(exist_prob, 6),
        negative_feedback_count=int((obj.cooccur_stats or {}).get("negative_feedback_count", 0) or 0),
        stability_bucket=stability_bucket(stability),
        query_modality=image_index.query_modality(query),
        sim_backend=sim.backend,
        reason=sim_reason,
    )


def rank_candidates(
    floors: List[Floor],
    query: QuerySpec,
    state_index: int,
    *,
    exploration_round: int = 0,
    max_candidates: int = 10,
    image_index: Optional[ImageGoalIndex] = None,
    params: Optional[FusionParams] = None,
) -> List[CandidateScore]:
    image_index = image_index or ImageGoalIndex()
    params = params or FusionParams()
    scores: List[CandidateScore] = []
    for obj in iter_objects(floors):
        cand = compute_candidate_score(obj, query, state_index, exploration_round, image_index, params)
        if cand is not None:
            scores.append(cand)
    scores.sort(key=lambda c: (-c.S_final, c.obj_id))
    deduped: List[CandidateScore] = []
    for cand in scores:
        if len(deduped) >= max_candidates:
            break
        if any(euclidean_2d([cand.world_x, cand.world_z], [old.world_x, old.world_z]) < 0.25 for old in deduped):
            continue
        deduped.append(cand)
    return deduped


def apply_negative_feedback(floors: List[Floor], obj_id: str, params: Optional[FusionParams] = None) -> Dict[str, Any]:
    params = params or FusionParams()
    for obj in iter_objects(floors):
        if obj.obj_id != obj_id:
            continue
        old_prob = max(0.0, min(1.0, float(obj.exist_prob if obj.exist_prob is not None else 1.0)))
        stability = max(0.0, min(1.0, float(obj.stability if obj.stability is not None else 0.5)))
        gamma = params.neg_gamma_min + params.neg_gamma_span * (1.0 - stability)
        new_prob = max(0.0, min(1.0, old_prob * gamma))
        obj.exist_prob = new_prob
        stats = obj.cooccur_stats or {}
        stats["consecutive_miss_count"] = int(stats.get("consecutive_miss_count", 0) or 0) + 1
        stats["negative_feedback_count"] = int(stats.get("negative_feedback_count", 0) or 0) + 1
        obj.cooccur_stats = stats
        return {
            "obj_id": obj_id,
            "old_exist_prob": old_prob,
            "new_exist_prob": new_prob,
            "gamma_neg": gamma,
            "consecutive_miss_count": stats["consecutive_miss_count"],
        }
    return {"obj_id": obj_id, "error": "object_not_found"}


def candidates_to_dict(candidates: List[CandidateScore]) -> List[Dict[str, Any]]:
    return [asdict(c) for c in candidates]
