"""Retrieval node: turn a plan into evidence.

Sub-tasks run concurrently and their ranked lists are fused, so a plan holding
both a visual and a transcript lookup searches both ways and merges.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.agents.planner import aggregate_bias, parse_speaker_ordinal, parse_time_filter
from app.agents.state import AgentState
from app.config import settings
from app.core.ranking import reciprocal_rank_fusion
from app.core.types import Citation, RetrievalRequest, ScoredHit, TaskKind
from app.logging_conf import get_logger
from app.retrieval.engine import engine
from app.store.db import VideoRepo, session_scope

log = get_logger(__name__)


async def resolve_speaker_filter(state: AgentState) -> list[str]:
    """Turn "the second speaker" into a concrete diarisation label."""
    if state.speakers:
        return state.speakers
    ordinal = parse_speaker_ordinal(state.effective_query)
    if ordinal is None:
        return []
    async with session_scope() as s:
        repo = VideoRepo(s)
        vids = state.video_ids or [v.id for v in await repo.list_videos(limit=25)]
        names: list[str] = []
        for vid in vids:
            row = await repo.get(vid)
            for spk in (row.speakers if row else []) or []:
                if spk not in names:
                    names.append(str(spk))
    if not names:
        return []
    names.sort()
    try:
        return [names[ordinal]]
    except IndexError:
        return []


async def retrieve(state: AgentState) -> dict[str, Any]:
    tasks = state.plan.tasks or []
    speakers = await resolve_speaker_filter(state)
    time_range = state.time_range or parse_time_filter(state.effective_query)
    prior = aggregate_bias(state.plan)

    retrieval_kinds = {
        TaskKind.VISUAL_LOOKUP, TaskKind.TRANSCRIPT_LOOKUP, TaskKind.SUMMARIZE,
        TaskKind.TEMPORAL_LOCATE, TaskKind.COMPARE, TaskKind.SPEAKER_ATTRIBUTION,
        TaskKind.CHART_EXTRACTION,
    }
    runnable = [t for t in tasks if t.kind in retrieval_kinds] or []
    queries: list[tuple[str, dict[str, float]]] = (
        [(t.query, t.modality_bias or prior) for t in runnable] if runnable else [(state.effective_query, prior)]
    )
    # Deduplicate identical sub-queries, plans often repeat the user's words.
    seen: dict[str, dict[str, float]] = {}
    for q, bias in queries:
        seen.setdefault(q.strip().lower(), bias)
    unique = [(q, seen[q]) for q in seen]

    async def one(query: str, bias: dict[str, float]) -> tuple[str, Any]:
        req = RetrievalRequest(
            query=query,
            video_ids=state.video_ids,
            speakers=speakers,
            time_range=time_range,
            top_k=max(state.top_k, settings.final_k),
            candidates=settings.retrieval_candidates,
        )
        return query, await engine.search(req, prior=bias or None)

    results = await asyncio.gather(*(one(q, b) for q, b in unique), return_exceptions=True)

    per_query: dict[str, list[tuple[str, float]]] = {}
    hit_index: dict[str, ScoredHit] = {}
    trace = state.trace
    for item in results:
        if isinstance(item, BaseException):
            state.notes.append(f"sub-query failed: {item}")
            continue
        query, res = item
        per_query[query] = [(h.chunk_id, h.score) for h in res.hits]
        for h in res.hits:
            prev = hit_index.get(h.chunk_id)
            if prev is None or h.score > prev.score:
                hit_index[h.chunk_id] = h
        trace.timings_ms.update({f"{query[:24]}:{k}": v for k, v in res.trace.timings_ms.items()})
        trace.per_modality.update(res.trace.per_modality)
        trace.notes.extend(res.trace.notes)

    if not hit_index:
        return {"hits": [], "notes": ["no evidence retrieved"]}

    if len(per_query) > 1:
        fused = reciprocal_rank_fusion(per_query, k=settings.rrf_k, adaptive=True)
        order = [c for c in fused.order if c in hit_index]
        trace.fused_order = order[:24]
    else:
        order = sorted(hit_index, key=lambda c: -hit_index[c].score)

    hits = [hit_index[c] for c in order[: max(state.top_k, settings.final_k)]]
    citations = [
        Citation(
            chunk_id=h.chunk_id,
            video_id=h.video_id,
            start=h.chunk.span.start if h.chunk else 0.0,
            end=h.chunk.span.end if h.chunk else 0.0,
            speaker=(h.chunk.speakers[0] if h.chunk and h.chunk.speakers else None),
            keyframe=(h.keyframes[0].path if h.keyframes else None),
            quote=_best_quote(h),
            relevance=round(h.score, 6),
        )
        for h in hits[:6]
    ]
    return {"hits": hits, "citations": citations, "trace": trace, "speakers": speakers}


def _best_quote(hit: ScoredHit, *, limit: int = 220) -> str:
    if hit.chunk and hit.chunk.sentences:
        return " ".join(s.text for s in hit.chunk.sentences[:2])[:limit]
    text = (hit.chunk.text if hit.chunk else "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."
