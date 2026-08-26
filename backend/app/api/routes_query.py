"""Query surface: agentic answers, streamed or not, and raw retrieval."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.agents.conversation import Turn
from app.agents.orchestrator import answer_stream, topology
from app.api.deps import iso_utc, sse
from app.api.schemas import (
    AnswerResponse,
    QueryRequest,
    SearchRequest,
    SearchResponse,
    SessionDetail,
    SessionOut,
    TurnOut,
)
from app.core.errors import NotFound
from app.core.security import is_valid_id
from app.core.types import Citation, RetrievalRequest, TimeSpan, stable_id, utcnow
from app.logging_conf import get_logger
from app.retrieval.engine import engine
from app.store.db import SessionRepo, VideoRepo, session_scope

log = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["query"])


async def _log_query(bundle: Any, req_video_ids: list[str]) -> None:
    try:
        async with session_scope() as s:
            await VideoRepo(s).log_query(
                id=stable_id(bundle.query, bundle.elapsed_ms),
                query=bundle.query,
                video_ids=req_video_ids,
                answer=bundle.answer[:4000],
                confidence=bundle.confidence,
                elapsed_ms=bundle.elapsed_ms,
                payload={
                    "citations": [c.model_dump() for c in bundle.citations],
                    "plan": bundle.plan.model_dump(),
                    "model": bundle.model_used,
                },
            )
    except Exception as exc:
        log.warning("query logging failed: %s", exc)


async def _load_history(session_id: str) -> list[Turn]:
    """Earlier turns, as the resolver expects them."""
    if not session_id:
        return []
    async with session_scope() as s:
        rows = await SessionRepo(s).history(session_id)
    return [
        Turn(
            query=row.query,
            answer=row.answer,
            citations=[Citation.model_validate(c) for c in (row.citations or []) if isinstance(c, dict)],
            keywords=[str(k) for k in (row.keywords or [])],
        )
        for row in rows
    ]


async def _record_turn(session_id: str, bundle: Any, video_ids: list[str]) -> None:
    # Prefer the vocabulary of the cited sentences: those are what the answer
    # was actually built from, whereas a merely-retrieved chunk may be about
    # something adjacent.
    from app.agents.conversation import subject_words

    keywords: list[str] = []
    for cite in bundle.citations[:3]:
        for word in subject_words(cite.quote)[:6]:
            if word not in keywords:
                keywords.append(word)
    if len(keywords) < 3:
        for hit in bundle.hits[:3]:
            for k in (hit.chunk.keywords if hit.chunk else [])[:4]:
                if k not in keywords and " " not in k:
                    keywords.append(k)
    async with session_scope() as s:
        repo = SessionRepo(s)
        await repo.ensure(session_id, video_ids=video_ids, title=bundle.query[:300])
        await repo.append(
            session_id,
            query=bundle.query,
            resolved_query=bundle.resolved_query,
            answer=bundle.answer[:8000],
            citations=[c.model_dump() for c in bundle.citations[:8]],
            keywords=keywords[:8],
            confidence=bundle.confidence,
            elapsed_ms=bundle.elapsed_ms,
        )


@router.post("/query", response_model=AnswerResponse)
async def ask(req: QueryRequest) -> AnswerResponse:
    session_id = req.session_id or stable_id("session", req.query, utcnow().isoformat())
    history = await _load_history(req.session_id)
    bundle = None
    async for kind, payload in answer_stream(
        req.query,
        video_ids=req.video_ids,
        speakers=req.speakers,
        time_range=req.time_range,
        top_k=req.top_k,
        history=history,
    ):
        if kind == "result":
            bundle = payload
    assert bundle is not None
    bundle.session_id = session_id
    await _log_query(bundle, req.video_ids)
    await _record_turn(session_id, bundle, req.video_ids)
    return AnswerResponse(answer=bundle)


@router.get("/sessions", response_model=list[SessionOut])
async def list_sessions(limit: int = Query(30, ge=1, le=100)) -> list[SessionOut]:
    async with session_scope() as s:
        rows = await SessionRepo(s).list_sessions(limit=limit)
        return [_session_out(r) for r in rows]


@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str) -> SessionDetail:
    async with session_scope() as s:
        repo = SessionRepo(s)
        row = await repo.get(session_id)
        if row is None:
            raise NotFound(f"session {session_id} not found")
        turns = await repo.history(session_id, limit=100)
        return SessionDetail(
            session=_session_out(row),
            turns=[
                TurnOut(
                    index=t.idx, query=t.query, resolved_query=t.resolved_query, answer=t.answer,
                    citations=[Citation.model_validate(c) for c in (t.citations or []) if isinstance(c, dict)],
                    confidence=t.confidence, elapsed_ms=t.elapsed_ms, created_at=iso_utc(t.created_at),
                )
                for t in turns
            ],
        )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    async with session_scope() as s:
        await SessionRepo(s).delete(session_id)
    return {"deleted": session_id}


def _session_out(row: Any) -> SessionOut:
    return SessionOut(
        id=row.id, title=row.title, video_ids=[str(v) for v in (row.video_ids or [])],
        turn_count=row.turn_count, created_at=iso_utc(row.created_at), updated_at=iso_utc(row.updated_at),
    )


@router.get("/query/stream")
async def ask_stream(
    q: str = Query(min_length=1, max_length=2000),
    session_id: str = Query("", max_length=64),
    video_ids: str = Query(""),
    speakers: str = Query(""),
    start: float | None = None,
    end: float | None = None,
    top_k: int = Query(8, ge=1, le=32),
) -> StreamingResponse:
    """Server-sent events: one ``agent`` event per graph transition, then
    ``answer``. This is what makes the swarm visible in the UI."""
    vids = [v for v in video_ids.split(",") if is_valid_id(v)][:50]
    spks = [re.sub(r"[^A-Za-z0-9_\- ]", "", s)[:64] for s in speakers.split(",") if s][:20]
    # An open-ended filter ("after 10:00") must stay open-ended: TimeSpan
    # requires both bounds, so the missing side becomes an effective infinity.
    span = (
        TimeSpan(start=start if start is not None else 0.0, end=end if end is not None else 1e9)
        if (start is not None or end is not None)
        else None
    )

    resolved_session = session_id or stable_id("session", q, utcnow().isoformat())

    async def gen() -> Any:
        yield sse("open", {"query": q, "graph": topology(), "session_id": resolved_session})
        bundle = None
        history = await _load_history(session_id)
        try:
            async for kind, payload in answer_stream(
                q, video_ids=vids, speakers=spks, time_range=span, top_k=top_k, history=history
            ):
                if kind == "event":
                    yield sse("agent", json.loads(payload.model_dump_json()))
                else:
                    bundle = payload
                    bundle.session_id = resolved_session
                    yield sse("answer", json.loads(payload.model_dump_json()))
        except asyncio.CancelledError:
            return
        except Exception as exc:
            log.exception("query stream failed")
            yield sse("error", {"message": str(exc)})
            return
        if bundle is not None:
            await _log_query(bundle, vids)
            await _record_turn(resolved_session, bundle, vids)
        yield sse("done", {"ok": True, "session_id": resolved_session})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    """Raw retrieval, no agents. Useful for evaluation and the debug panel."""
    result = await engine.search(
        RetrievalRequest(
            query=req.query,
            video_ids=req.video_ids,
            speakers=req.speakers,
            time_range=req.time_range,
            top_k=req.top_k,
            candidates=req.candidates,
            modalities=req.modalities,
            mmr_lambda=req.mmr_lambda,
            use_temporal_fusion=req.use_temporal_fusion,
        )
    )
    return SearchResponse(hits=result.hits, trace=result.trace)


@router.get("/query/history")
async def history(limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
    async with session_scope() as s:
        rows = await VideoRepo(s).recent_queries(limit=limit)
        return {
            "queries": [
                {
                    "id": r.id,
                    "query": r.query,
                    "answer": r.answer[:400],
                    "confidence": r.confidence,
                    "elapsed_ms": r.elapsed_ms,
                    "video_ids": r.video_ids,
                    "created_at": iso_utc(r.created_at),
                }
                for r in rows
            ]
        }
