"""Cross-modal retrieval: parallel search, rank fusion, diversification.

A query is encoded once into a sentence-encoder vector and a CLIP-text vector.
Four channels then run concurrently: text, summary, image and lexical. Image
hits are max-pooled onto their parent chunk, results are fused with Reciprocal
Rank Fusion, spread to temporal neighbours, and diversified with MMR before the
parent records are loaded from the relational store.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from app.config import settings
from app.core.concurrency import run_blocking
from app.core.ranking import mmr, reciprocal_rank_fusion, temporal_diffusion
from app.core.types import (
    RetrievalRequest,
    RetrievalResult,
    RetrievalTrace,
    ScoredHit,
    VideoChunk,
)
from app.embed import registry as embeds
from app.logging_conf import get_logger
from app.retrieval.lexical import index as bm25
from app.store.base import IMAGE, SUMMARY, TEXT, SearchFilter, rollup_children
from app.store.db import VideoRepo, session_scope
from app.store.registry import get_store

log = get_logger(__name__)

LEXICAL = "lexical"
CHANNELS = (TEXT, SUMMARY, IMAGE, LEXICAL)

#: Baseline priors. Text and lexical are the most reliable channels; ``image``
#: is boosted by the planner when the question is visual.
DEFAULT_PRIOR = {TEXT: 1.0, SUMMARY: 0.85, IMAGE: 0.8, LEXICAL: 0.9}

#: When CLIP is unavailable the image tower is a colour/structure descriptor.
#: It still separates images meaningfully, but its text-to-image projection is
#: a coarse lexicon mapping, so a text query's visual ranking carries much less
#: information. Halving the prior keeps the channel contributing without letting
#: it outvote channels that read the words.
DEGRADED_IMAGE_PRIOR = 0.4


async def base_prior() -> dict[str, float]:
    prior = dict(DEFAULT_PRIOR)
    try:
        encoder = await embeds.image_encoder()
        if encoder.degraded:
            prior[IMAGE] = DEGRADED_IMAGE_PRIOR
    except Exception:
        prior[IMAGE] = DEGRADED_IMAGE_PRIOR
    return prior


async def ensure_lexical_index(video_ids: list[str] | None = None) -> None:
    """Lazily (re)build the BM25 index from the relational store."""
    async with session_scope() as s:
        repo = VideoRepo(s)
        targets = video_ids or [v.id for v in await repo.list_videos(limit=500)]
        missing = [vid for vid in targets if not bm25.has_video(vid)]
        for vid in missing:
            for chunk in await repo.chunks(vid):
                bm25.add(
                    chunk.id,
                    vid,
                    f"{chunk.text} {' '.join(chunk.keywords)} {' '.join(chunk.speakers)}",
                    start=chunk.span.start,
                    end=chunk.span.end,
                )
    if missing:
        log.info("BM25 index warmed for %d video(s): %s", len(missing), bm25.stats())


def invalidate_lexical(video_id: str) -> None:
    bm25.remove_video(video_id)


class RetrievalEngine:
    async def search(self, req: RetrievalRequest, *, prior: dict[str, float] | None = None) -> RetrievalResult:
        trace = RetrievalTrace()
        t_start = time.perf_counter()
        flt = SearchFilter(
            video_ids=list(req.video_ids),
            speakers=list(req.speakers),
            start=req.time_range.start if req.time_range else None,
            end=req.time_range.end if req.time_range else None,
        )
        wanted = [m for m in CHANNELS if m in set(req.modalities) | {LEXICAL}] or list(CHANNELS)

        await ensure_lexical_index(req.video_ids or None)
        store = await get_store(await embeds.dims())

        t0 = time.perf_counter()
        qvecs = await embeds.embed_query_multimodal(req.query)
        trace.timings_ms["encode"] = round((time.perf_counter() - t0) * 1000, 2)

        async def dense(modality: str) -> tuple[str, list[tuple[str, float]]]:
            t = time.perf_counter()
            hits = await store.search(modality, qvecs[modality], limit=req.candidates, flt=flt)
            if modality == IMAGE:
                hits = rollup_children(hits)
            trace.timings_ms[modality] = round((time.perf_counter() - t) * 1000, 2)
            return modality, [(h.id, h.score) for h in hits if h.id]

        async def lexical() -> tuple[str, list[tuple[str, float]]]:
            t = time.perf_counter()
            res = await run_blocking(
                bm25.search,
                req.query,
                limit=req.candidates,
                video_ids=req.video_ids or None,
                start=flt.start,
                end=flt.end,
            )
            trace.timings_ms[LEXICAL] = round((time.perf_counter() - t) * 1000, 2)
            return LEXICAL, res

        jobs: list[Any] = [dense(m) for m in wanted if m != LEXICAL]
        if LEXICAL in wanted:
            jobs.append(lexical())
        gathered = await asyncio.gather(*jobs, return_exceptions=True)

        ranked: dict[str, list[tuple[str, float]]] = {}
        for item in gathered:
            if isinstance(item, BaseException):
                trace.notes.append(f"channel failed: {item}")
                continue
            modality, hits = item
            if hits:
                ranked[modality] = hits
                trace.per_modality[modality] = [h[0] for h in hits[:12]]
        if not ranked:
            trace.timings_ms["total"] = round((time.perf_counter() - t_start) * 1000, 2)
            return RetrievalResult(hits=[], trace=trace)

        # ---------------------------------------------------------- fuse
        fusion = reciprocal_rank_fusion(
            ranked, k=settings.rrf_k, adaptive=True, prior={**(await base_prior()), **(prior or {})}
        )
        trace.fused_order = fusion.order[:24]
        trace.notes.append(
            "weights " + ", ".join(f"{k}={v:.2f}" for k, v in sorted(fusion.weights.items()))
        )

        # ------------------------------------------------------- hydrate
        shortlist = fusion.order[: max(req.top_k * 4, 24)]
        async with session_scope() as s:
            repo = VideoRepo(s)
            chunks = await repo.chunks_by_ids(shortlist)
            frame_ids = [fid for c in chunks.values() for fid in c.keyframe_ids]
            frames = await repo.keyframes_by_ids(frame_ids)
        shortlist = [cid for cid in shortlist if cid in chunks]

        # ------------------------------------------------- temporal glue
        scores = {cid: fusion.scores[cid] for cid in shortlist}
        if req.use_temporal_fusion and len(scores) > 1:
            spans = {cid: (chunks[cid].span.start, chunks[cid].span.end) for cid in shortlist}
            videos = {cid: chunks[cid].video_id for cid in shortlist}
            scores = temporal_diffusion(
                scores, spans, decay_s=settings.temporal_decay_s, bonus=settings.neighbour_bonus,
                same_video=videos,
            )
            shortlist = sorted(shortlist, key=lambda c: -scores[c])

        # ------------------------------------------------------- diversify
        vectors = await store.fetch_vectors(shortlist, TEXT)
        if len(vectors) < len(shortlist) // 2:  # backend may not return vectors
            vectors = {}
        order = (
            mmr(shortlist, scores, vectors, k=req.top_k, lambda_=req.mmr_lambda or settings.mmr_lambda)
            if vectors
            else shortlist[: req.top_k]
        )
        trace.mmr_order = list(order)

        hits = [
            ScoredHit(
                chunk_id=cid,
                video_id=chunks[cid].video_id,
                score=round(scores.get(cid, 0.0), 6),
                ranks=fusion.ranks.get(cid, {}),
                raw_scores=fusion.raw.get(cid, {}),
                fusion=fusion.contributions.get(cid, {}),
                chunk=chunks[cid],
                keyframes=[frames[f] for f in chunks[cid].keyframe_ids if f in frames],
            )
            for cid in order
        ]
        trace.timings_ms["total"] = round((time.perf_counter() - t_start) * 1000, 2)
        log.info("retrieved %d hits for %r in %.0f ms", len(hits), req.query[:60], trace.timings_ms["total"])
        return RetrievalResult(hits=hits, trace=trace)

    async def neighbours(self, chunk: VideoChunk, radius: int = 1) -> list[VideoChunk]:
        """Adjacent chunks, used to widen context before answering."""
        async with session_scope() as s:
            all_chunks = await VideoRepo(s).chunks(chunk.video_id)
        lo, hi = max(0, chunk.index - radius), min(len(all_chunks), chunk.index + radius + 1)
        return all_chunks[lo:hi]


engine = RetrievalEngine()
