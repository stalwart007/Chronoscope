"""Health, capability reporting and introspection.

The engine degrades in several independent dimensions and each changes what the
answers mean, so the health payload is detailed and the UI renders it verbatim.
"""

from __future__ import annotations

import platform
import time
from typing import Any

from fastapi import APIRouter

from app.agents.orchestrator import topology
from app.api.schemas import HealthResponse
from app.config import settings
from app.core.concurrency import run_blocking
from app.core.events import bus
from app.embed import registry as embeds
from app.llm.router import router as llm_router
from app.retrieval.engine import invalidate_lexical
from app.retrieval.lexical import index as bm25
from app.store.registry import fallback_reason, get_store
from app.workers.runner import runner

router = APIRouter(prefix="/api/system", tags=["system"])
_STARTED = time.time()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    from app import __version__

    encoders = await embeds.warm_up()
    store = await get_store(await embeds.dims())
    store_health = await store.health()
    llm = await llm_router.health()

    degraded: list[str] = []
    if encoders["text"].get("degraded"):
        degraded.append("text embeddings running in hashing-sketch mode (install sentence-transformers)")
    if encoders["image"].get("degraded"):
        degraded.append("image embeddings running in colour-layout mode (install open_clip_torch + torch)")
    if not llm["any_available"]:
        degraded.append("no LLM provider reachable, answers are extractive, planning is rule-based")
    if fallback_reason():
        degraded.append(f"vector store fell back to in-process HNSW: {fallback_reason()}")
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        degraded.append("faster-whisper not installed, transcripts come from sidecars or VAD only")

    return HealthResponse(
        status="degraded" if degraded else "ok",
        version=__version__,
        env=settings.env,
        encoders=encoders,
        vector_store=store_health,
        llm=llm,
        jobs=runner.stats(),
        lexical=dict(bm25.stats()),
        degraded=degraded,
    )


@router.get("/stats")
async def stats() -> dict[str, Any]:
    return {
        "uptime_s": round(time.time() - _STARTED, 1),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "events": bus.stats(),
        "jobs": runner.stats(),
        "query_cache": embeds.cache_stats(),
        "lexical": dict(bm25.stats()),
    }


@router.get("/graph")
async def agent_graph() -> dict[str, Any]:
    return topology()


@router.post("/demo", status_code=202)
async def load_demo() -> dict[str, Any]:
    """Generate and ingest the synthetic demo talk.

    The single biggest obstacle to trying a video tool is having a suitable
    video to hand. This renders one, five scenes, two speakers, a readable
    bar chart and a scripted transcript, and puts it straight through the
    pipeline, so a new install is one click from a working example.
    """
    from app.core.security import sanitize_title
    from app.core.types import JobStatus, stable_id
    from app.demo import generate
    from app.ingest.decode import sha256_file
    from app.ingest.pipeline import ingest_video
    from app.store.db import VideoRepo, session_scope

    staging = settings.upload_dir / ".demo-build.mp4"
    await run_blocking(generate, staging)
    digest = await run_blocking(sha256_file, staging)
    video_id = stable_id(digest)
    final = settings.upload_dir / f"{video_id}.mp4"

    async with session_scope() as session:
        repo = VideoRepo(session)
        existing = await repo.get(video_id)
        if existing is not None and existing.status == JobStatus.COMPLETED:
            staging.unlink(missing_ok=True)
            return {"video_id": video_id, "status": existing.status, "already_loaded": True}
        staging.replace(final)
        # The generator writes its transcript beside the staging file.
        sidecar = staging.with_suffix(".srt")
        if sidecar.exists():
            sidecar.replace(settings.upload_dir / f"{video_id}.srt")
        if existing is None:
            await repo.create(
                id=video_id,
                filename="chronoscope_demo.mp4",
                path=str(final),
                title=sanitize_title("Demo. Q3 Engineering Review"),
                status=JobStatus.PENDING,
                size_bytes=final.stat().st_size,
                sha256=digest,
            )
        else:
            await repo.update(video_id, status=JobStatus.PENDING, progress=0.0, error="", stage="")

    runner.cancel(video_id)
    bus.drop_topic(f"video:{video_id}")
    invalidate_lexical(video_id)
    runner.submit(video_id, lambda: ingest_video(video_id, final), priority=0.0, kind="demo")
    return {"video_id": video_id, "status": "queued", "already_loaded": False}


@router.get("/config")
async def config() -> dict[str, Any]:
    """Non-secret runtime configuration, drives the UI's settings drawer."""
    return {
        "chunk_target_s": settings.chunk_target_s,
        "chunk_max_s": settings.chunk_max_s,
        "scene_threshold": settings.scene_threshold,
        "max_keyframes": settings.max_keyframes,
        "rrf_k": settings.rrf_k,
        "mmr_lambda": settings.mmr_lambda,
        "temporal_decay_s": settings.temporal_decay_s,
        "neighbour_bonus": settings.neighbour_bonus,
        "final_k": settings.final_k,
        "retrieval_candidates": settings.retrieval_candidates,
        "vector_backend": settings.vector_backend,
        "whisper_model": settings.whisper_model,
        "clip_model": f"{settings.clip_model}/{settings.clip_pretrained}",
        "text_model": settings.text_model,
        "llm_chain": settings.provider_chain,
        "max_upload_mb": settings.max_upload_mb,
    }
