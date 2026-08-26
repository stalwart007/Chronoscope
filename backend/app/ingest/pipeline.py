"""The ingestion pipeline.

Stages form a directed acyclic graph rather than a list, because the expensive
work is independent: transcription is CPU-bound on audio while keyframe
extraction is decode-bound on video.

probe feeds audio and scenes; audio feeds transcribe, which feeds diarize;
scenes feeds keyframes; align waits on diarize and keyframes; then embed, index
and summarize run in sequence.

Each stage waits on the ``asyncio.Event`` of its dependencies, so the graph
shape alone determines parallelism. Stages declare a progress weight, and the
reported percentage is the weighted completion.

Failure policy is per stage: optional stages degrade the result and record a
note, required stages abort the run with a ``PipelineError`` naming the stage.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings
from app.core.concurrency import KeyedLimiter, run_blocking
from app.core.errors import PipelineError
from app.core.events import bus
from app.core.types import (
    JobStatus,
    Keyframe,
    Scene,
    SpeakerTurn,
    StageName,
    TranscriptSegment,
    VideoChunk,
    VideoProbe,
)
from app.embed import registry as embeds
from app.ingest import align as align_mod
from app.ingest import chapters as chapter_mod
from app.ingest import decode, diarize
from app.ingest import keyframes as kf_mod
from app.ingest import scenes as scene_mod
from app.ingest import transcribe as asr
from app.logging_conf import get_logger
from app.logging_conf import job_id as job_ctx
from app.store.base import SUMMARY, TEXT, ChunkPoint, FramePoint
from app.store.db import VideoRepo, session_scope
from app.store.registry import get_store

log = get_logger(__name__)

_video_limiter = KeyedLimiter(limit=1)
_global_slots = asyncio.Semaphore(settings.ingest_concurrency)


@dataclass
class PipelineContext:
    video_id: str
    path: Path
    artifact_dir: Path
    probe: VideoProbe | None = None
    audio: np.ndarray | None = None
    audio_path: Path | None = None
    scenes: list[Scene] = field(default_factory=list)
    signal: scene_mod.ContentSignal | None = None
    keyframes: list[Keyframe] = field(default_factory=list)
    transcript: asr.TranscriptResult | None = None
    turns: list[SpeakerTurn] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    chunks: list[VideoChunk] = field(default_factory=list)
    text_vectors: dict[str, np.ndarray] = field(default_factory=dict)
    summary_vectors: dict[str, np.ndarray] = field(default_factory=dict)
    frame_vectors: dict[str, np.ndarray] = field(default_factory=dict)
    chapters: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False

    @property
    def segments(self) -> list[TranscriptSegment]:
        return self.transcript.segments if self.transcript else []


StageFn = Callable[[PipelineContext], Awaitable[None]]


@dataclass(slots=True)
class Stage:
    name: StageName
    fn: StageFn
    deps: tuple[StageName, ...] = ()
    weight: float = 1.0
    required: bool = True
    label: str = ""


# ------------------------------------------------------------------- stages
async def stage_probe(ctx: PipelineContext) -> None:
    ctx.probe = await decode.probe_async(ctx.path)
    if ctx.probe.duration <= 0:
        raise PipelineError("probe", "media reports zero duration")
    ctx.stats["probe"] = ctx.probe.model_dump()
    async with session_scope() as s:
        await VideoRepo(s).update(
            ctx.video_id,
            duration=ctx.probe.duration,
            width=ctx.probe.width,
            height=ctx.probe.height,
            fps=ctx.probe.fps,
        )
    await decode.make_poster(ctx.path, ctx.artifact_dir / ctx.video_id / "poster.jpg", at=min(1.0, ctx.probe.duration / 2))


async def stage_audio(ctx: PipelineContext) -> None:
    assert ctx.probe is not None
    if not ctx.probe.has_audio:
        ctx.audio = np.zeros(0, dtype=np.float32)
        ctx.notes.append("media has no audio track")
        return
    ctx.audio_path = ctx.artifact_dir / ctx.video_id / "audio.wav"
    ctx.audio = await run_blocking(decode.extract_audio, ctx.path, ctx.audio_path)
    ctx.stats["audio_seconds"] = round(float(ctx.audio.size) / 16000.0, 2)


async def stage_scenes(ctx: PipelineContext) -> None:
    assert ctx.probe is not None
    ctx.scenes, ctx.signal = await run_blocking(
        scene_mod.detect_scenes, str(ctx.path), duration=ctx.probe.duration
    )
    ctx.stats["scenes"] = len(ctx.scenes)
    async with session_scope() as s:
        await VideoRepo(s).replace_scenes(ctx.video_id, ctx.scenes)


async def stage_keyframes(ctx: PipelineContext) -> None:
    assert ctx.signal is not None
    ctx.keyframes = await run_blocking(
        kf_mod.extract_keyframes,
        str(ctx.path),
        ctx.video_id,
        ctx.scenes,
        ctx.signal,
        ctx.artifact_dir / "frames",
    )
    ctx.stats["keyframes"] = len(ctx.keyframes)
    ctx.stats["slides"] = sum(1 for k in ctx.keyframes if k.is_slide)
    async with session_scope() as s:
        await VideoRepo(s).replace_keyframes(ctx.video_id, ctx.keyframes)


async def stage_transcribe(ctx: PipelineContext) -> None:
    if ctx.audio is None or ctx.audio.size == 0:
        ctx.transcript = asr.TranscriptResult(segments=[], source="none", degraded=True)
        return
    assert ctx.audio_path is not None
    ctx.transcript = await run_blocking(asr.transcribe, ctx.path, ctx.audio_path, ctx.audio)
    ctx.stats["transcript"] = asr.transcript_stats(ctx.transcript) | {"source": ctx.transcript.source}
    if ctx.transcript.degraded:
        ctx.notes.append(f"transcription degraded ({ctx.transcript.source}), no ASR backend installed")


async def stage_diarize(ctx: PipelineContext) -> None:
    if ctx.audio is None or ctx.audio.size == 0:
        return
    result = None
    if settings.diarization_enabled and ctx.audio_path is not None:
        result = await run_blocking(diarize.diarize_pyannote, str(ctx.audio_path))
    if result is None:
        spans = [s.span for s in ctx.segments if s.span.duration > 0.4] or await run_blocking(
            asr.energy_vad, ctx.audio
        )
        result = await run_blocking(diarize.diarize_spectral, ctx.audio, spans)
        if result.degraded:
            ctx.notes.append("diarisation used the spectral MFCC fallback (set CS_HF_TOKEN for pyannote)")
    ctx.turns = result.turns
    ctx.speakers = diarize.assign_speakers(ctx.segments, ctx.turns) or result.speakers
    ctx.stats["speakers"] = len(ctx.speakers)
    ctx.stats["diarization_source"] = result.source
    async with session_scope() as s:
        await VideoRepo(s).replace_segments(ctx.video_id, ctx.segments)


async def stage_align(ctx: PipelineContext) -> None:
    assert ctx.probe is not None
    inputs = align_mod.AlignmentInputs(
        video_id=ctx.video_id,
        duration=ctx.probe.duration,
        segments=ctx.segments,
        turns=ctx.turns,
        scenes=ctx.scenes,
        keyframes=ctx.keyframes,
        signal=ctx.signal,
    )
    ctx.chunks = await run_blocking(align_mod.build_chunks, inputs)
    if not ctx.chunks:
        raise PipelineError("align", "no chunks produced, media may be empty")
    ctx.stats["chunks"] = len(ctx.chunks)
    async with session_scope() as s:
        await VideoRepo(s).replace_chunks(ctx.video_id, ctx.chunks)


def _contextual_texts(chunks: list[VideoChunk], *, tail_chars: int = 220) -> list[str]:
    """Widen each chunk with a tail of its neighbours, for embedding only.

    A sentence that straddles a boundary stays findable from either side, but
    the stored text, what BM25 indexes and the UI shows, remains exactly the
    span's own speech.
    """
    out: list[str] = []
    for i, c in enumerate(chunks):
        before = chunks[i - 1].text[-tail_chars:] if i > 0 else ""
        after = chunks[i + 1].text[:tail_chars] if i + 1 < len(chunks) else ""
        body = c.text or c.summary
        out.append(" ".join(p for p in (before, body, after) if p).strip())
    return out


async def stage_embed(ctx: PipelineContext) -> None:
    texts = _contextual_texts(ctx.chunks)
    summaries = [c.summary or c.text[:400] for c in ctx.chunks]
    frame_paths = [str(ctx.artifact_dir / "frames" / k.path) for k in ctx.keyframes]

    text_vecs, summary_vecs, frame_vecs = await asyncio.gather(
        embeds.embed_texts(texts),
        embeds.embed_texts(summaries),
        embeds.embed_images(frame_paths) if frame_paths else _empty(),
    )
    ctx.text_vectors = {c.id: v for c, v in zip(ctx.chunks, text_vecs, strict=True)}
    ctx.summary_vectors = {c.id: v for c, v in zip(ctx.chunks, summary_vecs, strict=True)}
    if len(frame_vecs):
        ctx.frame_vectors = {k.id: v for k, v in zip(ctx.keyframes, frame_vecs, strict=True)}
    ctx.stats["vectors"] = len(ctx.text_vectors) * 2 + len(ctx.frame_vectors)


async def _empty() -> np.ndarray:
    return np.zeros((0, 0), dtype=np.float32)


async def stage_chapters(ctx: PipelineContext) -> None:
    """Group chunks into topical chapters using their text embeddings."""
    if not ctx.chunks or not ctx.text_vectors:
        return
    chapters = await run_blocking(chapter_mod.segment, ctx.chunks, ctx.text_vectors)
    ctx.chapters = [c.to_dict() for c in chapters]
    ctx.stats["chapters"] = len(chapters)
    async with session_scope() as s:
        await VideoRepo(s).update(ctx.video_id, chapters=ctx.chapters)


async def stage_index(ctx: PipelineContext) -> None:
    store = await get_store(await embeds.dims())
    await store.delete_video(ctx.video_id)  # idempotent re-ingest
    chunk_points = [
        ChunkPoint(
            id=c.id,
            video_id=c.video_id,
            start=c.span.start,
            end=c.span.end,
            index=c.index,
            speakers=c.speakers,
            vectors={
                k: v
                for k, v in ((TEXT, ctx.text_vectors.get(c.id)), (SUMMARY, ctx.summary_vectors.get(c.id)))
                if v is not None
            },
            payload={"keywords": c.keywords[:8], "n_frames": len(c.keyframe_ids)},
        )
        for c in ctx.chunks
    ]
    chunk_of_frame = {kf_id: c.id for c in ctx.chunks for kf_id in c.keyframe_ids}
    frame_points = [
        FramePoint(
            id=k.id,
            chunk_id=chunk_of_frame.get(k.id, ""),
            video_id=ctx.video_id,
            timestamp=k.timestamp,
            vector=ctx.frame_vectors.get(k.id),
            payload={"is_slide": k.is_slide, "quality": k.quality, "scene": k.scene_index},
        )
        for k in ctx.keyframes
        if k.id in ctx.frame_vectors and chunk_of_frame.get(k.id)
    ]
    n_chunks = await store.upsert_chunks(chunk_points)
    n_frames = await store.upsert_frames(frame_points)
    ctx.stats["indexed"] = {"chunks": n_chunks, "frames": n_frames, "backend": store.name}


async def stage_summarize(ctx: PipelineContext) -> None:
    """Video-level abstract + topics. LLM if available, extractive otherwise."""
    from app.agents.summarizer import summarize_video

    summary, topics, used = await summarize_video(ctx.chunks, ctx.speakers, ctx.stats)
    ctx.stats["summary_source"] = used
    async with session_scope() as s:
        await VideoRepo(s).update(ctx.video_id, summary=summary, topics=topics, speakers=ctx.speakers)


STAGES: tuple[Stage, ...] = (
    Stage(StageName.PROBE, stage_probe, (), 2.0, True, "Inspecting media"),
    Stage(StageName.AUDIO, stage_audio, (StageName.PROBE,), 6.0, True, "Extracting audio"),
    Stage(StageName.SCENES, stage_scenes, (StageName.PROBE,), 16.0, True, "Detecting scenes"),
    Stage(StageName.KEYFRAMES, stage_keyframes, (StageName.SCENES,), 20.0, True, "Selecting keyframes"),
    Stage(StageName.TRANSCRIBE, stage_transcribe, (StageName.AUDIO,), 26.0, True, "Transcribing speech"),
    Stage(StageName.DIARIZE, stage_diarize, (StageName.TRANSCRIBE,), 8.0, False, "Identifying speakers"),
    Stage(StageName.ALIGN, stage_align, (StageName.DIARIZE, StageName.KEYFRAMES), 4.0, True, "Aligning modalities"),
    Stage(StageName.EMBED, stage_embed, (StageName.ALIGN,), 12.0, True, "Embedding chunks"),
    Stage(StageName.CHAPTERS, stage_chapters, (StageName.EMBED,), 3.0, False, "Finding chapters"),
    Stage(StageName.INDEX, stage_index, (StageName.EMBED,), 4.0, True, "Indexing vectors"),
    Stage(StageName.SUMMARIZE, stage_summarize, (StageName.INDEX,), 4.0, False, "Summarising"),
)


def validate_dag(stages: tuple[Stage, ...] = STAGES) -> list[StageName]:
    """Kahn's algorithm, topological order, and proof the graph is acyclic."""
    by_name = {s.name: s for s in stages}
    indeg = {s.name: len(s.deps) for s in stages}
    children: dict[StageName, list[StageName]] = {s.name: [] for s in stages}
    for s in stages:
        for d in s.deps:
            if d not in by_name:
                raise ValueError(f"stage {s.name} depends on unknown stage {d}")
            children[d].append(s.name)
    queue = [n for n, d in indeg.items() if d == 0]
    order: list[StageName] = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for c in children[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(c)
    if len(order) != len(stages):
        raise ValueError("pipeline DAG contains a cycle")
    return order


# ------------------------------------------------------------------ executor
class IngestPipeline:
    def __init__(self, stages: tuple[Stage, ...] = STAGES) -> None:
        validate_dag(stages)
        self.stages = stages
        self.total_weight = sum(s.weight for s in stages)

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        topic = f"video:{ctx.video_id}"
        done: dict[StageName, asyncio.Event] = {s.name: asyncio.Event() for s in self.stages}
        completed: dict[StageName, float] = {}
        failed: dict[StageName, str] = {}
        started = time.perf_counter()
        job_ctx.set(ctx.video_id)

        def progress() -> float:
            return round(100.0 * sum(completed.values()) / self.total_weight, 2)

        async def publish_status(stage: Stage, kind: str, **extra: Any) -> None:
            payload = {"stage": stage.name.value, "label": stage.label, "progress": progress(), **extra}
            bus.publish(topic, kind, **payload)
            async with session_scope() as s:
                await VideoRepo(s).update(
                    ctx.video_id, stage=stage.name.value, progress=payload["progress"], status=JobStatus.RUNNING
                )

        async def run_stage(stage: Stage) -> None:
            try:
                for dep in stage.deps:
                    await done[dep].wait()
                if ctx.cancelled or any(f in failed for f in _required_ancestors(stage, self.stages)):
                    done[stage.name].set()
                    return
                await publish_status(stage, "stage_start")
                t0 = time.perf_counter()
                async with _global_slots:
                    await stage.fn(ctx)
                elapsed = (time.perf_counter() - t0) * 1000
                completed[stage.name] = stage.weight
                ctx.stats.setdefault("timings_ms", {})[stage.name.value] = round(elapsed, 1)
                await publish_status(stage, "stage_done", elapsed_ms=round(elapsed, 1))
                log.info("stage %s finished in %.0f ms", stage.name.value, elapsed)
            except asyncio.CancelledError:
                ctx.cancelled = True
                raise
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                log.exception("stage %s failed", stage.name.value)
                if stage.required:
                    failed[stage.name] = msg
                    bus.publish(topic, "stage_error", stage=stage.name.value, error=msg, fatal=True)
                else:
                    ctx.notes.append(f"{stage.name.value} skipped: {msg}")
                    completed[stage.name] = stage.weight
                    bus.publish(topic, "stage_error", stage=stage.name.value, error=msg, fatal=False)
            finally:
                done[stage.name].set()

        tasks = [asyncio.create_task(run_stage(s), name=f"stage:{s.name}") for s in self.stages]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            with contextlib.suppress(Exception):
                await asyncio.gather(*tasks, return_exceptions=True)
            raise

        ctx.stats["elapsed_s"] = round(time.perf_counter() - started, 2)
        if failed:
            first = next(iter(failed))
            raise PipelineError(first.value, failed[first], detail=ctx.stats)
        return ctx


def _required_ancestors(stage: Stage, stages: tuple[Stage, ...]) -> set[StageName]:
    by_name = {s.name: s for s in stages}
    seen: set[StageName] = set()
    frontier = list(stage.deps)
    while frontier:
        n = frontier.pop()
        if n in seen:
            continue
        seen.add(n)
        frontier.extend(by_name[n].deps)
    return seen


async def ingest_video(video_id: str, path: Path) -> PipelineContext:
    """Full ingestion for one video, serialised per-video and globally bounded."""
    topic = f"video:{video_id}"
    ctx = PipelineContext(video_id=video_id, path=Path(path), artifact_dir=settings.artifact_dir)
    (ctx.artifact_dir / video_id).mkdir(parents=True, exist_ok=True)
    pipeline = IngestPipeline()
    bus.publish(topic, "job_start", video_id=video_id, stages=[s.name.value for s in pipeline.stages])
    try:
        async with _video_limiter.acquire(video_id):
            await pipeline.run(ctx)
        async with session_scope() as s:
            await VideoRepo(s).update(
                video_id,
                status=JobStatus.COMPLETED,
                progress=100.0,
                stage="",
                error="",
                stats=ctx.stats | {"notes": ctx.notes},
                language=(ctx.transcript.language if ctx.transcript else "") or "",
            )
        bus.publish(topic, "job_done", video_id=video_id, stats=ctx.stats, notes=ctx.notes)
        return ctx
    except asyncio.CancelledError:
        async with session_scope() as s:
            await VideoRepo(s).update(video_id, status=JobStatus.CANCELLED, stage="", error="cancelled")
        bus.publish(topic, "job_cancelled", video_id=video_id)
        raise
    except Exception as exc:
        async with session_scope() as s:
            await VideoRepo(s).update(video_id, status=JobStatus.FAILED, error=str(exc)[:1000])
        bus.publish(topic, "job_failed", video_id=video_id, error=str(exc))
        raise
    finally:
        bus.close_topic(topic)
