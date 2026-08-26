"""Video library: upload, inspect, stream, export, re-index, delete."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi import (
    Path as PathParam,
)
from fastapi.responses import FileResponse, StreamingResponse

from app.api import exports
from app.api.deps import sse, to_summary
from app.api.schemas import (
    SegmentOut,
    TimelineResponse,
    UploadResponse,
    UrlIngestRequest,
    VideoListResponse,
    VideoSummary,
    parse_chapters,
)
from app.config import settings
from app.core.concurrency import run_blocking
from app.core.errors import BadRequest, ChronoscopeError, NotFound, UnsupportedMedia
from app.core.events import bus
from app.core.security import (
    VideoId,
    check_quota,
    safe_join,
    sanitize_title,
    validate_media_header,
    validate_subtitle,
)
from app.core.types import JobStatus, Scene, TimeSpan, stable_id
from app.ingest.decode import probe_async, sha256_file
from app.ingest.fetch import fetch_media, max_download_bytes
from app.ingest.pipeline import ingest_video
from app.logging_conf import get_logger
from app.retrieval.engine import invalidate_lexical
from app.store.db import VideoRepo, session_scope
from app.store.registry import get_store
from app.workers.runner import runner, size_priority

log = get_logger(__name__)
router = APIRouter(prefix="/api/videos", tags=["videos"])

ALLOWED_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"}
TRANSCRIPT_SUFFIXES = {".srt", ".vtt", ".json"}


@router.post("", response_model=UploadResponse, status_code=202)
async def upload_video(
    request: Request,
    file: Annotated[UploadFile, File(description="Video container")],
    title: Annotated[str, Form()] = "",
    transcript: Annotated[UploadFile | None, File(description="Optional .srt/.vtt/.json")] = None,
    reingest: Annotated[bool, Form()] = False,
) -> UploadResponse:
    name = Path(file.filename or "upload.mp4").name
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise UnsupportedMedia(
            f"unsupported container {suffix!r}", detail={"allowed": sorted(ALLOWED_SUFFIXES)}
        )

    # Reserve headroom before writing anything, so a full disk fails fast
    # instead of half-way through a multi-gigabyte body.
    declared = int(request.headers.get("content-length") or 0)
    check_quota(min(declared, settings.max_upload_mb * 1024 * 1024))

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    tmp = settings.upload_dir / f".incoming-{stable_id(name, id(file))[:12]}{suffix}"
    size = 0
    limit = settings.max_upload_mb * 1024 * 1024
    try:
        with tmp.open("wb") as out:
            first = True
            while chunk := await file.read(1 << 20):
                if first:
                    # Check magic bytes before keeping any of it. The
                    # extension is attacker-controlled; the header is the only
                    # evidence of what the file actually is.
                    validate_media_header(chunk[:64], filename=name)
                    first = False
                size += len(chunk)
                if size > limit:
                    raise BadRequest(f"file exceeds {settings.max_upload_mb} MB limit")
                out.write(chunk)
        if size == 0:
            raise BadRequest("uploaded file is empty")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    # A well-formed header still proves nothing about the stream. Probing here
    # rejects malformed or absurdly large media before it reaches the queue,
    # and gives the user an immediate, specific error instead of a failed job.
    try:
        probe = await probe_async(tmp)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise UnsupportedMedia(
            "this file could not be decoded as video", detail={"filename": name, "reason": str(exc)[:200]}
        ) from exc
    if probe.duration > settings.max_video_duration_s:
        tmp.unlink(missing_ok=True)
        raise BadRequest(
            f"video is {probe.duration / 3600:.1f} h, the limit is "
            f"{settings.max_video_duration_s / 3600:.1f} h",
            code="video_too_long",
        )
    if probe.width * probe.height > settings.max_video_pixels:
        tmp.unlink(missing_ok=True)
        raise BadRequest(
            f"frame size {probe.width}x{probe.height} exceeds the configured limit",
            code="resolution_too_high",
        )

    # Content-addressed identity: the same bytes always map to the same video.
    digest = await asyncio.to_thread(sha256_file, tmp)
    video_id = stable_id(digest)
    final = settings.upload_dir / f"{video_id}{suffix}"

    async with session_scope() as s:
        repo = VideoRepo(s)
        existing = await repo.get(video_id)
        if existing is not None and not reingest:
            tmp.unlink(missing_ok=True)
            return UploadResponse(
                video_id=video_id, status=existing.status, duplicate=True,
                message="identical content already indexed; pass reingest=true to force",
            )
        tmp.replace(final)
        if transcript is not None and transcript.filename:
            t_suffix = Path(transcript.filename).suffix.lower()
            if t_suffix not in TRANSCRIPT_SUFFIXES:
                raise UnsupportedMedia(
                    f"unsupported transcript format {t_suffix!r}",
                    detail={"allowed": sorted(TRANSCRIPT_SUFFIXES)},
                )
            body = await transcript.read()
            validate_subtitle(body, filename=transcript.filename)
            (settings.upload_dir / f"{video_id}{t_suffix}").write_bytes(body)
            await transcript.close()
        clean_title = sanitize_title(title) or Path(name).stem
        if existing is None:
            await repo.create(
                id=video_id, filename=sanitize_title(name, limit=255), path=str(final),
                title=clean_title, status=JobStatus.PENDING, size_bytes=size, sha256=digest,
                duration=probe.duration, width=probe.width, height=probe.height, fps=probe.fps,
            )
        else:
            await repo.update(
                video_id, status=JobStatus.PENDING, progress=0.0, error="", stage="",
                path=str(final), size_bytes=size,
            )

    runner.cancel(video_id)
    bus.drop_topic(f"video:{video_id}")
    invalidate_lexical(video_id)
    runner.submit(
        video_id, lambda: ingest_video(video_id, final), priority=size_priority(size), kind="ingest"
    )
    log.info("queued %s (%s, %.1f MB)", video_id, name, size / 1e6)
    return UploadResponse(video_id=video_id, status="queued", message="ingestion queued")


@router.post("/from-url", response_model=UploadResponse, status_code=202)
async def ingest_from_url(request: Request, body: UrlIngestRequest) -> UploadResponse:
    """Download a direct media URL and queue it, reusing the upload pipeline.

    Only direct links to media are accepted. Page URLs from video platforms are
    rejected by name, because downloading one yields HTML and the resulting
    error would otherwise be misleading.
    """
    check_quota(settings.max_upload_mb * 1024 * 1024 // 4)

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    tmp = settings.upload_dir / f".incoming-{stable_id(body.url, id(body))[:12]}.bin"
    result = await fetch_media(body.url, tmp, max_bytes=max_download_bytes())

    suffix = Path(result.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        suffix = ".mp4"

    try:
        probe = await probe_async(tmp)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise UnsupportedMedia(
            "the downloaded file could not be decoded as video",
            detail={"url": body.url[:200], "reason": str(exc)[:200]},
        ) from exc
    if probe.duration > settings.max_video_duration_s:
        tmp.unlink(missing_ok=True)
        raise BadRequest(
            f"video is {probe.duration / 3600:.1f} h, over the "
            f"{settings.max_video_duration_s / 3600:.1f} h limit",
            code="video_too_long",
        )
    if probe.width * probe.height > settings.max_video_pixels:
        tmp.unlink(missing_ok=True)
        raise BadRequest("frame size exceeds the configured limit", code="resolution_too_high")

    digest = await asyncio.to_thread(sha256_file, tmp)
    video_id = stable_id(digest)
    final = settings.upload_dir / f"{video_id}{suffix}"

    async with session_scope() as s:
        repo = VideoRepo(s)
        existing = await repo.get(video_id)
        if existing is not None:
            tmp.unlink(missing_ok=True)
            return UploadResponse(
                video_id=video_id, status=existing.status, duplicate=True,
                message="identical content is already indexed",
            )
        tmp.replace(final)
        await repo.create(
            id=video_id,
            filename=sanitize_title(result.filename, limit=255),
            path=str(final),
            title=sanitize_title(body.title) or Path(result.filename).stem,
            status=JobStatus.PENDING,
            size_bytes=result.size,
            sha256=digest,
            duration=probe.duration, width=probe.width, height=probe.height, fps=probe.fps,
        )

    runner.cancel(video_id)
    bus.drop_topic(f"video:{video_id}")
    invalidate_lexical(video_id)
    runner.submit(video_id, lambda: ingest_video(video_id, final), priority=size_priority(result.size), kind="ingest")
    log.info("queued %s from %s (%.1f MB)", video_id, result.final_url[:80], result.size / 1e6)
    return UploadResponse(video_id=video_id, status="queued", message="download complete, ingestion queued")


@router.get("", response_model=VideoListResponse)
async def list_videos(
    limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), status: str | None = None
) -> VideoListResponse:
    async with session_scope() as s:
        repo = VideoRepo(s)
        rows = await repo.list_videos(limit=limit, offset=offset, status=status)
        total = await repo.count()
        return VideoListResponse(
            videos=[
                to_summary(r, poster_exists=(settings.artifact_dir / r.id / "poster.jpg").exists()) for r in rows
            ],
            total=total,
        )


@router.get("/{video_id}", response_model=VideoSummary)
async def get_video(video_id: VideoId) -> VideoSummary:
    async with session_scope() as s:
        row = await VideoRepo(s).get(video_id)
        if row is None:
            raise NotFound(f"video {video_id} not found")
        return to_summary(row, poster_exists=(settings.artifact_dir / row.id / "poster.jpg").exists())


@router.get("/{video_id}/timeline", response_model=TimelineResponse)
async def get_timeline(video_id: VideoId) -> TimelineResponse:
    async with session_scope() as s:
        repo = VideoRepo(s)
        row = await repo.get(video_id)
        if row is None:
            raise NotFound(f"video {video_id} not found")
        scene_rows, keyframes, chunks, seg_rows = await asyncio.gather(
            repo.scenes(video_id), repo.keyframes(video_id), repo.chunks(video_id), repo.segments(video_id)
        )
        return TimelineResponse(
            video=to_summary(row, poster_exists=(settings.artifact_dir / row.id / "poster.jpg").exists()),
            chapters=parse_chapters(row.chapters),
            scenes=[
                Scene(
                    index=r.idx, span=TimeSpan(start=r.start, end=r.end), cut_score=r.cut_score,
                    static_ratio=r.static_ratio, kind=r.kind,  # type: ignore[arg-type]
                )
                for r in scene_rows
            ],
            keyframes=keyframes,
            chunks=chunks,
            segments=[
                SegmentOut(
                    index=r.idx, start=r.start, end=r.end, text=r.text, speaker=r.speaker, confidence=r.confidence
                )
                for r in seg_rows
            ],
        )


@router.get("/{video_id}/events")
async def video_events(video_id: VideoId, after: int = Query(0, ge=0)) -> StreamingResponse:
    """SSE progress stream with replay, safe to connect after ingestion starts."""
    topic = f"video:{video_id}"

    async def gen() -> Any:
        yield sse("hello", {"video_id": video_id, "topic": topic})
        try:
            async for event in bus.subscribe(topic, after=after):
                yield sse(event.kind, event.to_dict())
                if event.kind in {"job_done", "job_failed", "job_cancelled"}:
                    break
        except asyncio.CancelledError:  # client disconnected
            return

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.post("/{video_id}/reindex", response_model=UploadResponse, status_code=202)
async def reindex(video_id: VideoId) -> UploadResponse:
    async with session_scope() as s:
        row = await VideoRepo(s).get(video_id)
        if row is None:
            raise NotFound(f"video {video_id} not found")
        path = Path(row.path)
        await VideoRepo(s).update(video_id, status=JobStatus.PENDING, progress=0.0, error="", stage="")
    if not path.exists():
        raise NotFound(f"source media for {video_id} is missing from disk")
    runner.cancel(video_id)
    bus.drop_topic(f"video:{video_id}")
    invalidate_lexical(video_id)
    runner.submit(video_id, lambda: ingest_video(video_id, path), priority=1.0, kind="reindex")
    return UploadResponse(video_id=video_id, status="queued", message="re-ingestion queued")


@router.post("/{video_id}/cancel", status_code=202)
async def cancel(video_id: VideoId) -> dict[str, Any]:
    return {"cancelled": runner.cancel(video_id)}


@router.delete("/{video_id}", status_code=200)
async def delete_video(video_id: VideoId, background: BackgroundTasks) -> dict[str, Any]:
    async with session_scope() as s:
        repo = VideoRepo(s)
        row = await repo.get(video_id)
        if row is None:
            raise NotFound(f"video {video_id} not found")
        path = Path(row.path)
        await repo.delete(video_id)
    runner.cancel(video_id)
    invalidate_lexical(video_id)
    store = await get_store()
    await store.delete_video(video_id)
    bus.drop_topic(f"video:{video_id}")

    def cleanup() -> None:
        # `video_id` is already pattern-validated by the route, but these joins
        # feed `rmtree`; re-checking the resolved path against its root keeps
        # that guarantee local instead of depending on a caller three layers up.
        path.unlink(missing_ok=True)
        for suffix in TRANSCRIPT_SUFFIXES:
            path.with_suffix(suffix).unlink(missing_ok=True)
        for root in (settings.artifact_dir, settings.artifact_dir / "frames"):
            try:
                shutil.rmtree(safe_join(root, video_id), ignore_errors=True)
            except ChronoscopeError:
                log.error("refusing to delete outside the artifact root: %r", video_id)

    background.add_task(cleanup)
    return {"deleted": video_id}


@router.get("/{video_id}/media")
async def stream_media(video_id: VideoId) -> FileResponse:
    """Serve the source file. Starlette handles HTTP Range for seeking."""
    async with session_scope() as s:
        row = await VideoRepo(s).get(video_id)
        if row is None:
            raise NotFound(f"video {video_id} not found")
    path = Path(row.path)
    if not path.exists():
        raise NotFound("source media missing from disk")
    return FileResponse(
        path,
        media_type="video/mp4" if path.suffix.lower() in {".mp4", ".m4v"} else "application/octet-stream",
        filename=row.filename,
        headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"},
    )


# --------------------------------------------------------------------- export
EXPORTABLE = {"transcript", "chunks", "scenes", "keyframes", "bundle"}


#: Declared before the parameterised route below. FastAPI matches in
#: declaration order, so `/export/{dataset}` would otherwise swallow this
#: path and reject "frames.zip" against the dataset pattern.
@router.get("/{video_id}/export/frames.zip")
async def export_frames(video_id: VideoId, background: BackgroundTasks) -> FileResponse:
    """Every keyframe as a zip, named by timestamp so they sort chronologically."""
    async with session_scope() as s:
        repo = VideoRepo(s)
        row = await repo.get(video_id)
        if row is None:
            raise NotFound(f"video {video_id} not found")
        keyframes = await repo.keyframes(video_id)
        title = row.title
    if not keyframes:
        raise NotFound("this video has no keyframes")

    frames_root = settings.artifact_dir / "frames"

    def build() -> Path:
        # Streamed to a temp file rather than assembled in memory: 400 frames
        # at full resolution is comfortably over a hundred megabytes.
        fd, name = tempfile.mkstemp(prefix="chronoscope-frames-", suffix=".zip")
        os.close(fd)
        target = Path(name)
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
            for kf in keyframes:
                source = safe_join(frames_root, kf.path)
                if not source.exists():
                    continue
                stamp = f"{int(kf.timestamp // 60):02d}m{kf.timestamp % 60:06.3f}s".replace(".", "_")
                archive.write(source, arcname=f"scene{kf.scene_index:02d}_{stamp}.jpg")
            archive.writestr("keyframes.csv", exports.keyframes_to_csv(keyframes))
        return target

    archive_path = await run_blocking(build)
    background.add_task(lambda: archive_path.unlink(missing_ok=True))
    return FileResponse(
        archive_path,
        media_type="application/zip",
        filename=exports.safe_filename(f"{title}-frames", "zip"),
        headers={"Cache-Control": "no-store"},
    )


@router.get("/{video_id}/export/{dataset}")
async def export_dataset(
    video_id: VideoId,
    dataset: Annotated[str, PathParam(pattern="^(transcript|chunks|scenes|keyframes|bundle)$")],
    format: Annotated[str, Query(pattern="^(srt|vtt|txt|csv|json)$")] = "csv",
    speakers: bool = True,
) -> Response:
    """Download any derived dataset in a format other tools already read.

    Rendered in a worker thread, a two-hour transcript is a few megabytes of
    string building, which would otherwise stall the event loop for every other
    request in flight.
    """
    async with session_scope() as s:
        repo = VideoRepo(s)
        row = await repo.get(video_id)
        if row is None:
            raise NotFound(f"video {video_id} not found")
        scenes, keyframes, chunks, segments = await asyncio.gather(
            repo.scenes(video_id), repo.keyframes(video_id), repo.chunks(video_id), repo.segments(video_id)
        )
        title = row.title
        segment_views = [
            SegmentOut(
                index=r.idx, start=r.start, end=r.end, text=r.text,
                speaker=r.speaker, confidence=r.confidence,
            )
            for r in segments
        ]
        scene_models = [
            Scene(
                index=r.idx, span=TimeSpan(start=r.start, end=r.end), cut_score=r.cut_score,
                static_ratio=r.static_ratio, kind=r.kind,  # type: ignore[arg-type]
            )
            for r in scenes
        ]
        video_row = row

    def as_json(models: list[Any]) -> str:
        return json.dumps([m.model_dump() for m in models], indent=2, default=str)

    # (dataset, format) -> writer. A table keeps every combination visible and
    # makes an unsupported pairing impossible to reach by accident.
    writers: dict[tuple[str, str], Any] = {
        ("bundle", format): lambda: exports.bundle(video_row, scene_models, chunks, keyframes, segment_views),
        ("transcript", "srt"): lambda: exports.to_srt(segment_views, with_speakers=speakers),
        ("transcript", "vtt"): lambda: exports.to_vtt(segment_views, with_speakers=speakers),
        ("transcript", "txt"): lambda: exports.to_plain_text(segment_views, with_speakers=speakers),
        ("transcript", "json"): lambda: exports.bundle(video_row, [], [], [], segment_views),
        ("transcript", "csv"): lambda: exports.segments_to_csv(segment_views),
        ("chunks", "json"): lambda: as_json(list(chunks)),
        ("chunks", "csv"): lambda: exports.chunks_to_csv(chunks),
        ("scenes", "json"): lambda: as_json(list(scene_models)),
        ("scenes", "csv"): lambda: exports.scenes_to_csv(scene_models),
        ("keyframes", "json"): lambda: as_json(list(keyframes)),
        ("keyframes", "csv"): lambda: exports.keyframes_to_csv(keyframes),
    }

    def render() -> str:
        writer = writers.get((dataset, format))
        if writer is None:
            raise BadRequest(f"{dataset} cannot be exported as {format}")
        return str(writer())

    body = await run_blocking(render)
    suffix = format if dataset != "bundle" else "json"
    filename = exports.safe_filename(f"{title}-{dataset}", suffix)
    return Response(
        content=body,
        media_type=exports.CONTENT_TYPES.get(suffix, "text/plain; charset=utf-8"),
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            # Downloads are user data, never a shared cache's business.
            "Cache-Control": "no-store",
        },
    )
