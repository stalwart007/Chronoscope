"""Relational metadata store (SQLAlchemy 2.0, async).

The vector store owns embeddings; this owns everything else: video records,
chunk text, keyframe geometry, transcript segments, job state and the query
log. Runs on SQLite with WAL by default; point ``CS_DATABASE_URL`` at Postgres
for production.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy import (
    JSON,
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    delete,
    event,
    func,
    select,
)
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import settings
from app.core.types import (
    JobStatus,
    Keyframe,
    Scene,
    Sentence,
    TimeSpan,
    TranscriptSegment,
    VideoChunk,
    utcnow,
)
from app.logging_conf import get_logger

log = get_logger(__name__)


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSON, list[Any]: JSON}


class VideoRow(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512))
    path: Mapped[str] = mapped_column(String(1024))
    title: Mapped[str] = mapped_column(String(512), default="")
    status: Mapped[str] = mapped_column(String(32), default=JobStatus.PENDING, index=True)
    stage: Mapped[str] = mapped_column(String(32), default="")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str] = mapped_column(Text, default="")
    duration: Mapped[float] = mapped_column(Float, default=0.0)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    sha256: Mapped[str] = mapped_column(String(64), default="", index=True)
    language: Mapped[str] = mapped_column(String(16), default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    speakers: Mapped[list[Any]] = mapped_column(JSON, default=list)
    topics: Mapped[list[Any]] = mapped_column(JSON, default=list)
    chapters: Mapped[list[Any]] = mapped_column(JSON, default=list)
    stats: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)

    chunks: Mapped[list[ChunkRow]] = relationship(back_populates="video", cascade="all, delete-orphan")


class ChunkRow(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    idx: Mapped[int] = mapped_column(Integer)
    start: Mapped[float] = mapped_column(Float, index=True)
    end: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text, default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    speakers: Mapped[list[Any]] = mapped_column(JSON, default=list)
    keyframe_ids: Mapped[list[Any]] = mapped_column(JSON, default=list)
    scene_indices: Mapped[list[Any]] = mapped_column(JSON, default=list)
    keywords: Mapped[list[Any]] = mapped_column(JSON, default=list)
    sentences: Mapped[list[Any]] = mapped_column(JSON, default=list)
    speech_rate: Mapped[float] = mapped_column(Float, default=0.0)
    visual_activity: Mapped[float] = mapped_column(Float, default=0.0)
    token_estimate: Mapped[int] = mapped_column(Integer, default=0)

    video: Mapped[VideoRow] = relationship(back_populates="chunks")

    __table_args__ = (Index("ix_chunk_video_idx", "video_id", "idx"),)


class KeyframeRow(Base):
    __tablename__ = "keyframes"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    scene_index: Mapped[int] = mapped_column(Integer, default=0)
    timestamp: Mapped[float] = mapped_column(Float, index=True)
    path: Mapped[str] = mapped_column(String(1024))
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    phash: Mapped[str] = mapped_column(String(32), default="0")
    quality: Mapped[float] = mapped_column(Float, default=0.0)
    sharpness: Mapped[float] = mapped_column(Float, default=0.0)
    entropy: Mapped[float] = mapped_column(Float, default=0.0)
    text_density: Mapped[float] = mapped_column(Float, default=0.0)
    is_slide: Mapped[bool] = mapped_column(Boolean, default=False)
    caption: Mapped[str] = mapped_column(Text, default="")


class SegmentRow(Base):
    __tablename__ = "segments"

    id: Mapped[str] = mapped_column(String(72), primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    idx: Mapped[int] = mapped_column(Integer)
    start: Mapped[float] = mapped_column(Float, index=True)
    end: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text, default="")
    speaker: Mapped[str] = mapped_column(String(64), default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    words: Mapped[list[Any]] = mapped_column(JSON, default=list)


class SceneRow(Base):
    __tablename__ = "scenes"

    id: Mapped[str] = mapped_column(String(72), primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    idx: Mapped[int] = mapped_column(Integer)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    cut_score: Mapped[float] = mapped_column(Float, default=0.0)
    static_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    kind: Mapped[str] = mapped_column(String(16), default="cut")


class SessionRow(Base):
    """A conversation. Turns are the individual exchanges within it."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(300), default="")
    video_ids: Mapped[list[Any]] = mapped_column(JSON, default=list)
    turn_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)


class TurnRow(Base):
    __tablename__ = "turns"

    id: Mapped[str] = mapped_column(String(72), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    idx: Mapped[int] = mapped_column(Integer)
    query: Mapped[str] = mapped_column(Text)
    resolved_query: Mapped[str] = mapped_column(Text, default="")
    answer: Mapped[str] = mapped_column(Text, default="")
    citations: Mapped[list[Any]] = mapped_column(JSON, default=list)
    keywords: Mapped[list[Any]] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    elapsed_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    __table_args__ = (Index("ix_turn_session_idx", "session_id", "idx"),)


class QueryRow(Base):
    __tablename__ = "queries"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query: Mapped[str] = mapped_column(Text)
    video_ids: Mapped[list[Any]] = mapped_column(JSON, default=list)
    answer: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    elapsed_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(default=utcnow, index=True)


# --------------------------------------------------------------------- engine
_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def _sqlite_pragmas(dbapi_conn: Any, _rec: Any) -> None:
    cur = dbapi_conn.cursor()
    for pragma in (
        "PRAGMA journal_mode=WAL",
        "PRAGMA synchronous=NORMAL",
        "PRAGMA foreign_keys=ON",
        "PRAGMA busy_timeout=5000",
        "PRAGMA cache_size=-64000",
        "PRAGMA temp_store=MEMORY",
    ):
        cur.execute(pragma)
    cur.close()


def get_engine() -> AsyncEngine:
    global _engine, _sessionmaker
    if _engine is None:
        url = settings.sqlalchemy_url
        kwargs: dict[str, Any] = {"echo": False, "future": True}
        if url.startswith("postgresql"):
            kwargs |= {"pool_size": 10, "max_overflow": 20, "pool_pre_ping": True}
        _engine = create_async_engine(url, **kwargs)
        if url.startswith("sqlite"):
            event.listen(_engine.sync_engine, "connect", _sqlite_pragmas)
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)
    return _engine


def sessionmaker() -> async_sessionmaker[AsyncSession]:
    get_engine()
    assert _sessionmaker is not None
    return _sessionmaker


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    async with sessionmaker()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_missing_columns)


def _add_missing_columns(connection: Any) -> None:
    """Add columns that exist on the models but not yet in the database.

    ``create_all`` only creates missing *tables*, so a schema change would
    otherwise break every existing install with "no such column" on the next
    query. This handles the additive case, which is all the schema has needed:
    new nullable or defaulted columns. Anything structural (renames, type
    changes, constraints) still needs a real migration tool, and this
    deliberately does not pretend otherwise.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(connection)
    existing_tables = set(inspector.get_table_names())
    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue
        present = {col["name"] for col in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in present:
                continue
            ddl_type = column.type.compile(connection.dialect)
            default = "'[]'" if isinstance(column.type, JSON) else "NULL"
            connection.execute(
                text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {ddl_type} DEFAULT {default}')
            )
            log.info("added column %s.%s", table.name, column.name)


async def dispose_db() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine, _sessionmaker = None, None


# ----------------------------------------------------------------- converters
def chunk_to_row(c: VideoChunk) -> dict[str, Any]:
    return {
        "id": c.id,
        "video_id": c.video_id,
        "idx": c.index,
        "start": c.span.start,
        "end": c.span.end,
        "text": c.text,
        "summary": c.summary,
        "speakers": c.speakers,
        "keyframe_ids": c.keyframe_ids,
        "scene_indices": c.scene_indices,
        "keywords": c.keywords,
        "sentences": [s.model_dump() for s in c.sentences],
        "speech_rate": c.speech_rate,
        "visual_activity": c.visual_activity,
        "token_estimate": c.token_estimate,
    }


def row_to_chunk(r: ChunkRow) -> VideoChunk:
    return VideoChunk(
        id=r.id,
        video_id=r.video_id,
        index=r.idx,
        span=TimeSpan(start=r.start, end=r.end),
        text=r.text,
        summary=r.summary,
        speakers=list(r.speakers or []),
        keyframe_ids=list(r.keyframe_ids or []),
        scene_indices=list(r.scene_indices or []),
        keywords=list(r.keywords or []),
        sentences=[Sentence(**s) for s in (r.sentences or []) if isinstance(s, dict)],
        speech_rate=r.speech_rate,
        visual_activity=r.visual_activity,
        token_estimate=r.token_estimate,
    )


def row_to_keyframe(r: KeyframeRow) -> Keyframe:
    return Keyframe(
        id=r.id,
        scene_index=r.scene_index,
        timestamp=r.timestamp,
        path=r.path,
        width=r.width,
        height=r.height,
        phash=int(r.phash or 0),
        quality=r.quality,
        sharpness=r.sharpness,
        entropy=r.entropy,
        text_density=r.text_density,
        is_slide=r.is_slide,
    )


# --------------------------------------------------------------- repositories
class VideoRepo:
    """All reads/writes for a video and its derived artefacts."""

    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def create(self, **fields: Any) -> VideoRow:
        row = VideoRow(**fields)
        self.s.add(row)
        await self.s.flush()
        return row

    async def get(self, video_id: str) -> VideoRow | None:
        return await self.s.get(VideoRow, video_id)

    async def by_sha(self, sha: str) -> VideoRow | None:
        res = await self.s.execute(select(VideoRow).where(VideoRow.sha256 == sha).limit(1))
        return res.scalar_one_or_none()

    async def list_videos(
        self, *, limit: int = 100, offset: int = 0, status: str | None = None
    ) -> Sequence[VideoRow]:
        q = select(VideoRow).order_by(VideoRow.created_at.desc()).limit(limit).offset(offset)
        if status:
            q = q.where(VideoRow.status == status)
        return (await self.s.execute(q)).scalars().all()

    async def count(self) -> int:
        return int((await self.s.execute(select(func.count(VideoRow.id)))).scalar() or 0)

    async def update(self, video_id: str, **fields: Any) -> None:
        row = await self.get(video_id)
        if row is None:
            return
        for k, v in fields.items():
            setattr(row, k, v)
        row.updated_at = utcnow()
        await self.s.flush()

    async def delete(self, video_id: str) -> None:
        for table in (ChunkRow, KeyframeRow, SegmentRow, SceneRow):
            await self.s.execute(delete(table).where(table.video_id == video_id))
        await self.s.execute(delete(VideoRow).where(VideoRow.id == video_id))

    # ------------------------------------------------------------------ bulk
    async def replace_chunks(self, video_id: str, chunks: list[VideoChunk]) -> None:
        await self.s.execute(delete(ChunkRow).where(ChunkRow.video_id == video_id))
        self.s.add_all([ChunkRow(**chunk_to_row(c)) for c in chunks])
        await self.s.flush()

    async def replace_keyframes(self, video_id: str, frames: list[Keyframe]) -> None:
        await self.s.execute(delete(KeyframeRow).where(KeyframeRow.video_id == video_id))
        self.s.add_all(
            [
                KeyframeRow(
                    id=k.id,
                    video_id=video_id,
                    scene_index=k.scene_index,
                    timestamp=k.timestamp,
                    path=k.path,
                    width=k.width,
                    height=k.height,
                    phash=str(k.phash),
                    quality=k.quality,
                    sharpness=k.sharpness,
                    entropy=k.entropy,
                    text_density=k.text_density,
                    is_slide=k.is_slide,
                )
                for k in frames
            ]
        )
        await self.s.flush()

    async def replace_segments(self, video_id: str, segments: list[TranscriptSegment]) -> None:
        await self.s.execute(delete(SegmentRow).where(SegmentRow.video_id == video_id))
        self.s.add_all(
            [
                SegmentRow(
                    id=f"{video_id}:{seg.index}",
                    video_id=video_id,
                    idx=seg.index,
                    start=seg.span.start,
                    end=seg.span.end,
                    text=seg.text,
                    speaker=seg.speaker or "",
                    confidence=seg.confidence,
                    words=[w.model_dump() for w in seg.words],
                )
                for seg in segments
            ]
        )
        await self.s.flush()

    async def replace_scenes(self, video_id: str, scenes: list[Scene]) -> None:
        await self.s.execute(delete(SceneRow).where(SceneRow.video_id == video_id))
        self.s.add_all(
            [
                SceneRow(
                    id=f"{video_id}:{sc.index}",
                    video_id=video_id,
                    idx=sc.index,
                    start=sc.span.start,
                    end=sc.span.end,
                    cut_score=sc.cut_score,
                    static_ratio=sc.static_ratio,
                    kind=sc.kind,
                )
                for sc in scenes
            ]
        )
        await self.s.flush()

    # ------------------------------------------------------------------ reads
    async def chunks(self, video_id: str) -> list[VideoChunk]:
        res = await self.s.execute(select(ChunkRow).where(ChunkRow.video_id == video_id).order_by(ChunkRow.idx))
        return [row_to_chunk(r) for r in res.scalars().all()]

    async def chunks_by_ids(self, ids: Sequence[str]) -> dict[str, VideoChunk]:
        if not ids:
            return {}
        res = await self.s.execute(select(ChunkRow).where(ChunkRow.id.in_(list(ids))))
        return {r.id: row_to_chunk(r) for r in res.scalars().all()}

    async def keyframes(self, video_id: str) -> list[Keyframe]:
        res = await self.s.execute(
            select(KeyframeRow).where(KeyframeRow.video_id == video_id).order_by(KeyframeRow.timestamp)
        )
        return [row_to_keyframe(r) for r in res.scalars().all()]

    async def keyframes_by_ids(self, ids: Sequence[str]) -> dict[str, Keyframe]:
        if not ids:
            return {}
        res = await self.s.execute(select(KeyframeRow).where(KeyframeRow.id.in_(list(ids))))
        return {r.id: row_to_keyframe(r) for r in res.scalars().all()}

    async def segments(self, video_id: str) -> list[SegmentRow]:
        res = await self.s.execute(
            select(SegmentRow).where(SegmentRow.video_id == video_id).order_by(SegmentRow.idx)
        )
        return list(res.scalars().all())

    async def scenes(self, video_id: str) -> list[SceneRow]:
        res = await self.s.execute(select(SceneRow).where(SceneRow.video_id == video_id).order_by(SceneRow.idx))
        return list(res.scalars().all())

    async def log_query(self, **fields: Any) -> None:
        payload = fields.pop("payload", {})
        self.s.add(QueryRow(payload=json.loads(json.dumps(payload, default=str)), **fields))
        await self.s.flush()

    async def recent_queries(self, limit: int = 20) -> Sequence[QueryRow]:
        res = await self.s.execute(select(QueryRow).order_by(QueryRow.created_at.desc()).limit(limit))
        return res.scalars().all()


class SessionRepo:
    """Conversations and their turns."""

    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    async def ensure(self, session_id: str, *, video_ids: list[str], title: str) -> SessionRow:
        row = await self.s.get(SessionRow, session_id)
        if row is None:
            row = SessionRow(id=session_id, title=title[:300], video_ids=video_ids)
            self.s.add(row)
            await self.s.flush()
        return row

    async def history(self, session_id: str, *, limit: int = 6) -> list[TurnRow]:
        """Most recent turns, oldest first."""
        res = await self.s.execute(
            select(TurnRow).where(TurnRow.session_id == session_id).order_by(TurnRow.idx.desc()).limit(limit)
        )
        return list(reversed(res.scalars().all()))

    async def append(self, session_id: str, **fields: Any) -> TurnRow:
        session = await self.s.get(SessionRow, session_id)
        index = session.turn_count if session else 0
        turn = TurnRow(id=f"{session_id}:{index}", session_id=session_id, idx=index, **fields)
        self.s.add(turn)
        if session is not None:
            session.turn_count = index + 1
            session.updated_at = utcnow()
        await self.s.flush()
        return turn

    async def list_sessions(self, limit: int = 50) -> Sequence[SessionRow]:
        res = await self.s.execute(select(SessionRow).order_by(SessionRow.updated_at.desc()).limit(limit))
        return res.scalars().all()

    async def get(self, session_id: str) -> SessionRow | None:
        return await self.s.get(SessionRow, session_id)

    async def delete(self, session_id: str) -> None:
        await self.s.execute(delete(TurnRow).where(TurnRow.session_id == session_id))
        await self.s.execute(delete(SessionRow).where(SessionRow.id == session_id))
