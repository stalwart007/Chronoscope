"""Request and response models for the HTTP surface."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.core.types import (
    AnswerBundle,
    Citation,
    Keyframe,
    RetrievalTrace,
    Scene,
    ScoredHit,
    TimeSpan,
    VideoChunk,
)


class VideoSummary(BaseModel):
    id: str
    filename: str
    title: str
    status: str
    stage: str = ""
    progress: float = 0.0
    error: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    size_bytes: int = 0
    language: str = ""
    summary: str = ""
    speakers: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    chapters: list[ChapterOut] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    poster: str | None = None
    created_at: str = ""


class VideoListResponse(BaseModel):
    videos: list[VideoSummary]
    total: int


class SegmentOut(BaseModel):
    index: int
    start: float
    end: float
    text: str
    speaker: str = ""
    confidence: float = 0.0


class ChapterOut(BaseModel):
    index: int
    start: float
    end: float
    title: str
    keywords: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    speakers: list[str] = Field(default_factory=list)
    boundary_strength: float = 0.0


def parse_chapters(raw: list[Any] | None) -> list[ChapterOut]:
    """Chapters are stored as JSON, so validate on the way out."""
    out: list[ChapterOut] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        try:
            out.append(ChapterOut.model_validate(item))
        except Exception:
            continue
    return out


class TimelineResponse(BaseModel):
    video: VideoSummary
    chapters: list[ChapterOut] = Field(default_factory=list)
    scenes: list[Scene]
    keyframes: list[Keyframe]
    chunks: list[VideoChunk]
    segments: list[SegmentOut]


class UploadResponse(BaseModel):
    video_id: str
    status: str
    duplicate: bool = False
    message: str = ""


class UrlIngestRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2048)
    title: str = ""


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    #: Continue an existing conversation. Omit to start a new one.
    session_id: str = Field(default="", max_length=64)
    #: Ids are validated at the edge so a crafted filter can never reach a
    #: path join, and the list is bounded so one request cannot fan out into
    #: hundreds of index scans.
    video_ids: list[str] = Field(default_factory=list, max_length=50)
    speakers: list[str] = Field(default_factory=list, max_length=20)
    time_range: TimeSpan | None = None
    top_k: int = Field(default=8, ge=1, le=32)

    _ids = field_validator("video_ids")(lambda cls, v: _clean_ids(v))
    _spk = field_validator("speakers")(lambda cls, v: _clean_speakers(v))


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    video_ids: list[str] = Field(default_factory=list, max_length=50)
    speakers: list[str] = Field(default_factory=list, max_length=20)
    time_range: TimeSpan | None = None
    top_k: int = Field(default=10, ge=1, le=50)
    candidates: int = Field(default=48, ge=8, le=256)
    modalities: list[str] = Field(default_factory=lambda: ["text", "summary", "image", "lexical"])
    mmr_lambda: float | None = Field(default=None, ge=0.0, le=1.0)
    use_temporal_fusion: bool = True

    _ids = field_validator("video_ids")(lambda cls, v: _clean_ids(v))
    _spk = field_validator("speakers")(lambda cls, v: _clean_speakers(v))


class TurnOut(BaseModel):
    index: int
    query: str
    resolved_query: str = ""
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    elapsed_ms: float = 0.0
    created_at: str = ""


class SessionOut(BaseModel):
    id: str
    title: str = ""
    video_ids: list[str] = Field(default_factory=list)
    turn_count: int = 0
    created_at: str = ""
    updated_at: str = ""


class SessionDetail(BaseModel):
    session: SessionOut
    turns: list[TurnOut] = Field(default_factory=list)


class SearchResponse(BaseModel):
    hits: list[ScoredHit]
    trace: RetrievalTrace


def _clean_ids(values: list[str]) -> list[str]:
    from app.core.security import is_valid_id

    return [v for v in values if is_valid_id(v)]


def _clean_speakers(values: list[str]) -> list[str]:
    import re

    return [re.sub(r"[^A-Za-z0-9_\- ]", "", v)[:64] for v in values if v.strip()]


class AnswerResponse(BaseModel):
    answer: AnswerBundle


class HealthResponse(BaseModel):
    status: str
    version: str
    env: str
    encoders: dict[str, Any]
    vector_store: dict[str, Any]
    llm: dict[str, Any]
    jobs: dict[str, Any]
    lexical: dict[str, Any]
    degraded: list[str] = Field(default_factory=list)
