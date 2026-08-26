"""Shared API helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.api.schemas import VideoSummary, parse_chapters
from app.store.db import VideoRow


def iso_utc(value: datetime | None) -> str:
    """Serialise as unambiguous UTC.

    SQLite has no datetime type and returns naive datetimes, whose isoformat
    carries no offset. Browsers then read the value as local time. Stamping the
    timezone here keeps SQLite and Postgres consistent.
    """
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def to_summary(row: VideoRow, *, poster_exists: bool = True) -> VideoSummary:
    return VideoSummary(
        id=row.id,
        filename=row.filename,
        title=row.title or row.filename,
        status=row.status,
        stage=row.stage,
        progress=row.progress,
        error=row.error,
        duration=row.duration,
        width=row.width,
        height=row.height,
        fps=row.fps,
        size_bytes=row.size_bytes,
        language=row.language,
        summary=row.summary,
        speakers=[str(s) for s in (row.speakers or [])],
        topics=[str(t) for t in (row.topics or [])],
        chapters=parse_chapters(row.chapters),
        stats=dict(row.stats or {}),
        poster=f"/artifacts/{row.id}/poster.jpg" if poster_exists else None,
        created_at=iso_utc(row.created_at),
    )


def sse(event: str, data: Any) -> str:
    import json

    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"
