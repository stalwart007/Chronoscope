"""Vector store interface with parent-child semantics.

A chunk carries two text-space vectors: ``text`` for verbatim phrasing and
``summary`` for conceptual queries. A keyframe carries one CLIP vector,
``image``. Frames are searched independently and max-pooled onto their parent,
so a chunk is as visually relevant as its single best frame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

TEXT = "text"
SUMMARY = "summary"
IMAGE = "image"
MODALITIES = (TEXT, SUMMARY, IMAGE)


@dataclass(slots=True)
class ChunkPoint:
    id: str
    video_id: str
    start: float
    end: float
    index: int
    speakers: list[str] = field(default_factory=list)
    vectors: dict[str, np.ndarray] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FramePoint:
    id: str
    chunk_id: str
    video_id: str
    timestamp: float
    vector: np.ndarray | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchFilter:
    video_ids: list[str] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    start: float | None = None
    end: float | None = None

    def matches(self, payload: dict[str, Any]) -> bool:
        if self.video_ids and payload.get("video_id") not in self.video_ids:
            return False
        if self.speakers:
            spk = payload.get("speakers") or ([payload["speaker"]] if payload.get("speaker") else [])
            if not set(spk) & set(self.speakers):
                return False
        if self.start is not None and float(payload.get("end", 0.0)) < self.start:
            return False
        return not (self.end is not None and float(payload.get("start", 0.0)) > self.end)

    @property
    def is_empty(self) -> bool:
        return not self.video_ids and not self.speakers and self.start is None and self.end is None


@dataclass(slots=True)
class VectorHit:
    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)
    #: For child hits, the parent chunk this frame rolls up into.
    parent_id: str | None = None


class VectorStore(ABC):
    """Backend contract. Both implementations are interchangeable at runtime."""

    name: str = "base"

    @abstractmethod
    async def ensure_schema(self, dims: dict[str, int]) -> None: ...

    @abstractmethod
    async def upsert_chunks(self, points: list[ChunkPoint]) -> int: ...

    @abstractmethod
    async def upsert_frames(self, points: list[FramePoint]) -> int: ...

    @abstractmethod
    async def search(
        self, modality: str, vector: np.ndarray, *, limit: int = 32, flt: SearchFilter | None = None
    ) -> list[VectorHit]: ...

    @abstractmethod
    async def delete_video(self, video_id: str) -> None: ...

    @abstractmethod
    async def fetch_vectors(self, ids: list[str], modality: str) -> dict[str, np.ndarray]: ...

    @abstractmethod
    async def stats(self) -> dict[str, Any]: ...

    async def health(self) -> dict[str, Any]:
        try:
            return {"backend": self.name, "ok": True, **(await self.stats())}
        except Exception as exc:
            return {"backend": self.name, "ok": False, "error": str(exc)}

    async def close(self) -> None:  # pragma: no cover - optional
        return None


def rollup_children(hits: list[VectorHit]) -> list[VectorHit]:
    """Max-pool child hits onto their parents, preserving the best child id."""
    best: dict[str, VectorHit] = {}
    for h in hits:
        pid = h.parent_id or h.id
        cur = best.get(pid)
        if cur is None or h.score > cur.score:
            best[pid] = VectorHit(id=pid, score=h.score, payload={**h.payload, "best_frame": h.id}, parent_id=pid)
    return sorted(best.values(), key=lambda h: -h.score)
