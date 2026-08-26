"""Dependency-free vector backend built on the in-repo HNSW index.

Selected with ``CS_VECTOR_BACKEND=memory``. Persists under
``data/artifacts/index`` so a restart does not force a re-embed. Index work runs
in a worker thread to keep it off the event loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings
from app.core.concurrency import run_blocking
from app.core.hnsw import HNSW
from app.logging_conf import get_logger
from app.store.base import (
    IMAGE,
    SUMMARY,
    TEXT,
    ChunkPoint,
    FramePoint,
    SearchFilter,
    VectorHit,
    VectorStore,
)

log = get_logger(__name__)


class MemoryVectorStore(VectorStore):
    name = "memory"

    def __init__(self, persist_dir: Path | None = None) -> None:
        self._indexes: dict[str, HNSW] = {}
        self._dims: dict[str, int] = {}
        self._dir = persist_dir or (settings.artifact_dir / "index")
        self._video_keys: dict[str, dict[str, set[str]]] = {}
        self._dirty = False

    # ------------------------------------------------------------------ setup
    async def ensure_schema(self, dims: dict[str, int]) -> None:
        for modality, dim in dims.items():
            if modality in self._indexes:
                continue
            path = self._dir / modality
            if path.with_suffix(".json").exists():
                try:
                    idx = await run_blocking(HNSW.load, path)
                    if idx.dim == dim:
                        self._indexes[modality] = idx
                        self._dims[modality] = dim
                        self._reindex_video_keys(modality, idx)
                        log.info("loaded %s index (%d vectors)", modality, len(idx))
                        continue
                    log.warning("dim mismatch for %s (%d != %d), rebuilding", modality, idx.dim, dim)
                except Exception as exc:
                    log.warning("could not load %s index: %s", modality, exc)
            self._indexes[modality] = HNSW(
                dim, m=settings.hnsw_m, ef_construction=settings.hnsw_ef_construction, ef_search=settings.hnsw_ef_search
            )
            self._dims[modality] = dim

    def _reindex_video_keys(self, modality: str, idx: HNSW) -> None:
        for meta in idx._meta:
            if meta.deleted:
                continue
            vid = meta.payload.get("video_id")
            if vid:
                self._video_keys.setdefault(vid, {}).setdefault(modality, set()).add(meta.key)

    def _track(self, modality: str, video_id: str, key: str) -> None:
        self._video_keys.setdefault(video_id, {}).setdefault(modality, set()).add(key)

    # ----------------------------------------------------------------- upsert
    async def upsert_chunks(self, points: list[ChunkPoint]) -> int:
        def work() -> int:
            n = 0
            for p in points:
                for modality in (TEXT, SUMMARY):
                    vec = p.vectors.get(modality)
                    if vec is None or modality not in self._indexes:
                        continue
                    payload = {
                        "video_id": p.video_id,
                        "chunk_id": p.id,
                        "start": p.start,
                        "end": p.end,
                        "index": p.index,
                        "speakers": p.speakers,
                        **p.payload,
                    }
                    self._indexes[modality].add(p.id, vec, payload)
                    self._track(modality, p.video_id, p.id)
                    n += 1
            return n

        count = await run_blocking(work)
        self._dirty = True
        return count

    async def upsert_frames(self, points: list[FramePoint]) -> int:
        def work() -> int:
            idx = self._indexes.get(IMAGE)
            if idx is None:
                return 0
            n = 0
            for p in points:
                if p.vector is None:
                    continue
                idx.add(
                    p.id,
                    p.vector,
                    {
                        "video_id": p.video_id,
                        "chunk_id": p.chunk_id,
                        "frame_id": p.id,
                        "timestamp": p.timestamp,
                        "start": p.timestamp,
                        "end": p.timestamp,
                        **p.payload,
                    },
                )
                self._track(IMAGE, p.video_id, p.id)
                n += 1
            return n

        count = await run_blocking(work)
        self._dirty = True
        return count

    # ----------------------------------------------------------------- search
    async def search(
        self, modality: str, vector: np.ndarray, *, limit: int = 32, flt: SearchFilter | None = None
    ) -> list[VectorHit]:
        idx = self._indexes.get(modality)
        if idx is None or len(idx) == 0:
            return []
        predicate = None
        if flt is not None and not flt.is_empty:
            predicate = lambda _k, payload: flt.matches(payload)  # noqa: E731

        def work() -> list[VectorHit]:
            raw = idx.search(vector, k=limit, predicate=predicate)
            return [
                VectorHit(
                    id=key,
                    score=score,
                    payload=payload,
                    parent_id=payload.get("chunk_id") if modality == IMAGE else key,
                )
                for key, score, payload in raw
            ]

        return await run_blocking(work)

    # ----------------------------------------------------------------- admin
    async def delete_video(self, video_id: str) -> None:
        def work() -> None:
            for modality, keys in self._video_keys.pop(video_id, {}).items():
                idx = self._indexes.get(modality)
                if idx is None:
                    continue
                for k in keys:
                    idx.remove(k)

        await run_blocking(work)
        self._dirty = True

    async def fetch_vectors(self, ids: list[str], modality: str) -> dict[str, np.ndarray]:
        idx = self._indexes.get(modality)
        if idx is None:
            return {}
        out = {}
        for i in ids:
            v = idx.get(i)
            if v is not None:
                out[i] = v
        return out

    async def flush(self, *, force: bool = False) -> None:
        """Persist the indexes if anything changed.

        Saving is a full compressed rewrite of every vector, so doing it after
        each upsert made ingestion cost O(index size) per batch instead of
        O(batch). Writes are coalesced behind a dirty flag and driven by the
        background checkpointer, plus an unconditional flush at shutdown.
        """
        if not (self._dirty or force):
            return
        self._dirty = False

        def work() -> None:
            for modality, idx in self._indexes.items():
                idx.save(self._dir / modality)

        try:
            await run_blocking(work)
        except Exception:
            self._dirty = True  # keep it queued rather than losing the write
            raise

    async def close(self) -> None:
        await self.flush(force=True)

    async def stats(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "collections": {m: idx.stats() for m, idx in self._indexes.items()},
            "videos": len(self._video_keys),
        }
