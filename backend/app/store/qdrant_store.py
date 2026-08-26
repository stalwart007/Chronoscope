"""Qdrant backend using named vectors and payload indexes.

One collection holds chunk points with ``text`` and ``summary`` vectors; a
second holds keyframe points with an ``image`` vector and a ``chunk_id`` link.
Qdrant requires UUID or integer point ids, so the 128-bit blake2b identifiers
are encoded as UUIDs with the original kept in the payload.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np

from app.config import settings
from app.core.errors import DependencyUnavailable
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

_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


def to_point_id(raw: str) -> str:
    try:
        return str(uuid.UUID(hex=raw))
    except (ValueError, AttributeError):
        return str(uuid.uuid5(_NAMESPACE, str(raw)))


class QdrantVectorStore(VectorStore):
    name = "qdrant"

    def __init__(self) -> None:
        try:
            from qdrant_client import AsyncQdrantClient
        except ImportError as exc:  # pragma: no cover
            raise DependencyUnavailable("qdrant-client is not installed", detail=str(exc)) from exc
        self._client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            prefer_grpc=settings.qdrant_prefer_grpc,
            timeout=30,
        )
        self.chunks_coll = settings.qdrant_collection
        self.frames_coll = f"{settings.qdrant_collection}_frames"
        self._ready = False

    # ------------------------------------------------------------------ setup
    async def ensure_schema(self, dims: dict[str, int]) -> None:
        from qdrant_client import models as qm

        text_dim = dims.get(TEXT) or dims.get(SUMMARY)
        image_dim = dims.get(IMAGE)
        hnsw = qm.HnswConfigDiff(m=settings.hnsw_m, ef_construct=settings.hnsw_ef_construction)
        optimizers = qm.OptimizersConfigDiff(default_segment_number=2, indexing_threshold=10_000)

        existing = {c.name for c in (await self._client.get_collections()).collections}
        if text_dim and self.chunks_coll not in existing:
            await self._client.create_collection(
                self.chunks_coll,
                vectors_config={
                    TEXT: qm.VectorParams(size=text_dim, distance=qm.Distance.COSINE, hnsw_config=hnsw),
                    SUMMARY: qm.VectorParams(size=text_dim, distance=qm.Distance.COSINE, hnsw_config=hnsw),
                },
                optimizers_config=optimizers,
            )
            await self._index_payload(self.chunks_coll, {"video_id": "keyword", "speakers": "keyword", "start": "float", "end": "float"})
            log.info("created collection %s (dim=%d)", self.chunks_coll, text_dim)

        if image_dim and self.frames_coll not in existing:
            await self._client.create_collection(
                self.frames_coll,
                vectors_config={IMAGE: qm.VectorParams(size=image_dim, distance=qm.Distance.COSINE, hnsw_config=hnsw)},
                optimizers_config=optimizers,
            )
            await self._index_payload(self.frames_coll, {"video_id": "keyword", "chunk_id": "keyword", "timestamp": "float"})
            log.info("created collection %s (dim=%d)", self.frames_coll, image_dim)
        self._ready = True

    async def _index_payload(self, collection: str, fields: dict[str, str]) -> None:
        from qdrant_client import models as qm

        kinds = {
            "keyword": qm.PayloadSchemaType.KEYWORD,
            "float": qm.PayloadSchemaType.FLOAT,
            "integer": qm.PayloadSchemaType.INTEGER,
        }
        for field_name, kind in fields.items():
            try:
                await self._client.create_payload_index(collection, field_name=field_name, field_schema=kinds[kind])
            except Exception as exc:
                log.debug("payload index %s.%s: %s", collection, field_name, exc)

    # ----------------------------------------------------------------- upsert
    async def upsert_chunks(self, points: list[ChunkPoint]) -> int:
        from qdrant_client import models as qm

        batch = []
        for p in points:
            vectors = {k: np.asarray(v, dtype=np.float32).tolist() for k, v in p.vectors.items() if k in (TEXT, SUMMARY)}
            if not vectors:
                continue
            batch.append(
                qm.PointStruct(
                    id=to_point_id(p.id),
                    vector=vectors,
                    payload={
                        "chunk_id": p.id,
                        "video_id": p.video_id,
                        "start": p.start,
                        "end": p.end,
                        "index": p.index,
                        "speakers": p.speakers,
                        **p.payload,
                    },
                )
            )
        for i in range(0, len(batch), 128):
            await self._client.upsert(self.chunks_coll, points=batch[i : i + 128], wait=False)
        return len(batch)

    async def upsert_frames(self, points: list[FramePoint]) -> int:
        from qdrant_client import models as qm

        batch = [
            qm.PointStruct(
                id=to_point_id(p.id),
                vector={IMAGE: np.asarray(p.vector, dtype=np.float32).tolist()},
                payload={
                    "frame_id": p.id,
                    "chunk_id": p.chunk_id,
                    "video_id": p.video_id,
                    "timestamp": p.timestamp,
                    "start": p.timestamp,
                    "end": p.timestamp,
                    **p.payload,
                },
            )
            for p in points
            if p.vector is not None
        ]
        for i in range(0, len(batch), 128):
            await self._client.upsert(self.frames_coll, points=batch[i : i + 128], wait=False)
        return len(batch)

    # ----------------------------------------------------------------- search
    def _build_filter(self, flt: SearchFilter | None, *, frames: bool) -> Any:
        from qdrant_client import models as qm

        if flt is None or flt.is_empty:
            return None
        must: list[Any] = []
        if flt.video_ids:
            must.append(qm.FieldCondition(key="video_id", match=qm.MatchAny(any=flt.video_ids)))
        if flt.speakers and not frames:
            must.append(qm.FieldCondition(key="speakers", match=qm.MatchAny(any=flt.speakers)))
        if flt.start is not None:
            must.append(qm.FieldCondition(key="end", range=qm.Range(gte=flt.start)))
        if flt.end is not None:
            must.append(qm.FieldCondition(key="start", range=qm.Range(lte=flt.end)))
        return qm.Filter(must=must) if must else None

    async def search(
        self, modality: str, vector: np.ndarray, *, limit: int = 32, flt: SearchFilter | None = None
    ) -> list[VectorHit]:
        frames = modality == IMAGE
        collection = self.frames_coll if frames else self.chunks_coll
        try:
            res = await self._client.query_points(
                collection_name=collection,
                query=np.asarray(vector, dtype=np.float32).tolist(),
                using=modality,
                limit=limit,
                with_payload=True,
                query_filter=self._build_filter(flt, frames=frames),
                search_params=self._search_params(),
            )
            points = res.points
        except Exception as exc:
            log.warning("qdrant search failed on %s/%s: %s", collection, modality, exc)
            return []
        out: list[VectorHit] = []
        for p in points:
            payload = dict(p.payload or {})
            ident = payload.get("frame_id" if frames else "chunk_id") or str(p.id)
            out.append(
                VectorHit(
                    id=ident,
                    score=float(p.score),
                    payload=payload,
                    parent_id=payload.get("chunk_id") if frames else ident,
                )
            )
        return out

    def _search_params(self) -> Any:
        from qdrant_client import models as qm

        return qm.SearchParams(hnsw_ef=settings.hnsw_ef_search, exact=False)

    # ------------------------------------------------------------------ admin
    async def delete_video(self, video_id: str) -> None:
        from qdrant_client import models as qm

        selector = qm.FilterSelector(
            filter=qm.Filter(must=[qm.FieldCondition(key="video_id", match=qm.MatchValue(value=video_id))])
        )
        for coll in (self.chunks_coll, self.frames_coll):
            try:
                await self._client.delete(coll, points_selector=selector, wait=True)
            except Exception as exc:
                log.warning("delete from %s failed: %s", coll, exc)

    async def fetch_vectors(self, ids: list[str], modality: str) -> dict[str, np.ndarray]:
        if not ids:
            return {}
        frames = modality == IMAGE
        collection = self.frames_coll if frames else self.chunks_coll
        try:
            recs = await self._client.retrieve(
                collection, ids=[to_point_id(i) for i in ids], with_vectors=True, with_payload=True
            )
        except Exception as exc:
            log.warning("retrieve failed: %s", exc)
            return {}
        out: dict[str, np.ndarray] = {}
        for r in recs:
            vec = (r.vector or {}).get(modality) if isinstance(r.vector, dict) else r.vector
            if vec is None:
                continue
            key = (r.payload or {}).get("frame_id" if frames else "chunk_id") or str(r.id)
            out[str(key)] = np.asarray(vec, dtype=np.float32)
        return out

    async def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {"backend": self.name, "collections": {}}
        for coll in (self.chunks_coll, self.frames_coll):
            try:
                info = await self._client.get_collection(coll)
                out["collections"][coll] = {
                    "points": info.points_count,
                    # Current clients expose indexed_vectors_count; it stays None
                    # until the optimiser has built the index at least once.
                    "vectors": getattr(info, "indexed_vectors_count", None),
                    "status": str(info.status),
                }
            except Exception as exc:
                out["collections"][coll] = {"error": str(exc)}
        return out

    async def close(self) -> None:
        await self._client.close()
