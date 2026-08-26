"""Vector store selection with graceful degradation.

Qdrant is used when reachable. Otherwise the engine falls back to the
in-process index and records the reason for the health endpoint.
"""

from __future__ import annotations

from app.config import settings
from app.logging_conf import get_logger
from app.store.base import VectorStore
from app.store.memory_store import MemoryVectorStore

log = get_logger(__name__)

_store: VectorStore | None = None
_fallback_reason: str = ""


async def get_store(dims: dict[str, int] | None = None) -> VectorStore:
    global _store
    if _store is None:
        _store = await _build()
    if dims:
        await _store.ensure_schema(dims)
    return _store


async def _build() -> VectorStore:
    global _fallback_reason
    if settings.vector_backend == "qdrant":
        try:
            from app.store.qdrant_store import QdrantVectorStore

            store = QdrantVectorStore()
            await store._client.get_collections()
            log.info("vector backend: qdrant @ %s", settings.qdrant_url)
            return store
        except Exception as exc:
            _fallback_reason = f"qdrant unavailable ({exc.__class__.__name__}: {exc})"
            log.warning("%s, falling back to in-process HNSW store", _fallback_reason)
    log.info("vector backend: in-process HNSW")
    return MemoryVectorStore()


def fallback_reason() -> str:
    return _fallback_reason


async def reset_store() -> None:
    global _store
    if _store is not None:
        await _store.close()
    _store = None
