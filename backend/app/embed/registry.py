"""Lazily constructed, process-wide encoder singletons.

Models are loaded once on first use, never at import time. If a model fails to
load the registry falls back to the lightweight encoders and records why, which
is surfaced by the health endpoint.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from app.core.concurrency import TTLCache, run_blocking
from app.embed.base import ImageEncoder, TextEncoder
from app.embed.fallback import ColorLayoutImageEncoder, HashingTextEncoder
from app.logging_conf import get_logger
from app.store.base import IMAGE, SUMMARY, TEXT

log = get_logger(__name__)

_text: TextEncoder | None = None
_image: ImageEncoder | None = None
_notes: dict[str, str] = {}
_lock = asyncio.Lock()

_query_cache: TTLCache[np.ndarray] = TTLCache(maxsize=1024, ttl=1800)


def _load_text() -> TextEncoder:
    try:
        from app.embed.models import SentenceTransformerEncoder

        return SentenceTransformerEncoder()
    except Exception as exc:
        _notes["text"] = f"{type(exc).__name__}: {exc}"
        log.warning("text model unavailable (%s), using hashing sketch encoder", exc)
        return HashingTextEncoder()


def _load_image() -> ImageEncoder:
    try:
        from app.embed.models import ClipEncoder

        return ClipEncoder()
    except Exception as exc:
        _notes["image"] = f"{type(exc).__name__}: {exc}"
        log.warning("CLIP unavailable (%s), using colour-layout descriptor", exc)
        return ColorLayoutImageEncoder()


async def text_encoder() -> TextEncoder:
    global _text
    if _text is None:
        async with _lock:
            if _text is None:
                _text = await run_blocking(_load_text)
    return _text


async def image_encoder() -> ImageEncoder:
    global _image
    if _image is None:
        async with _lock:
            if _image is None:
                _image = await run_blocking(_load_image)
    return _image


async def warm_up() -> dict[str, Any]:
    t, i = await asyncio.gather(text_encoder(), image_encoder())
    return {"text": t.info(), "image": i.info(), "notes": dict(_notes)}


async def dims() -> dict[str, int]:
    t, i = await asyncio.gather(text_encoder(), image_encoder())
    return {TEXT: t.dim, SUMMARY: t.dim, IMAGE: i.dim}


async def embed_texts(texts: list[str], *, query: bool = False) -> np.ndarray:
    enc = await text_encoder()
    fn = getattr(enc, "encode_query", None) if query else None
    return await run_blocking(fn or enc.encode, texts)


async def embed_images(paths: list[str]) -> np.ndarray:
    enc = await image_encoder()
    return await run_blocking(enc.encode_images, paths)


async def embed_clip_text(texts: list[str]) -> np.ndarray:
    enc = await image_encoder()
    return await run_blocking(enc.encode_text, texts)


async def embed_query_multimodal(query: str) -> dict[str, np.ndarray]:
    """One query -> three vectors, cached and single-flighted.

    The text query is encoded twice: once by the sentence encoder (for the
    ``text``/``summary`` towers) and once by CLIP's text tower (for the
    ``image`` tower). They live in different spaces and must never be mixed.
    """

    async def build() -> np.ndarray:
        t_vec, c_vec = await asyncio.gather(embed_texts([query], query=True), embed_clip_text([query]))
        return np.concatenate([t_vec[0], c_vec[0]])

    enc_t = await text_encoder()
    packed = await _query_cache.get_or_set(f"{enc_t.dim}:{query}", build)
    split = enc_t.dim
    t_vec, c_vec = packed[:split], packed[split:]
    return {TEXT: t_vec, SUMMARY: t_vec, IMAGE: c_vec}


def cache_stats() -> dict[str, Any]:
    return _query_cache.stats()


