"""Video-level summarisation, used by the final ingestion stage."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from app.core.types import VideoChunk, fmt_ts
from app.ingest.align import tokenize_text
from app.llm.base import Message
from app.llm.router import router
from app.logging_conf import get_logger

log = get_logger(__name__)

SCHEMA = """{"summary":"3-4 sentence abstract","topics":["topic one","topic two"]}"""


def video_topics(chunks: list[VideoChunk], limit: int = 8) -> list[str]:
    """Corpus-level topics via class-based TF-IDF over the full transcript.

    The video is treated as a single document. A term scores by how often it is
    said, damped sub-linearly so a repeated word cannot dominate, multiplied by
    how broadly it spreads across chunks. Terms already surfaced as chunk
    keywords get a small boost.

    Ranking by chunk count alone promotes filler, since an opening greeting
    appears as often as the subject matter.
    """
    if not chunks:
        return []
    n = len(chunks)
    tf: Counter[str] = Counter()
    df: Counter[str] = Counter()
    keyword_hits: Counter[str] = Counter()
    for c in chunks:
        tokens = [w for w in tokenize_text(c.text or c.summary) if len(w) > 3]
        tf.update(tokens)
        df.update(set(tokens))
        keyword_hits.update({k for k in c.keywords if " " not in k})

    def score(term: str) -> float:
        spread = (df[term] / n) ** 0.5
        boost = 1.0 + 0.35 * min(keyword_hits[term], 3)
        return (1.0 + math.log(tf[term])) * spread * boost

    # Prefer terms that recur; only relax when the video is too short to have any.
    pool = [t for t in tf if tf[t] >= 2] or list(tf)
    ranked = sorted(pool, key=lambda t: (-score(t), t))
    picked: list[str] = []
    for term in ranked:
        if any(term in other or other in term for other in picked):
            continue
        picked.append(term)
        if len(picked) >= limit:
            break
    return picked


def extractive_summary(chunks: list[VideoChunk], speakers: list[str]) -> tuple[str, list[str]]:
    """TextRank-lite: pick chunks whose keywords are most central to the video.

    Centrality is computed on the keyword bipartite projection, a chunk scores
    by how much of the video's overall keyword mass it covers, normalised by
    length so a long rambling chunk cannot dominate.
    """
    if not chunks:
        return "", []
    global_kw: Counter[str] = Counter()
    for c in chunks:
        global_kw.update(c.keywords)
    topics = video_topics(chunks)
    scores = {
        c.id: sum(global_kw[k] for k in c.keywords) / (1.0 + 0.15 * len(c.keywords))
        for c in chunks
    }
    picked = sorted(sorted(chunks, key=lambda c: -scores[c.id])[:3], key=lambda c: c.index)
    lead = " ".join(
        f"[{fmt_ts(c.span.start)}] {(c.text or c.summary).strip()[:190]}" for c in picked
    )
    who = f"{len(speakers)} speaker(s): {', '.join(speakers)}. " if speakers else ""
    span = chunks[-1].span.end
    return (
        f"{who}{len(chunks)} indexed segments covering {fmt_ts(span)}. Key moments, {lead}",
        topics,
    )


async def summarize_video(
    chunks: list[VideoChunk], speakers: list[str], stats: dict[str, Any]
) -> tuple[str, list[str], str]:
    fallback, topics = extractive_summary(chunks, speakers)
    if not chunks:
        return fallback, topics, "none"
    transcript = "\n".join(
        f"[{fmt_ts(c.span.start)}] {(', '.join(c.speakers) + ': ') if c.speakers else ''}{c.text[:400]}"
        for c in chunks[:40]
    )[:9000]
    try:
        data = await router.chat_json(
            [
                Message(
                    role="user",
                    content=(
                        "Summarise this video transcript for a search index. "
                        "Be concrete: name the systems, numbers and decisions mentioned.\n\n" + transcript
                    ),
                )
            ],
            schema_hint=SCHEMA,
            max_tokens=520,
        )
    except Exception as exc:
        log.info("video summarisation used extractive fallback: %s", exc)
        return fallback, topics, "extractive"
    summary = str(data.get("summary", "")).strip()
    llm_topics = [str(t)[:48] for t in (data.get("topics") or []) if isinstance(t, (str, int))][:8]
    return (summary or fallback), (llm_topics or topics), str(data.get("_meta", {}).get("model", "llm"))
