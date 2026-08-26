"""Topic segmentation over chunks, producing chapters.

A long recording is a flat list of hundreds of chunks with no structure above
the scene. Scenes track the picture, not the subject: a speaker can change
topic without the camera moving, and can change slides three times inside one
argument. Chapters give the coarse layer that navigation actually needs.

The method is lexical cohesion in embedding space, following Hearst's
TextTiling. For every gap between consecutive chunks, the mean embedding of the
preceding window is compared with the following window; a topic shift shows up
as a dip in that similarity. Raw dips are noisy, so each is scored by *depth*,
how far it falls below the nearest peak on each side, and only dips deeper than
a robust threshold become boundaries.

Depth rather than absolute similarity matters because different material sits
at different baseline similarities: a technical talk stays lexically tight
throughout, a panel wanders. Depth is relative to the local baseline, so the
same threshold works for both.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from app.core.types import VideoChunk

#: Chunks on each side of a candidate gap. Wider is steadier but blurs short
#: topics; three chunks is roughly a minute of speech at default sizing.
WINDOW = 3
MIN_CHAPTER_CHUNKS = 2


@dataclass(slots=True)
class Chapter:
    index: int
    start: float
    end: float
    title: str
    keywords: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    #: How sharply the topic changed at this chapter's start, 0-1.
    boundary_strength: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index, "start": round(self.start, 3), "end": round(self.end, 3),
            "title": self.title, "keywords": self.keywords, "chunk_ids": self.chunk_ids,
            "speakers": self.speakers, "boundary_strength": round(self.boundary_strength, 4),
        }


def cohesion_scores(vectors: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Similarity across each gap between consecutive chunks."""
    n = vectors.shape[0]
    if n < 2:
        return np.zeros(0, dtype=np.float32)
    unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    scores = np.zeros(n - 1, dtype=np.float32)
    for gap in range(n - 1):
        left = unit[max(0, gap + 1 - window) : gap + 1].mean(axis=0)
        right = unit[gap + 1 : min(n, gap + 1 + window)].mean(axis=0)
        left /= max(float(np.linalg.norm(left)), 1e-9)
        right /= max(float(np.linalg.norm(right)), 1e-9)
        scores[gap] = float(left @ right)
    return scores


def depth_scores(scores: np.ndarray) -> np.ndarray:
    """How far each dip falls below the nearest peak on either side.

    Walking outward from a gap until the curve stops rising gives the local
    peak on that side. The sum of the two drops is the depth, which is what
    separates a real topic shift from a slightly-less-similar pair of chunks.
    """
    n = scores.shape[0]
    depths = np.zeros(n, dtype=np.float32)
    for i in range(n):
        left = i
        while left > 0 and scores[left - 1] >= scores[left]:
            left -= 1
        right = i
        while right < n - 1 and scores[right + 1] >= scores[right]:
            right += 1
        depths[i] = (scores[left] - scores[i]) + (scores[right] - scores[i])
    return depths


def find_boundaries(scores: np.ndarray, *, sensitivity: float = 0.55) -> list[int]:
    """Gap indices that begin a new chapter."""
    if scores.shape[0] < 2:
        return []
    depths = depth_scores(scores)
    mean = float(depths.mean())
    sd = float(depths.std())
    threshold = mean + sensitivity * sd
    picked: list[int] = []
    for i, d in enumerate(depths):
        if d <= threshold or d <= 1e-6:
            continue
        # Only the deepest gap in a neighbourhood: a shift often produces two
        # or three adjacent dips, which would otherwise become empty chapters.
        lo, hi = max(0, i - 1), min(depths.shape[0], i + 2)
        if d >= depths[lo:hi].max():
            picked.append(i)
    return picked


def _title_from(keywords: list[str], text: str) -> str:
    if keywords:
        return ", ".join(k for k in keywords[:3])
    head = " ".join(text.split()[:8])
    return head or "Untitled section"


def segment(chunks: list[VideoChunk], vectors: dict[str, np.ndarray], *, sensitivity: float = 0.55) -> list[Chapter]:
    """Group chunks into chapters and label each from its own vocabulary."""
    ordered = sorted(chunks, key=lambda c: c.index)
    if len(ordered) < 3:
        if not ordered:
            return []
        return [_build(0, ordered, 0.0)]

    usable = [c for c in ordered if c.id in vectors]
    if len(usable) < 3:
        return [_build(0, ordered, 0.0)]

    matrix = np.stack([vectors[c.id] for c in usable]).astype(np.float32)
    scores = cohesion_scores(matrix)
    cuts = find_boundaries(scores, sensitivity=sensitivity)

    groups: list[list[VideoChunk]] = []
    strengths: list[float] = []
    start = 0
    depths = depth_scores(scores) if scores.size else np.zeros(0, dtype=np.float32)
    for cut in cuts:
        end = cut + 1
        if end - start < MIN_CHAPTER_CHUNKS or len(usable) - end < MIN_CHAPTER_CHUNKS:
            continue
        groups.append(usable[start:end])
        strengths.append(float(depths[cut]) if cut < depths.shape[0] else 0.0)
        start = end
    groups.append(usable[start:])
    strengths.append(0.0)

    # Boundary strength belongs to the chapter that starts at it.
    shifted = [0.0, *strengths[:-1]]
    peak = max(shifted) or 1.0
    return [
        _build(i, group, min(1.0, shifted[i] / peak))
        for i, group in enumerate(groups)
        if group
    ]


def _build(index: int, group: list[VideoChunk], strength: float) -> Chapter:
    counts: Counter[str] = Counter()
    for c in group:
        counts.update(k for k in c.keywords if " " not in k)
    # Prefer terms that recur inside the chapter, damped so one repeated word
    # cannot take every slot.
    ranked = sorted(counts, key=lambda k: (-(1.0 + math.log(counts[k])), k))
    keywords = ranked[:6]
    speakers: list[str] = []
    for c in group:
        for s in c.speakers:
            if s not in speakers:
                speakers.append(s)
    text = " ".join(c.text for c in group)[:400]
    return Chapter(
        index=index,
        start=group[0].span.start,
        end=group[-1].span.end,
        title=_title_from(keywords, text),
        keywords=keywords,
        chunk_ids=[c.id for c in group],
        speakers=speakers,
        boundary_strength=strength,
    )
