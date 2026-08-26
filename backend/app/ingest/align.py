"""Audio-visual alignment: turning parallel streams into retrievable chunks.

Transcript segments, speaker turns, scenes and keyframes each have their own
timing. The output is a list of chunks, each holding a coherent piece of speech
and the frames that were on screen while it was said.

Boundaries are chosen by dynamic programming over candidate cut points (scene
changes, speaker changes, sentence ends, silences), balancing deviation from
the target length against how natural each cut is.
"""

from __future__ import annotations

import bisect
import itertools
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config import settings
from app.core.interval_tree import IntervalTree
from app.core.types import (
    Keyframe,
    Scene,
    Sentence,
    SpeakerTurn,
    TimeSpan,
    TranscriptSegment,
    VideoChunk,
    stable_id,
)
from app.ingest.scenes import ContentSignal
from app.logging_conf import get_logger

log = get_logger(__name__)

_SENTENCE_END = re.compile(r"[.!?...]['\")\]]?\s*$")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'\-]+")
_STOP = frozenset(
    ["the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on", "at", "for", "with", "by", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that", "these", "those", "as", "from", "into", "about", "over", "under", "he", "she", "they", "we", "you", "i", "not", "no", "do", "does", "did", "so", "such", "than", "there", "here", "what", "when", "where", "which", "who", "whom", "will", "would", "can", "could", "should", "may", "might", "must", "have", "has", "had", "our", "your", "their", "my", "me", "us", "them", "very", "just", "also", "more", "most", "some", "any", "all", "one", "two", "three", "now", "well", "okay", "right", "like", "get", "got", "go", "going", "know", "think", "say", "said", "how", "why", "let", "look", "see", "lets", "going", "want", "need", "make", "made", "take", "put", "use", "used", "using", "thing", "things", "really", "actually"]
)

# Attractiveness of each kind of boundary, how "natural" a cut there feels.
Q_SCENE = 1.00
Q_SPEAKER = 0.95
Q_SENTENCE = 0.62
Q_SILENCE = 0.45
LAMBDA = 0.85


@dataclass(slots=True)
class AlignmentInputs:
    video_id: str
    duration: float
    segments: list[TranscriptSegment]
    turns: list[SpeakerTurn]
    scenes: list[Scene]
    keyframes: list[Keyframe]
    signal: ContentSignal | None = None


def tokenize_text(text: str) -> list[str]:
    """Content words only, the shared tokenizer for keywords and topics."""
    return [w for w in (m.lower() for m in _WORD.findall(text or "")) if w not in _STOP and len(w) > 2]


def candidate_boundaries(inp: AlignmentInputs) -> list[tuple[float, float, str]]:
    """``(time, attractiveness, kind)`` for every plausible cut point."""
    cands: dict[float, tuple[float, str]] = {}

    def offer(t: float, q: float, kind: str) -> None:
        t = round(max(0.0, min(t, inp.duration)), 3)
        cur = cands.get(t)
        if cur is None or q > cur[0]:
            cands[t] = (q, kind)

    for sc in inp.scenes[1:]:
        offer(sc.span.start, Q_SCENE, "scene")
    prev_speaker: str | None = None
    for turn in sorted(inp.turns, key=lambda t: t.span.start):
        if prev_speaker is not None and turn.speaker != prev_speaker:
            offer(turn.span.start, Q_SPEAKER, "speaker")
        prev_speaker = turn.speaker
    for i, seg in enumerate(inp.segments):
        if _SENTENCE_END.search(seg.text or ""):
            offer(seg.span.end, Q_SENTENCE, "sentence")
        if i + 1 < len(inp.segments):
            gap = inp.segments[i + 1].span.start - seg.span.end
            if gap > 0.45:
                offer(seg.span.end + gap / 2.0, min(Q_SILENCE + gap * 0.06, 0.85), "silence")
    return sorted((t, q, kind) for t, (q, kind) in cands.items())


def optimal_cuts(
    boundaries: list[tuple[float, float, str]],
    duration: float,
    *,
    target: float,
    max_len: float,
    lambda_: float = LAMBDA,
) -> list[float]:
    """DP over candidate boundaries; returns the chosen interior cut times."""
    points = [0.0, *[t for t, _, _ in boundaries if 0.0 < t < duration], duration]
    quality = {0.0: 1.0, duration: 1.0}
    for t, q, _ in boundaries:
        quality[round(t, 3)] = q
    n = len(points)
    if n <= 2:
        return []

    INF = float("inf")
    cost = [INF] * n
    parent = [-1] * n
    cost[0] = 0.0
    min_len = max(1.5, target * 0.28)
    for j in range(1, n):
        # Only predecessors within [j - max_len, j - min_len] can be optimal.
        lo = bisect.bisect_left(points, points[j] - max_len)
        hi = bisect.bisect_right(points, points[j] - min_len)
        span_lo, span_hi = max(0, lo), max(0, hi)
        if span_hi <= span_lo:  # nothing legal, allow the nearest predecessor
            span_lo, span_hi = max(0, j - 1), j
        for i in range(span_lo, min(span_hi, j)):
            if cost[i] == INF:
                continue
            d = points[j] - points[i]
            if d <= 0:
                continue
            length_cost = ((d - target) / target) ** 2
            cut_cost = 0.0 if j == n - 1 else lambda_ * (1.0 - quality.get(round(points[j], 3), 0.3))
            total = cost[i] + length_cost + cut_cost
            if total < cost[j]:
                cost[j], parent[j] = total, i
    if cost[n - 1] == INF:  # degenerate input, fall back to uniform windows
        steps = max(1, math.ceil(duration / target))
        return [round(duration * i / steps, 3) for i in range(1, steps)]
    out: list[float] = []
    j = n - 1
    while parent[j] > 0:
        j = parent[j]
        out.append(points[j])
    return sorted(out)


def extract_keywords(texts: list[str], *, top_k: int = 6) -> list[list[str]]:
    """Per-chunk TF-IDF keywords over unigrams and bigrams.

    IDF is computed within the video, so terms the speaker repeats constantly
    ("the model", "our system") are discounted while locally distinctive terms
    ("kubernetes", "reciprocal rank") float to the top.
    """
    docs: list[Counter[str]] = []
    for text in texts:
        toks = tokenize_text(text)
        c: Counter[str] = Counter(toks)
        c.update(f"{a} {b}" for a, b in itertools.pairwise(toks))
        docs.append(c)
    n = max(1, len(docs))
    df: Counter[str] = Counter()
    for d in docs:
        df.update(d.keys())
    out: list[list[str]] = []
    for d in docs:
        if not d:
            out.append([])
            continue
        total = sum(d.values())
        scored = {}
        for term, tf in d.items():
            is_bigram = " " in term
            # A bigram earns its slot only if it actually recurs; otherwise
            # every adjacent word pair in the chunk scores like a phrase.
            if is_bigram and tf < 2:
                continue
            if df[term] >= n > 1:
                continue
            weight = (1.0 + math.log(tf)) / math.log(2.0 + total)
            scored[term] = weight * math.log((n + 1) / (df[term] + 0.5)) * (1.4 if is_bigram else 1.0)
        ranked = sorted(scored, key=lambda t: -scored[t])
        chosen: list[str] = []
        for term in ranked:
            # Skip a unigram already covered by a chosen bigram, and vice versa.
            if any(term in c or c in term for c in chosen):
                continue
            chosen.append(term)
            if len(chosen) >= top_k:
                break
        out.append(chosen)
    return out


def split_sentences(core: list[Any], span: TimeSpan) -> list[Sentence]:
    """Split the chunk's transcript into timed, speaker-attributed sentences.

    Word timings give exact boundaries when Whisper provided them; otherwise
    the segment's duration is apportioned by character length, which is well
    within the tolerance a viewer notices when seeking.
    """
    out: list[Sentence] = []
    for iv in core:
        seg = iv.payload
        text = (seg.text or "").strip()
        if not text:
            continue
        pieces = [p.strip() for p in re.split(r"(?<=[.!?...])\s+", text) if p.strip()]
        if len(pieces) <= 1:
            out.append(
                Sentence(start=round(max(seg.span.start, span.start - 0.01), 3),
                         end=round(seg.span.end, 3), text=text, speaker=seg.speaker)
            )
            continue
        words = seg.words
        if words and len(words) >= len(text.split()) * 0.6:
            cursor = 0
            for piece in pieces:
                n = len(piece.split())
                window = words[cursor : cursor + n] or words[-1:]
                out.append(
                    Sentence(start=round(window[0].start, 3), end=round(window[-1].end, 3),
                             text=piece, speaker=seg.speaker)
                )
                cursor += n
        else:
            lengths = [len(p) + 1 for p in pieces]
            total = sum(lengths)
            t = seg.span.start
            for piece, length in zip(pieces, lengths, strict=True):
                dt = seg.span.duration * length / total
                out.append(
                    Sentence(start=round(t, 3), end=round(t + dt, 3), text=piece, speaker=seg.speaker)
                )
                t += dt
    return out


def _describe_visual(frames: list[Keyframe]) -> str:
    if not frames:
        return "no distinct visual"
    slides = sum(1 for f in frames if f.is_slide)
    if slides >= max(1, len(frames) // 2):
        return "slide or diagram on screen"
    if float(np.mean([f.text_density for f in frames])) > 0.04:
        return "text-heavy visual on screen"
    if float(np.mean([f.entropy for f in frames])) > 6.0:
        return "detailed camera shot"
    return "presenter shot"


def build_chunks(inp: AlignmentInputs) -> list[VideoChunk]:
    seg_tree: IntervalTree[TranscriptSegment] = IntervalTree(
        (s.span.start, s.span.end, s) for s in inp.segments if s.span.duration > 0
    )
    turn_tree: IntervalTree[SpeakerTurn] = IntervalTree((t.span.start, t.span.end, t) for t in inp.turns)
    scene_tree: IntervalTree[Scene] = IntervalTree((s.span.start, s.span.end, s) for s in inp.scenes)
    frame_times = [k.timestamp for k in sorted(inp.keyframes, key=lambda k: k.timestamp)]
    frames_sorted = sorted(inp.keyframes, key=lambda k: k.timestamp)

    boundaries = candidate_boundaries(inp)
    # Short clips need proportionally shorter chunks: a 60-second demo split
    # into three 20-second windows retrieves at the granularity of "the whole
    # video", which is not useful. Scale down until ~10 chunks exist, then hold.
    target = min(settings.chunk_target_s, max(6.0, inp.duration / 10.0))
    max_len = min(settings.chunk_max_s, max(target * 1.9, 12.0))
    cuts = optimal_cuts(boundaries, inp.duration, target=target, max_len=max_len)
    edges = [0.0, *cuts, inp.duration]
    overlap = settings.chunk_overlap_s

    raw: list[tuple[TimeSpan, str, str, list[str], list[str], list[int], float, float, list[Sentence]]] = []
    for i in range(len(edges) - 1):
        span = TimeSpan(start=round(edges[i], 3), end=round(edges[i + 1], 3))
        if span.duration < 0.4:
            continue
        core = seg_tree.query(span.start, span.end)
        wide = seg_tree.query(span.start - overlap, span.end + overlap)
        text = " ".join(iv.payload.text for iv in core if iv.payload.text).strip()
        context = " ".join(iv.payload.text for iv in wide if iv.payload.text).strip()

        speakers: list[str] = []
        for turn_iv in turn_tree.query(span.start, span.end):
            if turn_iv.overlap(span.start, span.end) > 0.3 and turn_iv.payload.speaker not in speakers:
                speakers.append(turn_iv.payload.speaker)
        for seg_iv in core:  # transcript-level labels win when present
            spk = seg_iv.payload.speaker
            if spk and spk not in speakers:
                speakers.append(spk)

        lo = bisect.bisect_left(frame_times, span.start)
        hi = bisect.bisect_left(frame_times, span.end)
        kf = frames_sorted[lo:hi]
        if not kf and frames_sorted:  # always give a chunk something to show
            nearest = min(frames_sorted, key=lambda k: abs(k.timestamp - span.mid))
            kf = [nearest]
        scene_idx = sorted({iv.payload.index for iv in scene_tree.query(span.start, span.end)})

        words = sum(len(s.payload.words) or len(s.payload.text.split()) for s in core)
        speech_rate = round(words / max(span.duration, 1e-3), 3)
        activity = 0.0
        if inp.signal is not None and len(inp.signal):
            mask = (inp.signal.times >= span.start) & (inp.signal.times < span.end)
            if mask.any():
                activity = round(float(inp.signal.values[mask].mean()), 3)
        sentences = split_sentences(core, span)
        raw.append((span, text, context, speakers, [k.id for k in kf], scene_idx, speech_rate, activity, sentences))

    keywords = extract_keywords([txt or ctx for _, txt, ctx, *_ in raw])
    frames_by_id = {k.id: k for k in inp.keyframes}
    chunks: list[VideoChunk] = []
    for idx, (span, text, context, speakers, kf_ids, scene_idx, rate, activity, sentences) in enumerate(raw):
        kws = keywords[idx]
        visual = _describe_visual([frames_by_id[i] for i in kf_ids if i in frames_by_id])
        head = (text or context or "").strip()
        lead = head[:180] + ("..." if len(head) > 180 else "")
        summary = "; ".join(
            part
            for part in (
                f"{', '.join(speakers)} speaking" if speakers else "",
                visual,
                f"topics: {', '.join(kws[:4])}" if kws else "",
                lead,
            )
            if part
        )
        chunks.append(
            VideoChunk(
                id=stable_id(inp.video_id, "chunk", idx, round(span.start, 2)),
                video_id=inp.video_id,
                index=idx,
                span=span,
                text=text or context,
                summary=summary,
                speakers=speakers,
                keyframe_ids=kf_ids,
                scene_indices=scene_idx,
                keywords=kws,
                sentences=sentences,
                speech_rate=rate,
                visual_activity=activity,
                token_estimate=int(len(text or context) / 4) + 1,
            )
        )
    log.info(
        "built %d chunks (mean %.1fs) from %d boundary candidates",
        len(chunks),
        float(np.mean([c.span.duration for c in chunks])) if chunks else 0.0,
        len(boundaries),
    )
    return chunks
