"""Answer synthesis.

Context assembly is the parent half of parent-child retrieval: the index found
small records, and here the full transcript window, speaker labels, frame
descriptions and computed values are pulled into a budgeted prompt. Chunks are
added in relevance order until the budget is reached, each trimmed around its
most query-relevant sentence.

Without a model the answer is extractive: the highest-scoring sentences with
timestamps, speaker attribution and computed figures.
"""

from __future__ import annotations

import re
from typing import Any

from app.agents.conversation import summarise_history
from app.agents.state import AgentState
from app.core.types import ScoredHit, Sentence, fmt_ts
from app.llm.base import Message
from app.llm.router import router
from app.logging_conf import get_logger

log = get_logger(__name__)

CONTEXT_TOKEN_BUDGET = 2600
_SENT = re.compile(r"(?<=[.!?])\s+")

SYSTEM = (
    "You are Chronoscope, a video analytics assistant. Answer ONLY from the provided evidence. "
    "Every factual claim must carry a timestamp citation in the form [mm:ss]. "
    "If the evidence does not answer the question, say so plainly and state what is missing. "
    "Never invent numbers: if a computation block is present, use its values verbatim."
)


def _score_sentence(sentence: str, terms: set[str]) -> float:
    words = {w.lower().strip(".,!?;:") for w in sentence.split()}
    if not words:
        return 0.0
    return len(words & terms) / (len(words) ** 0.5)


def rank_sentences(state: AgentState, *, limit: int = 6) -> list[tuple[Sentence, ScoredHit, float]]:
    """Best individual utterances across the retrieved chunks.

    Retrieval works at chunk granularity, but an answer should quote, and
    seek to, the exact sentence. Sentences are scored by query-term overlap,
    discounted by their parent chunk's rank, filtered by any speaker
    constraint, and de-duplicated (chunk overlap means the same sentence can
    appear in two neighbouring chunks).
    """
    terms = {w.lower().strip(".,!?;:'\"") for w in state.query.split() if len(w) > 2}
    wanted = set(state.speakers)
    scored: list[tuple[Sentence, ScoredHit, float]] = []
    for rank, hit in enumerate(state.hits):
        if hit.chunk is None:
            continue
        decay = 1.0 / (1.0 + 0.4 * rank)
        pool = hit.chunk.sentences or [
            Sentence(start=hit.chunk.span.start, end=hit.chunk.span.end,
                     text=hit.chunk.text or hit.chunk.summary,
                     speaker=(hit.chunk.speakers[0] if hit.chunk.speakers else None))
        ]
        for sent in pool:
            if wanted and sent.speaker and sent.speaker not in wanted:
                continue
            if len(sent.text.split()) < 3:
                continue
            scored.append((sent, hit, decay * (0.25 + _score_sentence(sent.text, terms))))
    scored.sort(key=lambda t: -t[2])
    out: list[tuple[Sentence, ScoredHit, float]] = []
    seen: set[str] = set()
    for sent, hit, score in scored:
        key = " ".join(sent.text.lower().split())[:90]
        if key in seen:
            continue
        seen.add(key)
        out.append((sent, hit, score))
        if len(out) >= limit:
            break
    return out


def _trim(text: str, query: str, *, max_chars: int) -> str:
    """Keep the window around the most query-relevant sentence."""
    if len(text) <= max_chars:
        return text
    sentences = _SENT.split(text)
    terms = {w.lower().strip(".,!?;:") for w in query.split() if len(w) > 2}
    best = max(range(len(sentences)), key=lambda i: _score_sentence(sentences[i], terms))
    out, lo, hi = sentences[best], best - 1, best + 1
    while len(out) < max_chars and (lo >= 0 or hi < len(sentences)):
        if lo >= 0 and len(out) + len(sentences[lo]) < max_chars:
            out = sentences[lo] + " " + out
            lo -= 1
        elif hi < len(sentences) and len(out) + len(sentences[hi]) < max_chars:
            out = out + " " + sentences[hi]
            hi += 1
        else:
            break
    return out.strip()


def build_context(state: AgentState) -> str:
    blocks: list[str] = []
    budget = CONTEXT_TOKEN_BUDGET
    for i, hit in enumerate(state.hits, start=1):
        if budget <= 0 or hit.chunk is None:
            break
        c = hit.chunk
        share = max(280, int(budget * 0.45)) * 4  # chars ~ 4 x tokens
        if c.sentences:
            # Inline per-sentence timestamps so the model can cite precisely
            # instead of guessing a time from the chunk header.
            lines: list[str] = []
            used = 0
            for sent in c.sentences:
                line = f"({fmt_ts(sent.start)}) {(sent.speaker + ': ') if sent.speaker else ''}{sent.text}"
                if used + len(line) > share:
                    break
                lines.append(line)
                used += len(line)
            body = "\n".join(lines) or _trim(c.text or c.summary, state.query, max_chars=share)
        else:
            body = _trim(c.text or c.summary, state.query, max_chars=share)
        speakers = ", ".join(c.speakers) or "unattributed"
        visual = ""
        frames = [f for f in state.visual_findings if f.get("chunk_id") == c.id]
        if frames:
            visual = " | on screen: " + "; ".join(
                f"{f['description'][:120]}{(' text: ' + f['on_screen_text'][:120]) if f.get('on_screen_text') else ''}"
                for f in frames[:2]
            )
        blocks.append(
            f"[E{i}] {fmt_ts(c.span.start)}-{fmt_ts(c.span.end)} | speaker: {speakers}{visual}\n{body}"
        )
        budget -= len(body) // 4 + 40
    if state.computations:
        for comp in state.computations:
            res = comp.get("result", {})
            blocks.append(
                "[COMPUTATION] "
                + f"series={comp.get('series', {})} -> {res.get('value', res.get('variables'))} "
                + f"({comp.get('explanation', '')})"
            )
    return "\n\n".join(blocks)


def extractive_answer(state: AgentState) -> str:
    """Deterministic answer built from the evidence itself."""
    if not state.hits:
        return "No matching moments were found in the indexed videos for that question."
    picks = rank_sentences(state, limit=4)
    lines = [
        f"- [{fmt_ts(sent.start)}] {(sent.speaker + ': ') if sent.speaker else ''}{sent.text.strip()[:260]}"
        for sent, _hit, _score in sorted(picks, key=lambda t: t[0].start)
    ]
    if not lines:
        lines = [
            f"- [{fmt_ts(h.chunk.span.start)}] {(h.chunk.text or h.chunk.summary)[:240]}"
            for h in state.hits[:3]
            if h.chunk
        ]
    head = f'Found {len(state.hits)} relevant moment(s) for "{state.query}".'
    body = "\n".join(lines)
    extras: list[str] = []
    for f in state.visual_findings[:2]:
        detail = f.get("on_screen_text") or f.get("description", "")
        if detail:
            extras.append(f"- [{f.get('timestamp_label', '?')}] on screen: {detail[:200]}")
    for comp in state.computations:
        res = comp.get("result", {})
        value = res.get("value")
        if value is None:
            value = res.get("variables")
        series = comp.get("series", {})
        if series.get("labels"):
            extras.append("- series: " + ", ".join(
                f"{name}={value:,.6g}"
                for name, value in zip(series["labels"], series["values"], strict=False)
            ))
        extras.append(f"- computed: {value}")
    tail = ("\n\n" + "\n".join(extras)) if extras else ""
    note = "\n\n(Generated without a language model, evidence extracted directly from the index.)"
    return f"{head}\n\n{body}{tail}{note}"


def estimate_confidence(state: AgentState) -> float:
    """Blend retrieval margin, modality agreement and evidence volume."""
    if not state.hits:
        return 0.0
    top = state.hits[0].score
    second = state.hits[1].score if len(state.hits) > 1 else 0.0
    margin = (top - second) / top if top > 1e-9 else 0.0
    modalities = len(state.hits[0].ranks) / 4.0
    volume = min(1.0, len(state.hits) / 5.0)
    computed = 0.12 if state.computations else 0.0
    visual = 0.08 if state.visual_findings else 0.0
    degraded = 0.85 if not state.llm_available else 1.0
    raw = 0.34 * min(1.0, margin * 3.0) + 0.3 * modalities + 0.2 * volume + computed + visual
    return round(min(0.97, raw) * degraded, 4)


def refine_citations(state: AgentState) -> list[Any]:
    """Re-point citations at the exact sentence that matched."""
    from app.core.types import Citation

    picks = rank_sentences(state, limit=6)
    if not picks:
        return state.citations
    return [
        Citation(
            chunk_id=hit.chunk_id,
            video_id=hit.video_id,
            start=sent.start,
            end=sent.end,
            speaker=sent.speaker,
            keyframe=(hit.keyframes[0].path if hit.keyframes else None),
            quote=sent.text[:300],
            relevance=round(score, 6),
        )
        for sent, hit, score in picks
    ]


async def synthesize(state: AgentState) -> dict[str, Any]:
    confidence = estimate_confidence(state)
    if not state.hits:
        return {
            "answer": "Nothing in the indexed videos matches that question. Try broadening it, "
            "or check that the relevant video finished processing.",
            "confidence": 0.0,
        }
    citations = refine_citations(state)
    if not state.llm_available:
        return {
            "answer": extractive_answer(state),
            "confidence": confidence,
            "model_used": "extractive",
            "citations": citations,
        }

    context = build_context(state)
    style = {
        "timestamped": "Answer in 2-4 sentences, each claim followed by its [mm:ss] citation.",
        "narrative": "Answer as a short structured summary with bullet points, each with [mm:ss].",
        "table": "Answer with a compact markdown table of the extracted values, then one sentence of commentary.",
        "numeric": "State the computed figure first, then one sentence of derivation, then citations.",
    }[state.plan.answer_style]
    try:
        prior = summarise_history(state.history) if state.history else ""
        preamble = f"Earlier in this conversation:\n{prior}\n\n" if prior else ""
        res = await router.chat(
            [
                Message(role="system", content=SYSTEM),
                Message(
                    role="user",
                    content=f"{preamble}Question: {state.query}\n\nEvidence:\n{context}\n\n{style}",
                ),
            ],
            max_tokens=700,
            temperature=0.2,
        )
        answer = res.text.strip()
        if not answer:
            raise ValueError("empty completion")
        return {
            "answer": answer,
            "confidence": confidence,
            "model_used": f"{res.provider}/{res.model}",
            "citations": citations,
        }
    except Exception as exc:
        log.warning("synthesis fell back to extractive: %s", exc)
        return {
            "answer": extractive_answer(state),
            "confidence": round(confidence * 0.9, 4),
            "model_used": "extractive",
            "citations": citations,
            "notes": [f"LLM synthesis failed: {exc}"],
        }
