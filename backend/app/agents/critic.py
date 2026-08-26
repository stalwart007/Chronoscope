"""Coverage check between retrieval and synthesis.

The graph retrieves once and answers. That is fine when the first query happens
to match the vocabulary of the recording, and poor when it does not: a question
phrased in the user's words against a transcript phrased in the speaker's gets
one mediocre shot and no second chance.

This node inspects what came back and decides whether another retrieval round
is worth it, then proposes queries for it. The signals are deliberately cheap
and model-free, so the behaviour is identical with or without an LLM:

* weak absolute scores, or a shallow margin between the best hit and the rest
* plan sub-tasks whose vocabulary never appears in anything retrieved
* a numeric or chart question that produced no numbers to compute on

Follow-up queries are built by pivoting on the vocabulary that *was* found,
which is how a second round escapes the phrasing of the first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.agents.state import AgentState
from app.core.types import TaskKind

MAX_ROUNDS = 2
_WORD = re.compile(r"[a-z0-9][a-z0-9'-]{2,}")
_STOP = frozenset(
    """the a an and or but if then of to in on at for with by is are was were be been being it its this that
    these those as from into about over under what when where which who how why does did do said say show
    shows tell me you your their there here more most some any all can could would should will""".split()
)


def terms(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP}


@dataclass(slots=True)
class Verdict:
    should_retry: bool
    reason: str = ""
    queries: list[str] = field(default_factory=list)
    coverage: float = 0.0
    margin: float = 0.0


def assess(state: AgentState) -> Verdict:
    """Decide whether the evidence answers the question."""
    if state.round >= MAX_ROUNDS:
        return Verdict(False, "retry budget spent")
    if not state.hits:
        # Nothing at all: broaden by dropping the rarest words, which are the
        # most likely to be absent from the transcript.
        return Verdict(True, "no evidence retrieved", queries=_broaden(state.query), coverage=0.0)

    wanted = terms(state.query)
    found: set[str] = set()
    for hit in state.hits:
        if hit.chunk:
            found |= terms(hit.chunk.text)
            found |= {k.lower() for k in hit.chunk.keywords}
    coverage = len(wanted & found) / len(wanted) if wanted else 1.0

    top = state.hits[0].score
    rest = [h.score for h in state.hits[1:4]]
    margin = (top - sum(rest) / len(rest)) / top if rest and top > 1e-9 else 0.0

    missing = sorted(wanted - found)
    needs_numbers = any(t.kind in {TaskKind.COMPUTATION, TaskKind.CHART_EXTRACTION} for t in state.plan.tasks)
    no_numbers = needs_numbers and not state.computations

    if coverage >= 0.6 and margin >= 0.04 and not no_numbers:
        return Verdict(False, "evidence covers the question", coverage=coverage, margin=margin)

    reasons = []
    if coverage < 0.6:
        reasons.append(f"{len(missing)} query terms absent from the evidence")
    if margin < 0.04:
        reasons.append("no clear best match")
    if no_numbers:
        reasons.append("a computation was planned but no numbers were found")

    return Verdict(
        True,
        "; ".join(reasons),
        queries=_followups(state, missing, found),
        coverage=coverage,
        margin=margin,
    )


def _broaden(query: str) -> list[str]:
    words = [w for w in _WORD.findall(query.lower()) if w not in _STOP]
    if len(words) <= 2:
        return [query]
    # Longest words are usually the most specific, and the most likely to be
    # the ones the speaker never said.
    trimmed = sorted(words, key=len)[: max(2, len(words) - 2)]
    return [" ".join(trimmed)]


def _followups(state: AgentState, missing: list[str], found: set[str]) -> list[str]:
    """Queries that approach the same question from the evidence's vocabulary."""
    out: list[str] = []
    anchors = [
        k
        for hit in state.hits[:3]
        if hit.chunk
        for k in hit.chunk.keywords
        if " " not in k
    ]
    anchor_terms = list(dict.fromkeys(anchors))[:4]

    if missing and anchor_terms:
        # Pair what the question asked for with what the recording actually
        # says, so the second round is not a rephrasing of the first.
        out.append(" ".join(missing[:3] + anchor_terms[:3]))
    if missing:
        out.append(" ".join(missing[:5]))
    if any(t.kind in {TaskKind.COMPUTATION, TaskKind.CHART_EXTRACTION} for t in state.plan.tasks):
        out.append(" ".join([*anchor_terms[:3], "numbers", "figures", "percent", "total"]))
    if not out:
        out.append(" ".join(sorted(found)[:6]) or state.query)

    deduped: list[str] = []
    for candidate in out:
        cleaned = candidate.strip()
        if cleaned and cleaned.lower() != state.query.lower() and cleaned not in deduped:
            deduped.append(cleaned)
    return deduped[:2]
