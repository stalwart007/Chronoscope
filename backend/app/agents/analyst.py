"""Arithmetic over extracted evidence, executed rather than estimated.

Language models are unreliable calculators, so the model writes the calculation
and a sandboxed interpreter runs it. Numeric evidence comes from chart data
points and from numbers mined out of the retrieved transcript. Without a model,
a standard analysis runs over the same series.
"""

from __future__ import annotations

from typing import Any

from app.agents.numeric import NumericFact, Series, build_series, extract_numbers
from app.agents.state import AgentState
from app.agents.tools import repl
from app.llm.base import Message
from app.llm.router import router
from app.logging_conf import get_logger

log = get_logger(__name__)

CODE_SCHEMA = """{"code":"python statements; last line is the expression to report","explanation":"one sentence"}"""


def collect_evidence(state: AgentState) -> tuple[Series, list[NumericFact]]:
    facts: list[NumericFact] = []
    for finding in state.visual_findings:
        for p in finding.get("data_points") or []:
            try:
                facts.append(
                    NumericFact(
                        value=float(p["value"]),
                        raw=str(p.get("label", "")),
                        label=str(p.get("label", "")),
                        unit=str(p.get("unit", "")),
                        position=len(facts),
                        source="frame",
                        context=finding.get("description", "")[:120],
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    if not facts:
        facts = _transcript_facts(state)
    return build_series(facts), facts


def _transcript_facts(state: AgentState) -> list[NumericFact]:
    """Mine numbers from the retrieved speech as one continuous passage.

    Extracting per chunk would break a series that a speaker reads across
    several sentences, "...sixty eight million and the fourth quarter reached
    ninety one million" often lands in the next chunk. Chunks from the winning
    video are re-joined in timeline order first, so the label-alignment pass
    sees the list exactly as it was spoken.
    """
    hits = [h for h in state.hits if h.chunk and h.chunk.text]
    if not hits:
        return []
    target_video = hits[0].video_id
    ordered = sorted(
        (h for h in hits if h.video_id == target_video), key=lambda h: h.chunk.span.start  # type: ignore[union-attr]
    )
    passage = " ".join(h.chunk.text.strip() for h in ordered)  # type: ignore[union-attr]
    return extract_numbers(passage, source="transcript")


def standard_analysis(series: Series) -> dict[str, Any]:
    """Deterministic period-over-period analysis, no model required."""
    code = (
        "deltas = [round(b - a, 4) for a, b in zip(values, values[1:])]\n"
        "growth = growth_series(values)\n"
        "overall = round(pct_change(values[0], values[-1]), 4)\n"
        "compound = round(cagr(values[0], values[-1], len(values) - 1), 4)\n"
        "{'deltas': deltas, 'growth_pct': growth, 'overall_pct': overall, "
        "'cagr_pct': compound, 'total': round(sum(values), 4), 'mean': round(mean(values), 4)}"
    )
    result = repl.run_sync(code, {"values": series.values, "labels": series.labels})
    return {
        "kind": "standard",
        "code": code,
        "explanation": "Period-over-period change, overall growth and CAGR across the extracted series.",
        "series": series.to_dict(),
        "result": result.to_dict(),
        "source": "deterministic",
    }


async def llm_analysis(state: AgentState, series: Series) -> dict[str, Any] | None:
    try:
        data = await router.chat_json(
            [
                Message(
                    role="user",
                    content=(
                        f"Question: {state.query}\n"
                        f"Pre-bound variables: values={series.values}, labels={series.labels}, "
                        f"unit={series.unit!r}\n"
                        "Write short Python that computes exactly what the question asks. "
                        "Available helpers: mean, median, stdev, pct_change(a,b), cagr(first,last,periods), "
                        "growth_series(list), sqrt, log, round, sum, min, max, sorted, zip, enumerate.\n"
                        "No imports, no I/O, no function definitions. The final line must be the expression "
                        "whose value answers the question."
                    ),
                )
            ],
            schema_hint=CODE_SCHEMA,
            max_tokens=420,
        )
    except Exception as exc:
        log.info("analyst LLM unavailable: %s", exc)
        return None
    code = str(data.get("code") or "").strip()
    if not code:
        return None
    result = await repl.run(code, {"values": series.values, "labels": series.labels})
    if not result.ok:
        log.info("analyst code rejected (%s), falling back to standard analysis", result.error)
        return None
    return {
        "kind": "generated",
        "code": code,
        "explanation": str(data.get("explanation", ""))[:300],
        "series": series.to_dict(),
        "result": result.to_dict(),
        "source": str(data.get("_meta", {}).get("model", "llm")),
    }


async def analyse(state: AgentState) -> dict[str, Any]:
    series, facts = collect_evidence(state)
    if not series.ok:
        if not facts:
            return {"notes": ["no numeric evidence found to compute over"]}
        return {
            "computations": [
                {
                    "kind": "facts_only",
                    "series": series.to_dict(),
                    "facts": [f.to_dict() for f in facts[:12]],
                    "explanation": "Numbers were found but not enough labelled points to form a series.",
                    "source": "deterministic",
                }
            ]
        }
    computation = (await llm_analysis(state, series)) if state.llm_available else None
    if computation is None:
        computation = standard_analysis(series)
    computation["facts"] = [f.to_dict() for f in facts[:12]]
    return {"computations": [computation]}
