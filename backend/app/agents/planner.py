"""Query planning: decompose a question into typed sub-tasks.

"Find the revenue chart and calculate the growth" is a visual retrieval and an
arithmetic task, the second depending on the first. Making that structure
explicit lets the graph route work to the right specialists and skip the ones
that are not needed.

Two implementations share one interface: an LLM planner with a strict JSON
prompt, and a rule-based planner over intent cues that is always available.
"""

from __future__ import annotations

import re

from app.core.types import QueryPlan, SubTask, TaskKind, TimeSpan, stable_id
from app.llm.base import Message
from app.llm.router import router
from app.logging_conf import get_logger

log = get_logger(__name__)

VISUAL_CUES = re.compile(
    r"\b(show|shows|showing|shown|display|displayed|appear|appears|visible|on screen|onscreen|"
    r"slide|slides|diagram|chart|graph|figure|table|whiteboard|demo|screenshot|image|frame|"
    r"picture|visual|wearing|color|colour|logo|animation)\b",
    re.I,
)
TEMPORAL_CUES = re.compile(
    r"\b(when|what time|timestamp|at which point|moment|first time|last time|before|after|during|"
    r"how long|duration|earliest|latest)\b",
    re.I,
)
COMPUTE_CUES = re.compile(
    r"\b(calculate|compute|how much|how many|total|sum|average|mean|median|growth|yoy|year over year|"
    r"cagr|percent|percentage|difference|increase|decrease|ratio|rate|per|times|compare numerically)\b",
    re.I,
)
ATTRIBUTION_CUES = re.compile(r"\b(who|speaker|said|says|asked|answered|presenter|host|panelist)\b", re.I)
SUMMARY_CUES = re.compile(r"\b(summar|overview|recap|tl;?dr|key points|main points|takeaway|agenda|outline)\b", re.I)
COMPARE_CUES = re.compile(r"\b(compare|versus|vs\.?|difference between|contrast|better|worse)\b", re.I)
CHART_CUES = re.compile(r"\b(chart|graph|plot|table|axis|bar|line|pie|figures|numbers on)\b", re.I)

ORDINALS = {
    "first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3,
    "fifth": 4, "5th": 4, "last": -1, "final": -1,
}
_ORDINAL_SPEAKER = re.compile(
    r"\b(" + "|".join(ORDINALS) + r")\s+(?:speaker|person|presenter|panelist|voice)\b", re.I
)
_TIME_EXPR = re.compile(r"\b(?:after|from|since)\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\b|\b(?:before|until|up to)\s+(\d{1,2}):(\d{2})\b", re.I)

PLAN_SCHEMA = """{
  "intent": "one sentence restating what the user wants",
  "answer_style": "timestamped|narrative|table|numeric",
  "needs_vision": true,
  "needs_computation": false,
  "tasks": [
    {"id":"t1","kind":"visual_lookup|transcript_lookup|temporal_locate|speaker_attribution|chart_extraction|computation|summarize|compare",
     "query":"a self-contained retrieval query","depends_on":[],"rationale":"why",
     "modality_bias":{"image":1.6,"text":1.0}}
  ]
}"""


def parse_speaker_ordinal(query: str) -> int | None:
    m = _ORDINAL_SPEAKER.search(query)
    return ORDINALS.get(m.group(1).lower()) if m else None


def parse_time_filter(query: str) -> TimeSpan | None:
    m = _TIME_EXPR.search(query)
    if not m:
        return None
    if m.group(1):
        secs = int(m.group(1)) * 60 + int(m.group(2)) + (int(m.group(3)) if m.group(3) else 0)
        if int(m.group(1)) > 5:  # "after 10:30" in a long talk means mm:ss
            secs = int(m.group(1)) * 60 + int(m.group(2))
        return TimeSpan(start=float(secs), end=float(10**6))
    secs = int(m.group(4)) * 60 + int(m.group(5))
    return TimeSpan(start=0.0, end=float(secs))


def heuristic_plan(query: str) -> QueryPlan:
    q = query.strip()
    visual = bool(VISUAL_CUES.search(q))
    temporal = bool(TEMPORAL_CUES.search(q))
    compute = bool(COMPUTE_CUES.search(q))
    attribution = bool(ATTRIBUTION_CUES.search(q))
    chart = bool(CHART_CUES.search(q))
    summarize = bool(SUMMARY_CUES.search(q))
    compare = bool(COMPARE_CUES.search(q))

    tasks: list[SubTask] = []

    def add(kind: TaskKind, text: str, *, bias: dict[str, float], depends: list[str] | None = None, why: str = "") -> str:
        tid = f"t{len(tasks) + 1}"
        tasks.append(
            SubTask(id=tid, kind=kind, query=text, depends_on=depends or [], rationale=why, modality_bias=bias)
        )
        return tid

    retrieval_id: str
    if visual or chart:
        retrieval_id = add(
            TaskKind.VISUAL_LOOKUP, q,
            bias={"image": 1.75, "summary": 1.15, "text": 0.85, "lexical": 0.8},
            why="query references on-screen content, so frame similarity leads",
        )
    elif summarize:
        retrieval_id = add(
            TaskKind.SUMMARIZE, q,
            bias={"summary": 1.5, "text": 1.1, "image": 0.6, "lexical": 0.7},
            why="broad request, summary vectors give the widest coverage",
        )
    else:
        retrieval_id = add(
            TaskKind.TRANSCRIPT_LOOKUP, q,
            bias={"text": 1.35, "lexical": 1.25, "summary": 0.95, "image": 0.55},
            why="spoken-content question, transcript and lexical channels lead",
        )

    if attribution:
        add(
            TaskKind.SPEAKER_ATTRIBUTION, q,
            bias={"text": 1.3, "lexical": 1.2, "image": 0.4},
            depends=[retrieval_id], why="answer must name who spoke",
        )
    if temporal:
        add(
            TaskKind.TEMPORAL_LOCATE, q,
            bias={"image": 1.2, "text": 1.0},
            depends=[retrieval_id], why="answer must resolve to timestamps",
        )
    if chart:
        add(
            TaskKind.CHART_EXTRACTION, q,
            bias={"image": 1.8},
            depends=[retrieval_id], why="numbers live inside the frame, not the transcript",
        )
    if compute:
        add(
            TaskKind.COMPUTATION, q,
            bias={},
            depends=[t.id for t in tasks], why="arithmetic over the retrieved values",
        )
    if compare:
        add(TaskKind.COMPARE, q, bias={"summary": 1.2}, depends=[retrieval_id], why="two-sided comparison")

    style = "numeric" if compute else ("table" if chart else ("narrative" if summarize else "timestamped"))
    intent_bits = [k for k, on in
                   (("visual", visual), ("temporal", temporal), ("numeric", compute),
                    ("attribution", attribution), ("summary", summarize)) if on]
    return QueryPlan(
        intent=f"{q}, treated as a {'/'.join(intent_bits) or 'content'} question",
        tasks=tasks,
        needs_computation=compute,
        needs_vision=visual or chart,
        answer_style=style,  # type: ignore[arg-type]
    )


async def llm_plan(query: str, *, context: str = "") -> QueryPlan | None:
    try:
        data = await router.chat_json(
            [
                Message(
                    role="user",
                    content=(
                        "Decompose this question about a video corpus into retrieval sub-tasks.\n"
                        f"Question: {query}\n"
                        f"{('Corpus context: ' + context) if context else ''}\n"
                        "Set needs_vision only if answering requires looking at frames. "
                        "Set needs_computation only if arithmetic on extracted values is required. "
                        "Keep tasks minimal, 1 to 4."
                    ),
                )
            ],
            schema_hint=PLAN_SCHEMA,
            max_tokens=700,
        )
    except Exception as exc:
        log.info("planner falling back to heuristics: %s", exc)
        return None

    raw_tasks = data.get("tasks") or []
    tasks: list[SubTask] = []
    valid = {k.value for k in TaskKind}
    for i, t in enumerate(raw_tasks[:6]):
        if not isinstance(t, dict):
            continue
        kind = str(t.get("kind", "")).strip().lower()
        if kind not in valid:
            kind = TaskKind.TRANSCRIPT_LOOKUP.value
        tasks.append(
            SubTask(
                id=str(t.get("id") or f"t{i + 1}"),
                kind=TaskKind(kind),
                query=str(t.get("query") or query)[:500],
                depends_on=[str(d) for d in (t.get("depends_on") or []) if isinstance(d, (str, int))],
                rationale=str(t.get("rationale", ""))[:300],
                modality_bias={
                    str(k): float(v)
                    for k, v in (t.get("modality_bias") or {}).items()
                    if isinstance(v, (int, float)) and 0.0 < float(v) <= 4.0
                },
            )
        )
    if not tasks:
        return None
    style = str(data.get("answer_style", "timestamped")).lower()
    if style not in {"timestamped", "narrative", "table", "numeric"}:
        style = "timestamped"
    return QueryPlan(
        intent=str(data.get("intent", ""))[:400] or query,
        tasks=tasks,
        needs_computation=bool(data.get("needs_computation", False)),
        needs_vision=bool(data.get("needs_vision", False)),
        answer_style=style,  # type: ignore[arg-type]
    )


async def plan_query(query: str, *, context: str = "", allow_llm: bool = True) -> tuple[QueryPlan, str]:
    """Return ``(plan, source)``, the LLM plan when possible, heuristics always."""
    heuristic = heuristic_plan(query)
    if not allow_llm:
        return heuristic, "heuristic"
    plan = await llm_plan(query, context=context)
    if plan is None:
        return heuristic, "heuristic"
    # Union the two signals: the rule set is conservative and catches cues that
    # small models routinely miss (ordinals, "on screen"), while the LLM is
    # better at intent. Neither alone is as reliable as both.
    plan.needs_vision = plan.needs_vision or heuristic.needs_vision
    plan.needs_computation = plan.needs_computation or heuristic.needs_computation
    known = {t.kind for t in plan.tasks}
    for t in heuristic.tasks:
        if t.kind not in known and t.kind in {TaskKind.CHART_EXTRACTION, TaskKind.COMPUTATION, TaskKind.SPEAKER_ATTRIBUTION}:
            plan.tasks.append(SubTask(**{**t.model_dump(), "id": stable_id(t.kind, t.query)[:8]}))
    return plan, "llm"


def aggregate_bias(plan: QueryPlan) -> dict[str, float]:
    """Combine sub-task modality biases into one prior for the fusion step."""
    acc: dict[str, list[float]] = {}
    for t in plan.tasks:
        for k, v in t.modality_bias.items():
            acc.setdefault(k, []).append(float(v))
    return {k: round(sum(v) / len(v), 4) for k, v in acc.items()}
