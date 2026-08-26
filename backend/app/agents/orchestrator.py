"""The reasoning graph.

plan runs, then retrieve, which fans out conditionally to visual_qa and analyst
before both feed synthesize. The vision branch runs only when the plan needs
pixels, the analyst only when it needs arithmetic. When both run they run
concurrently in one superstep and their partial updates merge through the
graph's reducers.

Every node transition is streamed so the client can display progress.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from app.agents.analyst import analyse
from app.agents.conversation import Turn
from app.agents.conversation import resolve as resolve_reference
from app.agents.critic import assess
from app.agents.graph import END, GraphEvent, StateGraph, append_reducer
from app.agents.planner import plan_query
from app.agents.retriever import retrieve
from app.agents.state import AgentState
from app.agents.synthesizer import synthesize
from app.agents.visual_qa import visual_qa
from app.config import settings
from app.core.types import AgentEvent, AnswerBundle, TimeSpan
from app.llm.router import router
from app.logging_conf import get_logger

log = get_logger(__name__)

#: ``citations`` replaces rather than appends. The retriever seeds chunk-level
#: citations and the synthesiser refines them to sentence precision.
REDUCERS: dict[str, Callable[[Any, Any], Any]] = {
    "visual_findings": append_reducer,
    "computations": append_reducer,
    "notes": append_reducer,
}


async def node_plan(state: AgentState) -> dict[str, Any]:
    """Resolve any reference to earlier turns, then plan the standalone question."""
    state.llm_available = await router.any_available()
    resolution = resolve_reference(state.query, state.history)
    plan, source = await plan_query(resolution.query, allow_llm=state.llm_available)
    notes = [f"plan source: {source}"]
    if resolution.is_followup:
        notes.append("follow-up: " + "; ".join(resolution.notes or ["carried context forward"]))
    return {
        "plan": plan,
        "resolved_query": resolution.query,
        "is_followup": resolution.is_followup,
        "resolution_notes": resolution.notes,
        "time_range": resolution.time_range or state.time_range,
        "notes": notes,
    }


async def node_retrieve(state: AgentState) -> dict[str, Any]:
    return await retrieve(state)


async def node_visual(state: AgentState) -> dict[str, Any]:
    return await visual_qa(state)


async def node_analyst(state: AgentState) -> dict[str, Any]:
    return await analyse(state)


async def node_review(state: AgentState) -> dict[str, Any]:
    """Judge whether the evidence answers the question before committing."""
    verdict = assess(state)
    note = f"round {state.round}: {verdict.reason} (coverage {verdict.coverage:.0%})"
    return {
        "followups": verdict.queries if verdict.should_retry else [],
        "coverage": verdict.coverage,
        "notes": [note],
    }


async def node_expand(state: AgentState) -> dict[str, Any]:
    """Retrieve again using the follow-up queries, merging with what we have."""
    merged = {h.chunk_id: h for h in state.hits}
    for query in state.followups:
        extra = await retrieve(AgentState(
            query=query,
            video_ids=state.video_ids,
            speakers=state.speakers,
            time_range=state.time_range,
            top_k=state.top_k,
            plan=state.plan,
            llm_available=state.llm_available,
        ))
        for hit in extra.get("hits", []):
            current = merged.get(hit.chunk_id)
            if current is None or hit.score > current.score:
                merged[hit.chunk_id] = hit
    hits = sorted(merged.values(), key=lambda h: -h.score)[: max(state.top_k, 8)]
    return {"hits": hits, "round": state.round + 1, "followups": []}


async def node_synthesize(state: AgentState) -> dict[str, Any]:
    return await synthesize(state)


async def node_fanout(_state: AgentState) -> dict[str, Any]:
    """Pass-through so the conditional fan-out has a single owner node."""
    return {}


def route_after_retrieve(state: AgentState) -> list[str]:
    """Conditional fan-out, only pay for the specialists the plan needs."""
    branches: list[str] = []
    if state.hits and state.plan.needs_vision:
        branches.append("vision")
    if state.hits and state.plan.needs_computation:
        branches.append("compute")
    return branches or ["direct"]


def route_after_review(state: AgentState) -> str:
    return "expand" if state.followups else "answer"


def build_graph() -> StateGraph[AgentState]:
    g: StateGraph[AgentState] = StateGraph(reducers=REDUCERS)
    g.add_node("plan", node_plan, label="Planner")
    g.add_node("retrieve", node_retrieve, label="Retriever", retries=1)
    g.add_node("review", node_review, label="Critic")
    g.add_node("expand", node_expand, label="Re-retrieve", retries=1)
    g.add_node("visual_qa", node_visual, label="Visual QA")
    g.add_node("analyst", node_analyst, label="Analyst")
    g.add_node("synthesize", node_synthesize, label="Synthesiser", retries=1)
    g.set_entry("plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "review")
    # The critic either sends the question back for another retrieval round or
    # releases it to the specialists. `expand` returns to the critic, so the
    # loop is bounded by the round budget rather than by graph shape.
    g.add_conditional_edges("review", route_after_review, {"expand": "expand", "answer": "specialists"})
    g.add_edge("expand", "review")
    g.add_node("specialists", node_fanout, label="Routing")
    g.add_conditional_edges(
        "specialists",
        route_after_retrieve,
        {"vision": "visual_qa", "compute": "analyst", "direct": "synthesize"},
    )
    g.add_edge("visual_qa", "synthesize")
    g.add_edge("analyst", "synthesize")
    g.add_edge("synthesize", END)
    return g


GRAPH = build_graph()


def topology() -> dict[str, Any]:
    return GRAPH.topology()


async def answer_stream(
    query: str,
    *,
    video_ids: list[str] | None = None,
    speakers: list[str] | None = None,
    time_range: TimeSpan | None = None,
    top_k: int = 8,
    history: list[Turn] | None = None,
) -> AsyncIterator[tuple[str, Any]]:
    """Yield ``(kind, payload)`` pairs: ``event`` ... then a final ``result``."""
    state = AgentState(
        query=query,
        video_ids=list(video_ids or []),
        speakers=list(speakers or []),
        time_range=time_range,
        top_k=top_k,
        history=list(history or []),
    )
    started = time.perf_counter()
    seq = 0
    trace: list[AgentEvent] = []
    async for ev in GRAPH.astream(state, max_steps=settings.agent_max_steps):
        seq += 1
        agent_ev = _to_agent_event(seq, ev)
        trace.append(agent_ev)
        yield "event", agent_ev
    bundle = state.to_bundle(elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
    bundle.trace = trace
    bundle.resolved_query = state.resolved_query or state.query
    bundle.is_followup = state.is_followup
    bundle.resolution_notes = state.resolution_notes
    yield "result", bundle


def _to_agent_event(seq: int, ev: GraphEvent) -> AgentEvent:
    kind = {"start": "start", "end": "result", "error": "error"}.get(ev.kind, "log")
    if ev.node == END:
        kind = "end"
    return AgentEvent(
        seq=seq,
        node=ev.node,
        kind=kind,  # type: ignore[arg-type]
        message=ev.message,
        data={**ev.data, "step": ev.step, "elapsed_ms": ev.elapsed_ms},
    )


async def answer(query: str, **kwargs: Any) -> AnswerBundle:
    bundle: AnswerBundle | None = None
    async for kind, payload in answer_stream(query, **kwargs):
        if kind == "result":
            bundle = payload
    assert bundle is not None
    return bundle
