"""A small state graph with bulk-synchronous execution.

A superstep runs every active node concurrently. Nodes return partial updates,
which are merged into shared state through per-field reducers, so two nodes
writing the same list in one superstep both contribute. Edges, static or
conditional, determine the next superstep. A step budget bounds loops.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from app.logging_conf import get_logger

log = get_logger(__name__)

END = "__end__"
START = "__start__"

S = TypeVar("S")

NodeFn = Callable[[Any], Awaitable[dict[str, Any] | None]]
Router = Callable[[Any], str | list[str]]


def append_reducer(old: Any, new: Any) -> Any:
    if old is None:
        return new
    if isinstance(old, list) and isinstance(new, list):
        return [*old, *new]
    if isinstance(old, dict) and isinstance(new, dict):
        return {**old, **new}
    return new


def replace_reducer(_old: Any, new: Any) -> Any:
    return new


@dataclass(slots=True)
class GraphEvent:
    step: int
    node: str
    kind: str
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0


@dataclass(slots=True)
class _Node:
    name: str
    fn: NodeFn
    label: str = ""
    retries: int = 0


class StateGraph(Generic[S]):
    def __init__(self, *, reducers: dict[str, Callable[[Any, Any], Any]] | None = None) -> None:
        self._nodes: dict[str, _Node] = {}
        self._edges: dict[str, list[str]] = {}
        self._conditionals: dict[str, tuple[Router, dict[str, str]]] = {}
        self._entry: str | None = None
        self.reducers = reducers or {}

    # ------------------------------------------------------------- building
    def add_node(self, name: str, fn: NodeFn, *, label: str = "", retries: int = 0) -> StateGraph[S]:
        if name in (END, START):
            raise ValueError(f"{name} is reserved")
        self._nodes[name] = _Node(name, fn, label or name, retries)
        return self

    def add_edge(self, src: str, dst: str) -> StateGraph[S]:
        self._edges.setdefault(src, []).append(dst)
        return self

    def add_conditional_edges(self, src: str, router: Router, mapping: dict[str, str]) -> StateGraph[S]:
        self._conditionals[src] = (router, mapping)
        return self

    def set_entry(self, name: str) -> StateGraph[S]:
        self._entry = name
        return self

    def validate(self) -> None:
        if self._entry is None:
            raise ValueError("graph has no entry point")
        known = set(self._nodes) | {END}
        for src, dsts in self._edges.items():
            if src not in self._nodes:
                raise ValueError(f"edge from unknown node {src}")
            for d in dsts:
                if d not in known:
                    raise ValueError(f"edge {src} -> unknown node {d}")
        for src, (_r, mapping) in self._conditionals.items():
            if src not in self._nodes:
                raise ValueError(f"conditional edge from unknown node {src}")
            for d in mapping.values():
                if d not in known:
                    raise ValueError(f"conditional edge {src} -> unknown node {d}")

    def topology(self) -> dict[str, Any]:
        """Serialisable graph shape, the frontend draws this."""
        edges = [{"from": s, "to": d, "kind": "static"} for s, ds in self._edges.items() for d in ds]
        edges += [
            {"from": s, "to": d, "kind": "conditional", "when": k}
            for s, (_r, m) in self._conditionals.items()
            for k, d in m.items()
        ]
        return {
            "entry": self._entry,
            "nodes": [{"id": n.name, "label": n.label} for n in self._nodes.values()],
            "edges": edges,
        }

    # -------------------------------------------------------------- running
    def _successors(self, node: str, state: S) -> list[str]:
        out: list[str] = list(self._edges.get(node, []))
        cond = self._conditionals.get(node)
        if cond is not None:
            router, mapping = cond
            choice = router(state)
            keys = [choice] if isinstance(choice, str) else list(choice)
            out.extend(mapping.get(k, END) for k in keys)
        return out or [END]

    def _merge(self, state: S, update: dict[str, Any]) -> None:
        for key, value in update.items():
            if not hasattr(state, key):
                continue
            reducer = self.reducers.get(key, replace_reducer)
            setattr(state, key, reducer(getattr(state, key), value))

    async def astream(self, state: S, *, max_steps: int = 24):  # type: ignore[no-untyped-def]
        """Run the graph, yielding a ``GraphEvent`` for every transition."""
        self.validate()
        assert self._entry is not None
        active: list[str] = [self._entry]
        step = 0
        started = time.perf_counter()
        while active and step < max_steps:
            step += 1
            frontier = [n for n in dict.fromkeys(active) if n != END]
            if not frontier:
                break
            for name in frontier:
                yield GraphEvent(step, name, "start", self._nodes[name].label)

            async def run_one(name: str) -> tuple[str, dict[str, Any] | None, Exception | None, float]:
                node = self._nodes[name]
                t0 = time.perf_counter()
                attempt = 0
                while True:
                    try:
                        update = await node.fn(state)
                        return name, update, None, (time.perf_counter() - t0) * 1000
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        attempt += 1
                        if attempt > node.retries:
                            return name, None, exc, (time.perf_counter() - t0) * 1000
                        await asyncio.sleep(0.25 * attempt)

            results = await asyncio.gather(*(run_one(n) for n in frontier))
            nxt: list[str] = []
            for name, update, error, elapsed in results:
                if error is not None:
                    log.warning("node %s failed: %s", name, error)
                    yield GraphEvent(step, name, "error", str(error), elapsed_ms=round(elapsed, 1))
                else:
                    if update:
                        self._merge(state, update)
                    yield GraphEvent(
                        step, name, "end",
                        self._nodes[name].label,
                        data={k: _summarize(v) for k, v in (update or {}).items()},
                        elapsed_ms=round(elapsed, 1),
                    )
                nxt.extend(self._successors(name, state))
            active = [n for n in dict.fromkeys(nxt) if n != END]
        yield GraphEvent(step, END, "end", "complete", elapsed_ms=round((time.perf_counter() - started) * 1000, 1))

    async def invoke(self, state: S, *, max_steps: int = 24) -> tuple[S, list[GraphEvent]]:
        events: list[GraphEvent] = []
        async for ev in self.astream(state, max_steps=max_steps):
            events.append(ev)
        return state, events


def _summarize(value: Any, *, limit: int = 160) -> Any:
    """Compact a state update for the event stream (never ship megabytes)."""
    if isinstance(value, str):
        return value[:limit] + ("..." if len(value) > limit else "")
    if isinstance(value, (list, tuple)):
        return {"count": len(value), "sample": [_summarize(v, limit=60) for v in list(value)[:2]]}
    if isinstance(value, dict):
        return {k: _summarize(v, limit=60) for k, v in list(value.items())[:6]}
    if hasattr(value, "model_dump"):
        return _summarize(value.model_dump(), limit=limit)
    return value


