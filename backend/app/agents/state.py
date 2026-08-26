"""Shared state for the reasoning graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.types import (
    AnswerBundle,
    Citation,
    QueryPlan,
    RetrievalTrace,
    ScoredHit,
    TimeSpan,
)


@dataclass
class AgentState:
    query: str
    video_ids: list[str] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    time_range: TimeSpan | None = None
    top_k: int = 8

    plan: QueryPlan = field(default_factory=QueryPlan)
    hits: list[ScoredHit] = field(default_factory=list)
    trace: RetrievalTrace = field(default_factory=RetrievalTrace)
    visual_findings: list[dict[str, Any]] = field(default_factory=list)
    computations: list[dict[str, Any]] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    answer: str = ""
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)
    model_used: str = ""
    llm_available: bool = False
    #: Retrieval rounds completed. The critic uses this to bound retries.
    round: int = 0
    #: Earlier turns in this conversation, oldest first.
    history: list[Any] = field(default_factory=list)
    #: The question after reference resolution; equals ``query`` when standalone.
    resolved_query: str = ""
    is_followup: bool = False
    resolution_notes: list[str] = field(default_factory=list)
    followups: list[str] = field(default_factory=list)
    coverage: float = 0.0

    @property
    def effective_query(self) -> str:
        """What retrieval should actually search for."""
        return self.resolved_query or self.query

    def context_chunks(self, limit: int = 6) -> list[ScoredHit]:
        return self.hits[:limit]

    def to_bundle(self, elapsed_ms: float = 0.0) -> AnswerBundle:
        return AnswerBundle(
            query=self.query,
            answer=self.answer,
            plan=self.plan,
            citations=self.citations,
            hits=self.hits,
            computations=self.computations,
            visual_findings=self.visual_findings,
            confidence=self.confidence,
            elapsed_ms=elapsed_ms,
            model_used=self.model_used,
        )
