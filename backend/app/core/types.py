"""Domain models shared by the pipeline, store, retrieval and API layers.

These are value objects. Pipeline stages transform them rather than mutating
shared state, which keeps the stage scheduler free of ordering hazards.
"""

from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

Seconds = Annotated[float, Field(ge=0.0)]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]


def utcnow() -> datetime:
    return datetime.now(UTC)


def stable_id(*parts: object) -> str:
    """Deterministic 32-hex id, makes ingestion idempotent and re-runnable."""
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


class Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, ser_json_timedelta="float")


# --------------------------------------------------------------------- status
class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageName(StrEnum):
    PROBE = "probe"
    AUDIO = "audio"
    SCENES = "scenes"
    KEYFRAMES = "keyframes"
    TRANSCRIBE = "transcribe"
    DIARIZE = "diarize"
    ALIGN = "align"
    EMBED = "embed"
    CHAPTERS = "chapters"
    INDEX = "index"
    SUMMARIZE = "summarize"


# ------------------------------------------------------------------ intervals
class TimeSpan(Base):
    """Half-open interval ``[start, end)`` in seconds."""

    start: Seconds
    end: Seconds

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end < self.start:
            object.__setattr__(self, "end", self.start)
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2.0

    def overlap(self, other: TimeSpan) -> float:
        return max(0.0, min(self.end, other.end) - max(self.start, other.start))

    def iou(self, other: TimeSpan) -> float:
        inter = self.overlap(other)
        union = self.duration + other.duration - inter
        return inter / union if union > 1e-9 else 0.0

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end

    def clamp(self, lo: float, hi: float) -> TimeSpan:
        return TimeSpan(start=min(max(self.start, lo), hi), end=min(max(self.end, lo), hi))

    def __repr__(self) -> str:  # pragma: no cover - debug nicety
        return f"[{self.start:.2f}->{self.end:.2f}]"


# ----------------------------------------------------------------- video meta
class VideoProbe(Base):
    duration: float
    width: int
    height: int
    fps: float
    codec: str = "unknown"
    has_audio: bool = True
    audio_channels: int = 0
    audio_sample_rate: int = 0
    bit_rate: int = 0
    container: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0


class Scene(Base):
    index: int
    span: TimeSpan
    #: Mean absolute HSV-delta against the previous scene: the "cut strength".
    cut_score: float = 0.0
    #: Fraction of the scene's frames that are near-static (slide detection).
    static_ratio: float = 0.0
    kind: Literal["cut", "fade", "static", "synthetic"] = "cut"


class Word(Base):
    text: str
    start: float
    end: float
    prob: float = 1.0


class TranscriptSegment(Base):
    index: int
    span: TimeSpan
    text: str
    speaker: str | None = None
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    words: list[Word] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence(self) -> float:
        """Map Whisper's log-prob to a 0-1 confidence with a soft floor."""
        p = math.exp(max(-6.0, min(0.0, self.avg_logprob)))
        return round(max(0.0, min(1.0, p * (1.0 - self.no_speech_prob))), 4)


class SpeakerTurn(Base):
    speaker: str
    span: TimeSpan
    confidence: float = 1.0


class Keyframe(Base):
    id: str
    scene_index: int
    timestamp: float
    path: str
    width: int
    height: int
    phash: int = 0
    #: Composite visual-quality score used by the keyframe budget allocator.
    quality: float = 0.0
    sharpness: float = 0.0
    entropy: float = 0.0
    text_density: float = 0.0
    is_slide: bool = False


class Sentence(Base):
    """One utterance inside a chunk, the unit of citation.

    Chunks are sized for embedding quality (~20 s); citations must be far more
    precise than that. Keeping sentence spans on the chunk lets an answer point
    at the exact moment a claim was made, and lets a speaker filter apply
    inside a chunk where two people alternate.
    """

    start: float
    end: float
    text: str
    speaker: str | None = None


class VideoChunk(Base):
    """A cohesive audio-visual unit: the atom of retrieval.

    A chunk is a parent document. Its child representations (a summary
    embedding and per-modality embeddings) live in the vector store, while the
    full transcript and frame paths stay here for context assembly.
    """

    id: str
    video_id: str
    index: int
    span: TimeSpan
    text: str = ""
    summary: str = ""
    speakers: list[str] = Field(default_factory=list)
    keyframe_ids: list[str] = Field(default_factory=list)
    scene_indices: list[int] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    sentences: list[Sentence] = Field(default_factory=list)
    #: Confidence-weighted density of speech in the chunk (words per second).
    speech_rate: float = 0.0
    visual_activity: float = 0.0
    token_estimate: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def label(self) -> str:
        return f"{fmt_ts(self.span.start)}-{fmt_ts(self.span.end)}"


def fmt_ts(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    frac = int((t - int(t)) * 10)
    return (f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}") + f".{frac}"


# --------------------------------------------------------------------- search
class ScoredHit(Base):
    chunk_id: str
    video_id: str
    score: float
    #: Per-modality rank contributions, kept for explainability in the UI.
    ranks: dict[str, int] = Field(default_factory=dict)
    raw_scores: dict[str, float] = Field(default_factory=dict)
    fusion: dict[str, float] = Field(default_factory=dict)
    chunk: VideoChunk | None = None
    keyframes: list[Keyframe] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def modalities(self) -> list[str]:
        return sorted(self.ranks)


class RetrievalRequest(Base):
    query: str
    video_ids: list[str] = Field(default_factory=list)
    speakers: list[str] = Field(default_factory=list)
    time_range: TimeSpan | None = None
    top_k: int = 8
    candidates: int = 48
    modalities: list[str] = Field(default_factory=lambda: ["text", "image", "summary"])
    mmr_lambda: float | None = None
    use_temporal_fusion: bool = True


class RetrievalTrace(Base):
    per_modality: dict[str, list[str]] = Field(default_factory=dict)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    fused_order: list[str] = Field(default_factory=list)
    mmr_order: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RetrievalResult(Base):
    hits: list[ScoredHit]
    trace: RetrievalTrace = Field(default_factory=RetrievalTrace)


# --------------------------------------------------------------------- agents
class TaskKind(StrEnum):
    VISUAL_LOOKUP = "visual_lookup"
    TRANSCRIPT_LOOKUP = "transcript_lookup"
    TEMPORAL_LOCATE = "temporal_locate"
    SPEAKER_ATTRIBUTION = "speaker_attribution"
    CHART_EXTRACTION = "chart_extraction"
    COMPUTATION = "computation"
    SUMMARIZE = "summarize"
    COMPARE = "compare"


class SubTask(Base):
    id: str
    kind: TaskKind
    query: str
    depends_on: list[str] = Field(default_factory=list)
    rationale: str = ""
    modality_bias: dict[str, float] = Field(default_factory=dict)
    filters: dict[str, Any] = Field(default_factory=dict)


class QueryPlan(Base):
    intent: str = ""
    tasks: list[SubTask] = Field(default_factory=list)
    needs_computation: bool = False
    needs_vision: bool = False
    answer_style: Literal["timestamped", "narrative", "table", "numeric"] = "timestamped"


class Citation(Base):
    chunk_id: str
    video_id: str
    start: float
    end: float
    speaker: str | None = None
    keyframe: str | None = None
    quote: str = ""
    relevance: float = 0.0


class AgentEvent(Base):
    seq: int = 0
    ts: datetime = Field(default_factory=utcnow)
    node: str
    kind: Literal["start", "log", "delta", "result", "error", "end"] = "log"
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class AnswerBundle(Base):
    query: str
    #: The question after references to earlier turns were resolved.
    resolved_query: str = ""
    is_followup: bool = False
    resolution_notes: list[str] = Field(default_factory=list)
    session_id: str = ""
    answer: str = ""
    plan: QueryPlan = Field(default_factory=QueryPlan)
    citations: list[Citation] = Field(default_factory=list)
    hits: list[ScoredHit] = Field(default_factory=list)
    computations: list[dict[str, Any]] = Field(default_factory=list)
    visual_findings: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    elapsed_ms: float = 0.0
    trace: list[AgentEvent] = Field(default_factory=list)
    model_used: str = ""
