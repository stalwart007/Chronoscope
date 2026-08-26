"""Visual question answering over selected keyframes.

Runs only when the plan needs pixels. Each candidate frame goes to a vision
model with a scoped question. Without one, frame metadata from ingestion and
numbers mined from the co-timed transcript still provide signal, since
presenters usually read their charts aloud.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from app.agents.numeric import build_series, extract_numbers
from app.agents.state import AgentState
from app.config import settings
from app.core.types import Keyframe, ScoredHit, TaskKind, fmt_ts
from app.llm.base import Message
from app.llm.providers import encode_image
from app.llm.router import router
from app.logging_conf import get_logger

log = get_logger(__name__)

MAX_FRAMES = 4

VQA_SCHEMA = """{"description":"what is visible","on_screen_text":"verbatim text you can read, or empty",
"data_points":[{"label":"Q1","value":42,"unit":"$M"}],"answers_question":true,"confidence":0.0}"""


def select_frames(hits: list[ScoredHit], *, chart_mode: bool, limit: int = MAX_FRAMES) -> list[tuple[Keyframe, ScoredHit]]:
    """Rank candidate frames by usefulness for this question."""
    scored: list[tuple[float, Keyframe, ScoredHit]] = []
    for rank, hit in enumerate(hits):
        decay = 1.0 / (1.0 + rank * 0.35)
        for kf in hit.keyframes:
            utility = decay * (0.45 + 0.55 * kf.quality)
            if chart_mode:
                utility *= 1.0 + 1.4 * kf.text_density + (0.5 if kf.is_slide else 0.0)
            scored.append((utility, kf, hit))
    scored.sort(key=lambda t: -t[0])
    out: list[tuple[Keyframe, ScoredHit]] = []
    seen: set[str] = set()
    for _u, kf, hit in scored:
        if kf.id in seen:
            continue
        seen.add(kf.id)
        out.append((kf, hit))
        if len(out) >= limit:
            break
    return out


def heuristic_finding(kf: Keyframe, hit: ScoredHit) -> dict[str, Any]:
    text = hit.chunk.text if hit.chunk else ""
    facts = extract_numbers(text, source="transcript")
    series = build_series(facts)
    kind = (
        "slide or diagram" if kf.is_slide
        else "text-heavy visual" if kf.text_density > 0.04
        else "detailed scene" if kf.entropy > 6.0
        else "presenter shot"
    )
    return {
        "frame_id": kf.id,
        "chunk_id": hit.chunk_id,
        "timestamp": kf.timestamp,
        "timestamp_label": fmt_ts(kf.timestamp),
        "image": kf.path,
        "description": f"{kind} (sharpness {kf.sharpness:.3f}, text density {kf.text_density:.3f})",
        "on_screen_text": "",
        "data_points": [
            {"label": f.label or "", "value": f.value, "unit": f.unit} for f in facts if f.label
        ]
        or [
            {"label": name, "value": value, "unit": series.unit}
            for name, value in zip(series.labels, series.values, strict=False)
        ],
        "answers_question": bool(series.ok or kf.is_slide),
        "confidence": round(0.35 + 0.25 * kf.quality + (0.2 if series.ok else 0.0), 3),
        "source": "metadata+transcript",
    }


async def _ask_vision(kf: Keyframe, hit: ScoredHit, question: str, chart_mode: bool) -> dict[str, Any] | None:
    path = Path(settings.artifact_dir) / "frames" / kf.path
    if not path.exists():
        return None
    try:
        data_url = encode_image(path)
        instruction = (
            "Read every number and axis label in the chart or table and return them as data_points."
            if chart_mode
            else "Describe what is on screen and any legible text."
        )
        data = await router.chat_json(
            [
                Message(
                    role="user",
                    content=(
                        f"Question about this video frame (at {fmt_ts(kf.timestamp)}): {question}\n"
                        f"{instruction}\nIf the frame does not help, set answers_question=false."
                    ),
                    images=[data_url],
                )
            ],
            schema_hint=VQA_SCHEMA,
            max_tokens=600,
            require_vision=True,
        )
    except Exception as exc:
        log.info("vision call failed for frame %s: %s", kf.id, exc)
        return None

    points = []
    for p in (data.get("data_points") or [])[:24]:
        if not isinstance(p, dict):
            continue
        raw_value = p.get("value")
        if raw_value is None:
            continue
        try:
            points.append(
                {
                    "label": str(p.get("label", ""))[:40],
                    "value": float(raw_value),
                    "unit": str(p.get("unit", ""))[:8],
                }
            )
        except (TypeError, ValueError):
            continue
    return {
        "frame_id": kf.id,
        "chunk_id": hit.chunk_id,
        "timestamp": kf.timestamp,
        "timestamp_label": fmt_ts(kf.timestamp),
        "image": kf.path,
        "description": str(data.get("description", ""))[:600],
        "on_screen_text": str(data.get("on_screen_text", ""))[:800],
        "data_points": points,
        "answers_question": bool(data.get("answers_question", True)),
        "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.6) or 0.6))),
        "source": str(data.get("_meta", {}).get("model", "vision")),
    }


async def visual_qa(state: AgentState) -> dict[str, Any]:
    if not state.hits:
        return {}
    chart_mode = any(t.kind == TaskKind.CHART_EXTRACTION for t in state.plan.tasks)
    frames = select_frames(state.hits, chart_mode=chart_mode)
    if not frames:
        return {"notes": ["no keyframes available for visual analysis"]}

    findings: list[dict[str, Any]]
    if state.llm_available:
        results = await asyncio.gather(
            *(_ask_vision(kf, hit, state.query, chart_mode) for kf, hit in frames), return_exceptions=True
        )
        findings = []
        for (kf, hit), res in zip(frames, results, strict=True):
            if isinstance(res, BaseException) or res is None:
                findings.append(heuristic_finding(kf, hit))
            else:
                # Merge in transcript-derived numbers the model may have missed.
                if chart_mode and not res["data_points"]:
                    res["data_points"] = heuristic_finding(kf, hit)["data_points"]
                findings.append(res)
    else:
        findings = [heuristic_finding(kf, hit) for kf, hit in frames]

    useful = [f for f in findings if f.get("answers_question") or f.get("data_points")]
    return {"visual_findings": useful or findings}
