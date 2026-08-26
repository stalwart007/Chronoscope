"""Export formats.

Derived artefacts are written in formats other tools already read: subtitles
for an editor, CSV for a spreadsheet, JSON for an archive. The writers are pure
functions over loaded rows so they can be tested without a database, and each
escapes its own format.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable, Sequence
from typing import Any

from app.core.types import Keyframe, Scene, VideoChunk


# --------------------------------------------------------------- subtitles
def _ts(seconds: float, *, comma: bool = True) -> str:
    ms = max(0, round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    sep = "," if comma else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _clean_cue(text: str) -> str:
    """Cue bodies cannot contain a blank line, it terminates the cue."""
    collapsed = " ".join(text.split())
    return collapsed or "..."


def to_srt(segments: Sequence[Any], *, with_speakers: bool = True) -> str:
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        body = _clean_cue(seg.text)
        if with_speakers and seg.speaker:
            body = f"[{seg.speaker}] {body}"
        lines += [str(i), f"{_ts(seg.start)} --> {_ts(seg.end)}", body, ""]
    return "\n".join(lines)


def to_vtt(segments: Sequence[Any], *, with_speakers: bool = True) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        body = _clean_cue(seg.text)
        cue = f"<v {seg.speaker}>{body}" if with_speakers and seg.speaker else body
        lines += [f"{_ts(seg.start, comma=False)} --> {_ts(seg.end, comma=False)}", cue, ""]
    return "\n".join(lines)


def to_plain_text(segments: Sequence[Any], *, with_speakers: bool = True, with_times: bool = True) -> str:
    """Readable transcript, grouped into paragraphs by speaker turn."""
    out: list[str] = []
    current: list[str] = []
    speaker: str | None = None
    start = 0.0
    for seg in segments:
        if seg.speaker != speaker and current:
            out.append(_paragraph(speaker, start, current, with_speakers, with_times))
            current = []
        if not current:
            start = seg.start
        speaker = seg.speaker
        current.append(_clean_cue(seg.text))
    if current:
        out.append(_paragraph(speaker, start, current, with_speakers, with_times))
    return "\n\n".join(out)


def _paragraph(
    speaker: str | None, start: float, parts: list[str], with_speakers: bool, with_times: bool
) -> str:
    prefix = ""
    if with_times:
        prefix += f"[{_ts(start, comma=False)[:-4]}] "
    if with_speakers and speaker:
        prefix += f"{speaker}: "
    return prefix + " ".join(parts)


# --------------------------------------------------------------------- csv
def _csv(rows: Iterable[Sequence[Any]], header: Sequence[str]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow([_csv_safe(cell) for cell in row])
    return buffer.getvalue()


def _csv_safe(value: Any) -> Any:
    """Neutralise spreadsheet formula injection.

    A cell beginning ``=``, ``+``, ``-`` or ``@`` is executed as a formula by
    Excel, Sheets and LibreOffice on open. Transcript text is user-controlled,
    so a line starting with "=" would become a live formula in the recipient's
    spreadsheet. Prefixing a single quote renders it as literal text.
    """
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def segments_to_csv(segments: Sequence[Any]) -> str:
    return _csv(
        ((s.index, round(s.start, 3), round(s.end, 3), s.speaker or "", round(s.confidence, 4), s.text)
         for s in segments),
        ["index", "start_s", "end_s", "speaker", "confidence", "text"],
    )


def chunks_to_csv(chunks: Sequence[VideoChunk]) -> str:
    return _csv(
        (
            (
                c.index, round(c.span.start, 3), round(c.span.end, 3), round(c.span.duration, 3),
                "|".join(c.speakers), "|".join(c.keywords), len(c.keyframe_ids),
                round(c.speech_rate, 3), round(c.visual_activity, 3), c.text,
            )
            for c in chunks
        ),
        ["index", "start_s", "end_s", "duration_s", "speakers", "keywords", "keyframes",
         "words_per_s", "visual_activity", "text"],
    )


def scenes_to_csv(scenes: Sequence[Scene]) -> str:
    return _csv(
        ((s.index, round(s.span.start, 3), round(s.span.end, 3), round(s.span.duration, 3),
          s.kind, round(s.cut_score, 3), round(s.static_ratio, 3)) for s in scenes),
        ["index", "start_s", "end_s", "duration_s", "kind", "cut_score", "static_ratio"],
    )


def keyframes_to_csv(keyframes: Sequence[Keyframe]) -> str:
    return _csv(
        ((k.id, k.scene_index, round(k.timestamp, 3), k.path, k.width, k.height,
          round(k.quality, 4), round(k.sharpness, 5), round(k.entropy, 3),
          round(k.text_density, 5), int(k.is_slide)) for k in keyframes),
        ["id", "scene", "timestamp_s", "path", "width", "height", "quality",
         "sharpness", "entropy", "text_density", "is_slide"],
    )


# -------------------------------------------------------------------- json
def bundle(video: Any, scenes: Sequence[Scene], chunks: Sequence[VideoChunk],
           keyframes: Sequence[Keyframe], segments: Sequence[Any]) -> str:
    """Complete, self-describing analysis archive."""
    payload = {
        "schema": "chronoscope/video-analysis@1",
        "video": {
            "id": video.id, "title": video.title, "filename": video.filename,
            "duration_s": video.duration, "width": video.width, "height": video.height,
            "fps": video.fps, "language": video.language, "summary": video.summary,
            "speakers": list(video.speakers or []), "topics": list(video.topics or []),
            "stats": dict(video.stats or {}),
        },
        "scenes": [s.model_dump() for s in scenes],
        "chunks": [c.model_dump() for c in chunks],
        "keyframes": [k.model_dump() for k in keyframes],
        "segments": [
            {"index": s.index, "start": s.start, "end": s.end, "speaker": s.speaker,
             "confidence": s.confidence, "text": s.text}
            for s in segments
        ],
    }
    return json.dumps(payload, indent=2, default=str)


CONTENT_TYPES = {
    "srt": "application/x-subrip",
    "vtt": "text/vtt",
    "txt": "text/plain; charset=utf-8",
    "csv": "text/csv; charset=utf-8",
    "json": "application/json",
    "md": "text/markdown; charset=utf-8",
}


def safe_filename(title: str, suffix: str) -> str:
    """ASCII, no separators, the value lands in a Content-Disposition header."""
    import re

    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", title or "chronoscope").strip("-.")[:80]
    return f"{stem or 'chronoscope'}.{suffix}"
