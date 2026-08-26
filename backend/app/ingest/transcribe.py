"""Speech to text with word-level timestamps.

Sources are tried in order: a sidecar caption file next to the media, then
faster-whisper, then energy-based voice activity detection. The VAD path emits
speech structure with empty text so the timeline, diarisation and visual
retrieval still work, and marks the result degraded.
"""

from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings
from app.core.types import TimeSpan, TranscriptSegment, Word
from app.logging_conf import get_logger

log = get_logger(__name__)

_TS = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})")

#: RMS below this is silence for practical purposes (about -60 dBFS).
SILENCE_RMS = 1e-3


@dataclass(slots=True)
class TranscriptResult:
    segments: list[TranscriptSegment]
    language: str = ""
    source: str = ""
    degraded: bool = False

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments).strip()


# ------------------------------------------------------------------- sidecars
def _parse_ts(value: str) -> float:
    m = _TS.search(value)
    if not m:
        try:
            return float(value)
        except ValueError:
            return 0.0
    h, mm, ss, ms = m.groups()
    return int(h) * 3600 + int(mm) * 60 + int(ss) + int(ms.ljust(3, "0")) / 1000.0


def _interp_words(text: str, start: float, end: float) -> list[Word]:
    """Distribute a cue's duration across its tokens by character length."""
    tokens = text.split()
    if not tokens:
        return []
    weights = np.asarray([len(t) + 1 for t in tokens], dtype=np.float64)
    frac = np.concatenate([[0.0], np.cumsum(weights / weights.sum())])
    span = max(end - start, 1e-3)
    return [
        Word(text=tok, start=round(start + frac[i] * span, 3), end=round(start + frac[i + 1] * span, 3), prob=0.9)
        for i, tok in enumerate(tokens)
    ]


def parse_subtitle(path: Path) -> list[TranscriptSegment]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".json":
        return _parse_json_transcript(json.loads(raw))
    blocks = re.split(r"\n\s*\n", raw.replace("\r\n", "\n").strip())
    segments: list[TranscriptSegment] = []
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip() and not ln.strip().upper().startswith("WEBVTT")]
        if not lines:
            continue
        time_line = next((ln for ln in lines if "-->" in ln), None)
        if time_line is None:
            continue
        left, right = time_line.split("-->")[:2]
        start, end = _parse_ts(left.strip()), _parse_ts(right.strip().split(" ")[0])
        body_lines = [ln for ln in lines if "-->" not in ln and not ln.strip().isdigit()]
        speaker = None
        body = " ".join(body_lines).strip()
        if body.startswith("[") and "]" in body:  # "[Speaker 1] hello"
            speaker, body = body[1 : body.index("]")].strip(), body[body.index("]") + 1 :].strip()
        elif re.match(r"^<v ([^>]+)>", body):
            m = re.match(r"^<v ([^>]+)>(.*)", body)
            assert m
            speaker, body = m.group(1).strip(), m.group(2).strip()
        body = re.sub(r"<[^>]+>", "", body).strip()
        if not body:
            continue
        segments.append(
            TranscriptSegment(
                index=len(segments),
                span=TimeSpan(start=start, end=max(end, start + 0.2)),
                text=body,
                speaker=speaker,
                avg_logprob=-0.2,
                words=_interp_words(body, start, max(end, start + 0.2)),
            )
        )
    return segments


def _parse_json_transcript(data: object) -> list[TranscriptSegment]:
    items = data.get("segments", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    out: list[TranscriptSegment] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        start = float(it.get("start") or it.get("from") or 0.0)
        end = float(it.get("end") or it.get("to") or start + 1.0)
        text = str(it.get("text", "")).strip()
        if not text:
            continue
        words = [
            Word(text=str(w.get("word", w.get("text", ""))).strip(), start=float(w.get("start", start)),
                 end=float(w.get("end", end)), prob=float(w.get("probability", w.get("prob", 0.9))))
            for w in it.get("words", []) or []
            if str(w.get("word", w.get("text", ""))).strip()
        ]
        out.append(
            TranscriptSegment(
                index=i,
                span=TimeSpan(start=start, end=end),
                text=text,
                speaker=it.get("speaker"),
                avg_logprob=float(it.get("avg_logprob", -0.2)),
                no_speech_prob=float(it.get("no_speech_prob", 0.0)),
                words=words or _interp_words(text, start, end),
            )
        )
    return out


def find_sidecar(video_path: Path) -> Path | None:
    for suffix in (".srt", ".vtt", ".json"):
        for candidate in (video_path.with_suffix(suffix), video_path.parent / f"{video_path.stem}.transcript{suffix}"):
            if candidate.exists():
                return candidate
    return None


# ------------------------------------------------------------------- whisper
def transcribe_whisper(audio_path: Path) -> TranscriptResult | None:
    if importlib.util.find_spec("faster_whisper") is None:
        return None
    try:
        model = _whisper_model()
        segments, info = model.transcribe(
            str(audio_path),
            beam_size=settings.whisper_beam_size,
            word_timestamps=True,
            vad_filter=settings.whisper_vad,
            vad_parameters={"min_silence_duration_ms": 500},
            language=settings.language,
            condition_on_previous_text=False,
        )
        out: list[TranscriptSegment] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            words = [
                Word(text=w.word.strip(), start=round(w.start, 3), end=round(w.end, 3), prob=round(w.probability, 4))
                for w in (seg.words or [])
                if w.word and w.word.strip()
            ]
            out.append(
                TranscriptSegment(
                    index=len(out),
                    span=TimeSpan(start=round(seg.start, 3), end=round(seg.end, 3)),
                    text=text,
                    avg_logprob=float(getattr(seg, "avg_logprob", -0.3)),
                    no_speech_prob=float(getattr(seg, "no_speech_prob", 0.0)),
                    words=words or _interp_words(text, seg.start, seg.end),
                )
            )
        return TranscriptResult(segments=out, language=getattr(info, "language", "") or "", source="faster-whisper")
    except Exception as exc:
        log.warning("whisper transcription failed: %s", exc)
        return None


_MODEL_CACHE: dict[str, Any] = {}


def _whisper_model() -> Any:
    from faster_whisper import WhisperModel

    key = f"{settings.whisper_model}:{settings.whisper_compute_type}"
    if key not in _MODEL_CACHE:
        from app.embed.base import pick_device

        device = pick_device(settings.device)
        # CTranslate2 has no MPS backend; CPU + int8 is the fast path there.
        ct_device = "cuda" if device == "cuda" else "cpu"
        compute = settings.whisper_compute_type if ct_device == "cpu" else "float16"
        log.info("loading whisper %s on %s (%s)", settings.whisper_model, ct_device, compute)
        _MODEL_CACHE[key] = WhisperModel(
            settings.whisper_model,
            device=ct_device,
            compute_type=compute,
            download_root=str(settings.model_cache_dir) if settings.model_cache_dir else None,
        )
    return _MODEL_CACHE[key]


# ----------------------------------------------------------------- energy VAD
def energy_vad(
    audio: np.ndarray, sample_rate: int = 16000, *, frame_ms: float = 30.0, min_speech: float = 0.35
) -> list[TimeSpan]:
    """Adaptive-threshold voice activity detection.

    The threshold sits between the noise floor (10th percentile of frame
    energy) and the speech peak (90th percentile) in log space, which adapts
    across recordings far better than a fixed dB cut.

    A silent or near-silent track has no percentile spread at all, so an
    absolute level check runs first. Without it a muted track reads as uniform
    and every frame is reported as speech.
    """
    if audio.size == 0:
        return []
    hop = max(1, int(sample_rate * frame_ms / 1000.0))
    n = audio.size // hop
    if n < 2:
        return []
    frames = audio[: n * hop].reshape(n, hop)
    energy = np.log10(np.maximum((frames**2).mean(axis=1), 1e-10))

    # Absolute gate: RMS below about -60 dBFS carries no recoverable speech.
    if float(np.sqrt((audio**2).mean())) < SILENCE_RMS:
        return []

    floor, peak = float(np.percentile(energy, 10)), float(np.percentile(energy, 90))
    if peak - floor < 0.35:  # uniform level, and loud enough to be content
        return [TimeSpan(start=0.0, end=float(audio.size / sample_rate))]
    threshold = floor + 0.45 * (peak - floor)
    active = energy > threshold
    # Morphological closing: bridge gaps shorter than ~150 ms.
    bridge = max(1, int(0.15 / (frame_ms / 1000.0)))
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return []
    spans: list[TimeSpan] = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i - prev > bridge:
            spans.append(TimeSpan(start=start * hop / sample_rate, end=(prev + 1) * hop / sample_rate))
            start = i
        prev = i
    spans.append(TimeSpan(start=start * hop / sample_rate, end=(prev + 1) * hop / sample_rate))
    return [s for s in spans if s.duration >= min_speech]


def transcribe_fallback(audio: np.ndarray, sample_rate: int = 16000) -> TranscriptResult:
    spans = energy_vad(audio, sample_rate)
    segments = [
        TranscriptSegment(index=i, span=s, text="", avg_logprob=-2.5, no_speech_prob=0.0) for i, s in enumerate(spans)
    ]
    return TranscriptResult(segments=segments, source="energy-vad", degraded=True)


# ------------------------------------------------------------------ entry pt
def transcribe(video_path: Path, audio_path: Path, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptResult:
    sidecar = find_sidecar(video_path)
    if sidecar is not None:
        try:
            segments = parse_subtitle(sidecar)
            if segments:
                log.info("using sidecar transcript %s (%d cues)", sidecar.name, len(segments))
                return TranscriptResult(segments=segments, source=f"sidecar:{sidecar.suffix.lstrip('.')}")
        except Exception as exc:
            log.warning("sidecar %s unusable: %s", sidecar, exc)

    result = transcribe_whisper(audio_path)
    if result is not None and result.segments:
        log.info("whisper produced %d segments (%s)", len(result.segments), result.language)
        return result

    log.warning("no ASR backend available, falling back to energy VAD")
    return transcribe_fallback(audio, sample_rate)


def transcript_stats(result: TranscriptResult) -> dict[str, float]:
    words = sum(len(s.words) or len(s.text.split()) for s in result.segments)
    speech = sum(s.span.duration for s in result.segments)
    return {
        "segments": len(result.segments),
        "words": words,
        "speech_seconds": round(speech, 2),
        "wpm": round(words / (speech / 60.0), 1) if speech > 1 else 0.0,
        "mean_confidence": round(
            float(np.mean([s.confidence for s in result.segments])) if result.segments else 0.0, 4
        ),
    }
