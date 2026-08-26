"""Ingestion-stage tests against the synthetic talk (deterministic ground truth)."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from app.ingest.align import AlignmentInputs, build_chunks, candidate_boundaries, extract_keywords
from app.ingest.decode import extract_audio, probe_video
from app.ingest.diarize import assign_speakers, diarize_spectral, mfcc
from app.ingest.keyframes import allocate_budget, extract_keyframes
from app.ingest.pipeline import STAGES, validate_dag
from app.ingest.scenes import detect_scenes
from app.ingest.transcribe import energy_vad, parse_subtitle, transcribe

# The generator lays scenes out at exactly these boundaries.
TRUE_CUTS = [8.0, 22.0, 34.0, 48.0]
TRUE_TURNS = [(0.5, 7.5, 0), (8.5, 21.0, 0), (22.5, 27.0, 1), (27.5, 33.5, 0), (34.5, 47.0, 1), (48.5, 57.0, 0)]


@pytest.fixture(scope="module")
def probe(sample_video: Path):  # type: ignore[no-untyped-def]
    return probe_video(sample_video)


@pytest.fixture(scope="module")
def scenes_and_signal(sample_video: Path, probe):  # type: ignore[no-untyped-def]
    return detect_scenes(str(sample_video), duration=probe.duration)


@pytest.fixture(scope="module")
def audio(sample_video: Path, tmp_path_factory):  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("audio") / "a.wav"
    return extract_audio(sample_video, out), out


def test_probe_reads_container(probe) -> None:  # type: ignore[no-untyped-def]
    assert probe.width == 960 and probe.height == 540
    assert probe.has_audio and probe.audio_sample_rate == 16000
    assert 57.0 < probe.duration < 59.5


def test_dag_is_acyclic_and_ordered() -> None:
    order = validate_dag(STAGES)
    assert order.index("probe") == 0
    assert order.index("align") > order.index("keyframes")
    assert order.index("index") > order.index("embed")


def test_scene_detection_finds_every_cut(scenes_and_signal, probe) -> None:  # type: ignore[no-untyped-def]
    scenes, _signal = scenes_and_signal
    detected = [s.span.start for s in scenes[1:]]
    assert len(scenes) == len(TRUE_CUTS) + 1
    for truth in TRUE_CUTS:
        assert min(abs(d - truth) for d in detected) < 1.0
    assert scenes[-1].span.end == pytest.approx(probe.duration, abs=0.5)


def test_keyframe_budget_covers_every_scene(sample_video, scenes_and_signal, tmp_path) -> None:  # type: ignore[no-untyped-def]
    scenes, signal = scenes_and_signal
    alloc = allocate_budget(scenes, signal, 40)
    assert sum(alloc.values()) <= 40 and all(v >= 1 for v in alloc.values())
    frames = extract_keyframes(str(sample_video), "vtest", scenes, signal, tmp_path / "frames", budget=40)
    assert frames
    assert {f.scene_index for f in frames} == {s.index for s in scenes}
    assert all((tmp_path / "frames" / f.path).exists() for f in frames)
    assert any(f.is_slide for f in frames)


def test_vad_recovers_speech_turns(audio) -> None:  # type: ignore[no-untyped-def]
    pcm, _path = audio
    spans = energy_vad(pcm)
    assert len(spans) == len(TRUE_TURNS)
    for (t_start, t_end, _spk), span in zip(TRUE_TURNS, spans, strict=True):
        assert abs(span.start - t_start) < 0.4
        assert abs(span.end - t_end) < 0.6


def test_spectral_diarization_separates_two_voices(audio) -> None:  # type: ignore[no-untyped-def]
    pcm, _path = audio
    spans = energy_vad(pcm)
    result = diarize_spectral(pcm, spans)
    assert len(result.speakers) == 2
    predicted = [t.speaker for t in result.turns]
    truth = [f"SPEAKER_{s:02d}" for _a, _b, s in TRUE_TURNS]
    # Cluster ids are arbitrary; compare the partition, not the labels.
    assert _same_partition(predicted, truth)


def _same_partition(a: list[str], b: list[str]) -> bool:
    mapping: dict[str, str] = {}
    for x, y in zip(a, b, strict=True):
        if mapping.setdefault(x, y) != y:
            return False
    return len(set(mapping.values())) == len(set(mapping))


def test_mfcc_shape(audio) -> None:  # type: ignore[no-untyped-def]
    pcm, _ = audio
    feats = mfcc(pcm)
    assert feats.ndim == 2 and feats.shape[1] == 40
    assert np.isfinite(feats).all()


def test_sidecar_transcript(sample_video, audio) -> None:  # type: ignore[no-untyped-def]
    pcm, wav = audio
    srt = sample_video.with_suffix(".srt")
    assert srt.exists(), "generator should emit a sidecar"
    result = transcribe(sample_video, wav, pcm)
    assert result.source.startswith("sidecar")
    assert len(result.segments) == len(parse_subtitle(srt))
    assert all(s.words for s in result.segments)
    assert any("kubernetes" in s.text.lower() for s in result.segments)


def test_chunking_respects_speaker_and_scene_boundaries(
    sample_video, probe, scenes_and_signal, audio, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    scenes, signal = scenes_and_signal
    pcm, wav = audio
    transcript = transcribe(sample_video, wav, pcm)
    diar = diarize_spectral(pcm, energy_vad(pcm))
    assign_speakers(transcript.segments, diar.turns)
    frames = extract_keyframes(str(sample_video), "v", scenes, signal, tmp_path / "frames", budget=30)
    inputs = AlignmentInputs("v", probe.duration, transcript.segments, diar.turns, scenes, frames, signal)

    candidates = candidate_boundaries(inputs)
    assert {kind for _t, _q, kind in candidates} >= {"scene", "sentence"}

    chunks = build_chunks(inputs)
    assert len(chunks) >= 5
    assert chunks[0].span.start == 0.0
    assert chunks[-1].span.end == pytest.approx(probe.duration, abs=0.6)
    # Contiguous, non-overlapping cover of the timeline.
    for a, b in itertools.pairwise(chunks):
        assert a.span.end == pytest.approx(b.span.start, abs=1e-3)
    # Most chunks should be single-speaker once boundaries snap to turns.
    single = sum(1 for c in chunks if len(c.speakers) <= 1)
    assert single / len(chunks) > 0.7
    assert all(c.sentences for c in chunks if c.text)
    assert any("kubernetes" in kw for c in chunks for kw in c.keywords)


def test_keyword_extraction_discards_filler() -> None:
    docs = [
        "The Kubernetes rollout finished two weeks early with zero downtime across all regions.",
        "Here is the system architecture diagram for the multimodal retrieval pipeline.",
    ]
    kws = extract_keywords(docs)
    assert "kubernetes" in kws[0]
    assert "architecture" in kws[1]
    assert not any(k in {"the", "is", "with", "for"} for doc in kws for k in doc)


def test_video_topics_prefer_subject_matter_over_filler() -> None:
    """Topics must surface what the talk is *about*, not how it opens."""
    from app.agents.summarizer import video_topics
    from app.core.types import TimeSpan, VideoChunk

    lines = [
        "Welcome everyone to the engineering review for the third quarter.",
        "Here is the system architecture diagram for the retrieval pipeline.",
        "The Kubernetes rollout finished two weeks early with zero downtime.",
        "We now autoscale the embedding workers based on Kubernetes queue depth.",
        "Quarterly revenue was forty two million in the first quarter.",
        "Revenue reached ninety one million by the fourth quarter.",
    ]
    chunks = [
        VideoChunk(id=str(i), video_id="v", index=i, span=TimeSpan(start=i * 10, end=i * 10 + 10), text=t)
        for i, t in enumerate(lines)
    ]
    topics = video_topics(chunks)
    assert {"kubernetes", "revenue", "quarter"} <= set(topics), topics
    assert "welcome" not in topics and "everyone" not in topics


class TestDegenerateInput:
    """Media that is empty, silent or structureless must not crash a stage."""

    def test_silence_yields_no_speech(self) -> None:
        from app.ingest.transcribe import energy_vad, transcribe_fallback

        silence = np.zeros(16000 * 5, dtype=np.float32)
        assert energy_vad(silence) == []
        assert transcribe_fallback(silence).segments == []

        rng = np.random.default_rng(0)
        near_silent = rng.normal(0, 3e-4, 16000 * 5).astype(np.float32)
        assert energy_vad(near_silent) == [], "-70 dBFS is not speech"

    def test_quiet_but_present_audio_is_kept(self) -> None:
        from app.ingest.transcribe import energy_vad

        rng = np.random.default_rng(1)
        room_tone = rng.normal(0, 5e-3, 16000 * 3).astype(np.float32)
        assert len(energy_vad(room_tone)) >= 1, "quiet content must survive the gate"

    def test_empty_inputs_do_not_crash(self) -> None:
        from app.core.ranking import mmr, reciprocal_rank_fusion, temporal_diffusion
        from app.ingest.diarize import diarize_spectral
        from app.ingest.keyframes import allocate_budget
        from app.ingest.scenes import ContentSignal, detect_scenes_builtin, select_cuts

        empty = ContentSignal(np.array([], np.float32), np.array([], np.float32), np.array([], np.float32))
        assert len(detect_scenes_builtin(empty, duration=0.0)) == 1
        assert select_cuts(np.array([]), np.array([]), 1.0) == []
        assert allocate_budget([], empty, 10) == {}
        assert diarize_spectral(np.zeros(1000, dtype=np.float32), []).speakers == []
        assert reciprocal_rank_fusion({}).order == []
        assert mmr([], {}, {}) == []
        assert temporal_diffusion({}, {}) == {}

    def test_video_without_speech_still_produces_a_chunk(self) -> None:
        """Silent footage must remain visually searchable."""
        from app.core.types import Scene, TimeSpan
        from app.ingest.align import AlignmentInputs, build_chunks

        scenes = [Scene(index=0, span=TimeSpan(start=0, end=30))]
        chunks = build_chunks(AlignmentInputs("v", 30.0, [], [], scenes, [], None))
        assert len(chunks) == 1
        assert chunks[0].span.end == pytest.approx(30.0, abs=0.5)


class TestAdaptiveSampling:
    """Coverage must not degrade with recording length.

    A fixed sample budget spread uniformly means one frame every few seconds on
    a long recording, so anything brief falls between samples. The sampler
    scans every frame cheaply and spends the expensive descriptor only where
    the picture actually changes.
    """

    @pytest.mark.parametrize(
        ("stride", "equivalent"),
        [(0.38, "10 min"), (1.12, "30 min"), (2.25, "60 min"), (4.5, "2 hours"), (6.75, "3 hours")],
    )
    def test_brief_events_survive_every_resolution(self, brief_events_video, stride, equivalent) -> None:
        from app.ingest.scenes import content_signal, detect_scenes_builtin

        path, edges = brief_events_video
        signal = content_signal(str(path), duration=240.0, stride=stride)
        scenes = detect_scenes_builtin(signal, duration=240.0)
        found = [s.span.start for s in scenes[1:]]
        hits = sum(1 for e in edges if any(abs(e - f) < max(2.0, stride) for f in found))
        assert hits == len(edges), (
            f"at {equivalent} resolution only {hits}/{len(edges)} event edges were detected"
        )

    def test_dense_scan_does_not_explode_the_sample_count(self, brief_events_video) -> None:
        """Anomaly frames are added on top of the grid, not instead of it, so a
        pathological input must not blow past the budget."""
        from app.ingest.scenes import content_signal

        path, _ = brief_events_video
        coarse = content_signal(str(path), duration=240.0, stride=4.5)
        assert len(coarse) < 200, f"expected a bounded sample count, got {len(coarse)}"
