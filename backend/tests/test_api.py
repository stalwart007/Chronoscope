"""End-to-end API tests: upload -> ingest -> retrieve -> answer."""

from __future__ import annotations

import asyncio
import io
import json
import shutil
import tempfile
import zipfile
from itertools import pairwise
from pathlib import Path

import pytest
import pytest_asyncio

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def _wait_for_completion(client, video_id: str, timeout: float = 120.0) -> dict:  # type: ignore[no-untyped-def]
    deadline = asyncio.get_running_loop().time() + timeout
    payload: dict = {}
    while asyncio.get_running_loop().time() < deadline:
        payload = (await client.get(f"/api/videos/{video_id}")).json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        await asyncio.sleep(0.2)
    raise AssertionError(f"ingestion did not finish: {payload}")


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def ingested(client, sample_video: Path):  # type: ignore[no-untyped-def]
    srt = sample_video.with_suffix(".srt")
    with sample_video.open("rb") as fh, srt.open("rb") as sub:
        response = await client.post(
            "/api/videos",
            files={
                "file": (sample_video.name, fh, "video/mp4"),
                "transcript": (srt.name, sub, "application/x-subrip"),
            },
            data={"title": "Q3 Engineering Review"},
        )
    assert response.status_code == 202, response.text
    video_id = response.json()["video_id"]
    payload = await _wait_for_completion(client, video_id)
    assert payload["status"] == "completed", payload.get("error")
    return payload


class TestSystem:
    async def test_health_reports_capabilities(self, client) -> None:  # type: ignore[no-untyped-def]
        body = (await client.get("/api/system/health")).json()
        assert body["status"] in {"ok", "degraded"}
        assert set(body["encoders"]) >= {"text", "image"}
        assert "vector_store" in body and "llm" in body
        # Running without models must be *reported*, never silent.
        if body["encoders"]["text"]["degraded"]:
            assert any("hashing" in d for d in body["degraded"])

    async def test_graph_topology_is_connected(self, client) -> None:  # type: ignore[no-untyped-def]
        topo = (await client.get("/api/system/graph")).json()
        node_ids = {n["id"] for n in topo["nodes"]}
        assert {"plan", "retrieve", "visual_qa", "analyst", "synthesize"} <= node_ids
        reachable = {topo["entry"]}
        for _ in range(len(node_ids)):
            for e in topo["edges"]:
                if e["from"] in reachable:
                    reachable.add(e["to"])
        assert node_ids <= reachable

    async def test_openapi_documents_every_router(self, client) -> None:  # type: ignore[no-untyped-def]
        paths = (await client.get("/api/openapi.json")).json()["paths"]
        assert {"/api/videos", "/api/query", "/api/search", "/api/system/health"} <= set(paths)


class TestLibrary:
    async def test_upload_is_content_addressed(self, client, ingested, sample_video) -> None:  # type: ignore[no-untyped-def]
        with sample_video.open("rb") as fh:
            again = await client.post("/api/videos", files={"file": (sample_video.name, fh, "video/mp4")})
        assert again.json()["duplicate"] is True
        assert again.json()["video_id"] == ingested["id"]

    async def test_rejects_unsupported_container(self, client) -> None:  # type: ignore[no-untyped-def]
        r = await client.post("/api/videos", files={"file": ("notes.txt", b"hello", "text/plain")})
        assert r.status_code == 415
        assert r.json()["error"]["code"] == "unsupported_media"

    async def test_ingestion_statistics(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        stats = ingested["stats"]
        assert stats["scenes"] == 5
        assert stats["chunks"] >= 5
        assert stats["keyframes"] >= 5
        assert stats["indexed"]["chunks"] > 0
        assert len(ingested["speakers"]) == 2

    async def test_timeline_is_consistent(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        tl = (await client.get(f"/api/videos/{ingested['id']}/timeline")).json()
        assert len(tl["scenes"]) == 5
        assert tl["chunks"] and tl["keyframes"] and tl["segments"]
        keyframe_ids = {k["id"] for k in tl["keyframes"]}
        for chunk in tl["chunks"]:
            assert set(chunk["keyframe_ids"]) <= keyframe_ids
        assert all(s["speaker"] for s in tl["segments"])

    async def test_media_and_frames_are_served(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        head = await client.get(f"/api/videos/{ingested['id']}/media", headers={"Range": "bytes=0-1023"})
        assert head.status_code in {200, 206}
        tl = (await client.get(f"/api/videos/{ingested['id']}/timeline")).json()
        frame = await client.get(f"/frames/{tl['keyframes'][0]['path']}")
        assert frame.status_code == 200 and frame.headers["content-type"].startswith("image/")

    async def test_missing_video_is_404(self, client) -> None:  # type: ignore[no-untyped-def]
        # Well-formed but absent -> 404; malformed -> 422 at the routing layer,
        # before any database or filesystem access happens.
        r = await client.get(f"/api/videos/{'0' * 32}")
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
        r = await client.get("/api/videos/does-not-exist")
        assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


class TestRetrieval:
    async def test_search_fuses_multiple_modalities(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        body = (
            await client.post("/api/search", json={"query": "architecture diagram", "top_k": 5})
        ).json()
        assert body["hits"]
        assert len(body["hits"][0]["ranks"]) >= 2, "expected agreement across channels"
        assert "total" in body["trace"]["timings_ms"]
        assert any("weights" in n for n in body["trace"]["notes"])

    async def test_lexical_channel_finds_rare_literals(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        body = (await client.post("/api/search", json={"query": "Kubernetes", "top_k": 3})).json()
        top = body["hits"][0]
        assert "lexical" in top["ranks"]
        assert "kubernetes" in top["chunk"]["text"].lower()

    async def test_open_ended_time_filter(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        """A one-sided window ("after 34s") must not require an upper bound."""
        async with client.stream(
            "GET", "/api/query/stream", params={"q": "revenue", "start": 34.0, "top_k": 3}
        ) as response:
            assert response.status_code == 200
            saw_answer = False
            async for line in response.aiter_lines():
                if line.startswith("event: answer"):
                    saw_answer = True
                if line.startswith("event: done"):
                    break
        assert saw_answer

    async def test_time_filter_restricts_results(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        body = (
            await client.post(
                "/api/search",
                json={"query": "revenue", "top_k": 5, "time_range": {"start": 34.0, "end": 58.0}},
            )
        ).json()
        assert body["hits"]
        assert all(h["chunk"]["span"]["end"] >= 34.0 for h in body["hits"])


class TestAgents:
    async def test_visual_question_locates_the_diagram(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        body = (
            await client.post(
                "/api/query", json={"query": "When does the speaker show the architecture diagram?"}
            )
        ).json()["answer"]
        assert body["plan"]["needs_vision"] is True
        assert body["visual_findings"], "vision branch should have produced findings"
        starts = [c["start"] for c in body["citations"]]
        assert any(8.0 <= s <= 22.0 for s in starts), f"expected a citation in the diagram scene, got {starts}"

    async def test_speaker_scoped_question(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        body = (
            await client.post(
                "/api/query", json={"query": "What did the second speaker say about Kubernetes?"}
            )
        ).json()["answer"]
        speakers = {c["speaker"] for c in body["citations"] if c["speaker"]}
        assert len(speakers) == 1, f"answer should stay within one speaker, got {speakers}"

    async def test_chart_question_computes_growth(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        body = (
            await client.post(
                "/api/query",
                json={"query": "Find the revenue chart and calculate the year over year growth"},
            )
        ).json()["answer"]
        assert body["plan"]["needs_computation"] is True
        assert body["computations"], "analyst branch should have run"
        comp = body["computations"][0]
        assert comp["series"]["values"] == [42e6, 51e6, 68e6, 91e6]
        value = comp["result"]["value"]
        assert value["overall_pct"] == pytest.approx(116.6667, abs=0.01)
        assert value["cagr_pct"] == pytest.approx(29.3989, abs=0.01)

    async def test_unanswerable_question_says_so(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        body = (
            await client.post("/api/query", json={"query": "zzzqqq nonexistent topic xyzzy"})
        ).json()["answer"]
        assert body["confidence"] < 0.75

    async def test_stream_emits_agent_events_then_answer(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        kinds: list[str] = []
        answer_payload = None
        async with client.stream(
            "GET", "/api/query/stream", params={"q": "what is the architecture", "top_k": 4}
        ) as response:
            assert response.status_code == 200
            event = None
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: ") and event:
                    kinds.append(event)
                    if event == "answer":
                        answer_payload = json.loads(line[6:])
                    if event == "done":
                        break
        assert kinds[0] == "open"
        assert kinds.count("agent") >= 3, kinds
        assert answer_payload and answer_payload["answer"]

    async def test_query_is_logged(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        await client.post("/api/query", json={"query": "kubernetes rollout"})
        history = (await client.get("/api/query/history")).json()["queries"]
        assert any(q["query"] == "kubernetes rollout" for q in history)


class TestLifecycle:
    async def test_delete_removes_everything(self, client, sample_video) -> None:  # type: ignore[no-untyped-def]
        from app.config import get_settings

        settings = get_settings()
        copy = settings.upload_dir / "second_copy.mp4"
        shutil.copy(sample_video, copy)
        with copy.open("rb") as fh:
            r = await client.post("/api/videos", files={"file": ("second_copy.mp4", fh, "video/mp4")})
        video_id = r.json()["video_id"]
        await _wait_for_completion(client, video_id)

        assert (await client.delete(f"/api/videos/{video_id}")).status_code == 200
        assert (await client.get(f"/api/videos/{video_id}")).status_code == 404
        hits = (await client.post("/api/search", json={"query": "architecture", "top_k": 10})).json()["hits"]
        assert all(h["video_id"] != video_id for h in hits)


@pytest.mark.asyncio(loop_scope="module")
class TestOnboarding:
    async def test_demo_endpoint_produces_a_searchable_video(self, client) -> None:  # type: ignore[no-untyped-def]
        """One click must go from empty install to answerable footage."""
        res = await client.post("/api/system/demo")
        assert res.status_code == 202, res.text
        video_id = res.json()["video_id"]

        payload = await _wait_for_completion(client, video_id)
        assert payload["status"] == "completed", payload.get("error")
        assert payload["stats"]["scenes"] == 5
        assert len(payload["speakers"]) == 2

        answer = (
            await client.post(
                "/api/query",
                json={"query": "what did they say about kubernetes", "video_ids": [video_id]},
            )
        ).json()["answer"]
        assert answer["citations"], "the demo must be immediately answerable"

    async def test_demo_is_idempotent(self, client) -> None:  # type: ignore[no-untyped-def]
        first = (await client.post("/api/system/demo")).json()
        second = (await client.post("/api/system/demo")).json()
        assert first["video_id"] == second["video_id"]
        assert second["already_loaded"] is True


@pytest.mark.asyncio(loop_scope="module")
class TestExports:
    """Every derived dataset must leave the system in a format other tools read."""

    @pytest.mark.parametrize(
        ("dataset", "fmt", "expect"),
        [
            ("transcript", "srt", "-->"),
            ("transcript", "vtt", "WEBVTT"),
            ("transcript", "txt", "SPEAKER_"),
            ("transcript", "csv", "index,start_s"),
            ("chunks", "csv", "words_per_s"),
            ("scenes", "csv", "cut_score"),
            ("keyframes", "csv", "text_density"),
            ("bundle", "json", "chronoscope/video-analysis@1"),
        ],
    )
    async def test_dataset_exports(self, client, ingested, dataset, fmt, expect) -> None:  # type: ignore[no-untyped-def]
        res = await client.get(f"/api/videos/{ingested['id']}/export/{dataset}?format={fmt}")
        assert res.status_code == 200, res.text
        assert expect in res.text
        assert "attachment" in res.headers["content-disposition"]
        assert res.headers["cache-control"] == "no-store"

    async def test_srt_roundtrips_through_the_parser(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        """An exported transcript must be re-ingestable, otherwise it is not a
        real subtitle file, just something that looks like one."""
        from app.ingest.transcribe import parse_subtitle

        body = (await client.get(f"/api/videos/{ingested['id']}/export/transcript?format=srt")).text
        tmp = Path(tempfile.mkdtemp()) / "out.srt"
        tmp.write_text(body, encoding="utf-8")
        cues = parse_subtitle(tmp)
        assert len(cues) >= 10
        assert all(c.span.end >= c.span.start for c in cues)
        assert any("kubernetes" in c.text.lower() for c in cues)

    async def test_csv_neutralises_formula_injection(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        from app.api.exports import _csv_safe

        assert _csv_safe("=HYPERLINK(1)").startswith("'")
        body = (await client.get(f"/api/videos/{ingested['id']}/export/chunks?format=csv")).text
        for line in body.splitlines()[1:]:
            assert not line.startswith(("=", "+", "@")), line[:40]

    async def test_frames_zip_contains_every_frame(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        res = await client.get(f"/api/videos/{ingested['id']}/export/frames.zip")
        assert res.status_code == 200
        archive = zipfile.ZipFile(io.BytesIO(res.content))
        names = archive.namelist()
        assert "keyframes.csv" in names
        assert sum(1 for n in names if n.endswith(".jpg")) == ingested["stats"]["keyframes"]
        assert archive.testzip() is None, "archive must not be corrupt"

    async def test_unknown_dataset_is_rejected(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        assert (await client.get(f"/api/videos/{ingested['id']}/export/secrets")).status_code == 422
        assert (
            await client.get(f"/api/videos/{ingested['id']}/export/chunks?format=srt")
        ).status_code in {400, 422}


@pytest.mark.asyncio(loop_scope="module")
class TestIterativeRetrieval:
    """The critic should retry when evidence is thin and stop when it is not."""

    async def test_well_covered_question_answers_in_one_round(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        body = (
            await client.post("/api/query", json={"query": "kubernetes rollout downtime"})
        ).json()["answer"]
        rounds = [e for e in body["trace"] if e["node"] == "expand"]
        assert not rounds, "a question the transcript answers directly should not need a second round"
        assert body["citations"]

    async def test_poorly_phrased_question_triggers_another_round(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        """Vocabulary the speaker never used should send the critic back."""
        body = (
            await client.post(
                "/api/query",
                json={"query": "orchestration substrate provisioning cadence remediation"},
            )
        ).json()["answer"]
        notes = " ".join(str(e.get("message", "")) for e in body["trace"])
        assert "expand" in notes.lower() or any(e["node"] == "expand" for e in body["trace"]), (
            "expected the critic to request a second retrieval round"
        )

    async def test_retry_budget_is_bounded(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        body = (
            await client.post("/api/query", json={"query": "zzzz qqqq xxxx vvvv wwww"})
        ).json()["answer"]
        expands = [e for e in body["trace"] if e["node"] == "expand" and e["kind"] == "start"]
        assert len(expands) <= 2, f"retry loop ran {len(expands)} times"
        assert body["answer"], "an unanswerable question must still produce a response"


@pytest.mark.asyncio(loop_scope="module")
class TestChapters:
    async def test_chapters_partition_the_timeline(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        timeline = (await client.get(f"/api/videos/{ingested['id']}/timeline")).json()
        chapters = timeline["chapters"]
        assert chapters, "expected topic segmentation to produce chapters"
        assert chapters[0]["start"] == pytest.approx(0.0, abs=0.01)
        for a, b in pairwise(chapters):
            assert a["end"] == pytest.approx(b["start"], abs=0.01), "chapters must be contiguous"
        assert all(c["title"] for c in chapters)
        assert chapters[-1]["end"] == pytest.approx(timeline["video"]["duration"], abs=1.0)


@pytest.mark.asyncio(loop_scope="module")
class TestConversations:
    """A thread should remember what was asked and forget what was not."""

    async def test_a_followup_is_answered_against_the_previous_turn(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        first = (
            await client.post(
                "/api/query",
                json={"query": "When does the speaker show the architecture diagram?"},
            )
        ).json()["answer"]
        assert first["is_followup"] is False
        session = first["session_id"]
        assert session

        second = (
            await client.post(
                "/api/query",
                json={"query": "what did they say right after that?", "session_id": session},
            )
        ).json()["answer"]
        assert second["is_followup"] is True
        assert second["resolved_query"] != second["query"], "the reference was never resolved"
        assert second["resolution_notes"], "a resolved question must explain itself"
        assert second["session_id"] == session

    async def test_a_new_topic_does_not_inherit_the_thread(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        first = (
            await client.post(
                "/api/query",
                json={"query": "When does the speaker show the architecture diagram?"},
            )
        ).json()["answer"]
        second = (
            await client.post(
                "/api/query",
                json={"query": "what about the revenue numbers", "session_id": first["session_id"]},
            )
        ).json()["answer"]
        assert second["is_followup"] is False, "a self-contained question was polluted with context"
        assert "diagram" not in second["resolved_query"]

    async def test_a_thread_is_listed_readable_and_removable(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        first = (await client.post("/api/query", json={"query": "kubernetes rollout"})).json()["answer"]
        session = first["session_id"]
        await client.post("/api/query", json={"query": "tell me more about it", "session_id": session})

        listing = (await client.get("/api/sessions")).json()
        assert any(s["id"] == session and s["turn_count"] == 2 for s in listing)

        detail = (await client.get(f"/api/sessions/{session}")).json()
        assert [t["index"] for t in detail["turns"]] == [0, 1]
        assert detail["turns"][0]["query"] == "kubernetes rollout"
        assert detail["turns"][1]["citations"], "a stored turn should keep its citations"

        assert (await client.delete(f"/api/sessions/{session}")).status_code == 200
        assert (await client.get(f"/api/sessions/{session}")).status_code == 404

    async def test_an_unknown_thread_is_not_invented(self, client, ingested) -> None:  # type: ignore[no-untyped-def]
        assert (await client.get("/api/sessions/deadbeef")).status_code == 404
