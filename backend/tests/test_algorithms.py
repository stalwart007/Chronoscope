"""Unit tests for the algorithmic core: index, trees, fusion, sandbox."""

from __future__ import annotations

import numpy as np
import pytest

from app.agents.conversation import Turn, resolve, subject_words
from app.agents.numeric import build_series, extract_numbers, words_to_number
from app.agents.tools import repl
from app.core.bktree import BKTree, hamming
from app.core.errors import SandboxViolation
from app.core.hnsw import HNSW, l2_normalize
from app.core.interval_tree import Coverage, IntervalTree
from app.core.ranking import (
    adaptive_modality_weights,
    mean_reciprocal_rank,
    mmr,
    ndcg_at_k,
    normalized_entropy,
    recall_at_k,
    reciprocal_rank_fusion,
    temporal_diffusion,
)
from app.core.types import Citation
from app.ingest.scenes import select_cuts


class TestIntervalTree:
    def test_matches_bruteforce(self) -> None:
        rng = np.random.default_rng(3)
        data = []
        for i in range(800):
            s = float(rng.uniform(0, 500))
            data.append((s, s + float(rng.uniform(0.1, 4)), i))
        tree = IntervalTree(data)
        for _ in range(40):
            lo = float(rng.uniform(0, 500))
            hi = lo + float(rng.uniform(0.5, 15))
            got = {iv.payload for iv in tree.query(lo, hi)}
            expected = {p for s, e, p in data if s < hi and e > lo}
            assert got == expected

    def test_balanced_after_sequential_inserts(self) -> None:
        tree: IntervalTree[int] = IntervalTree()
        for i in range(1000):  # worst case for an unbalanced BST
            tree.insert(float(i), float(i + 1), i)
        assert len(tree) == 1000
        assert tree.height <= 2 * 10 + 2  # AVL bound ~ 1.44 log2(n)

    def test_best_overlap(self) -> None:
        tree = IntervalTree([(0.0, 5.0, "a"), (4.0, 12.0, "b")])
        best = tree.best_overlap(4.5, 11.0)
        assert best is not None and best.payload == "b"

    def test_coverage_union(self) -> None:
        c = Coverage()
        c.add(0, 5)
        c.add(4, 8)
        c.add(20, 22)
        assert c.merged() == [(0, 8), (20, 22)]
        assert c.total() == 10


class TestHNSW:
    def test_recall_against_exact_search(self) -> None:
        rng = np.random.default_rng(0)
        x = l2_normalize(rng.normal(size=(1200, 48)).astype(np.float32))
        idx = HNSW(48, m=16, ef_construction=140, ef_search=90)
        for i in range(x.shape[0]):
            idx.add(f"k{i}", x[i], {"i": i})
        recall = 0.0
        queries = l2_normalize(rng.normal(size=(25, 48)).astype(np.float32))
        for q in queries:
            got = {k for k, _, _ in idx.search(q, 10)}
            exact = {k for k, _ in idx.brute_force(q, 10)}
            recall += len(got & exact) / 10
        assert recall / len(queries) > 0.9

    def test_filtered_search_and_delete(self) -> None:
        rng = np.random.default_rng(1)
        idx = HNSW(16, m=8)
        for i in range(200):
            idx.add(f"k{i}", rng.normal(size=16), {"grp": i % 4})
        q = rng.normal(size=16)
        hits = idx.search(q, 5, predicate=lambda _k, p: p["grp"] == 2)
        assert hits and all(p["grp"] == 2 for _, _, p in hits)
        removed_vector = idx.get("k0")
        assert removed_vector is not None
        assert idx.remove("k0")
        assert "k0" not in {k for k, _, _ in idx.search(removed_vector, 20)}

    def test_upsert_replaces_vector(self) -> None:
        idx = HNSW(8, m=6)
        a = np.ones(8, dtype=np.float32)
        b = np.arange(8, dtype=np.float32)
        idx.add("x", a)
        idx.add("x", b)
        assert len(idx) == 1
        assert np.allclose(idx.get("x"), l2_normalize(b), atol=1e-5)

    def test_roundtrip_persistence(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        rng = np.random.default_rng(5)
        idx = HNSW(12, m=8)
        for i in range(80):
            idx.add(f"k{i}", rng.normal(size=12))
        q = rng.normal(size=12)
        before = [k for k, _, _ in idx.search(q, 5)]
        idx.save(tmp_path / "idx")
        after = [k for k, _, _ in HNSW.load(tmp_path / "idx").search(q, 5)]
        assert before == after


class TestBKTree:
    def test_radius_search_is_exact(self) -> None:
        rng = np.random.default_rng(2)
        hashes = [int(rng.integers(0, 2**63)) for _ in range(1500)]
        tree = BKTree((h, i) for i, h in enumerate(hashes))
        probe = hashes[10] ^ 0b1101
        found = {v for _d, _h, v in tree.find(probe, 5)}
        expected = {i for i, h in enumerate(hashes) if hamming(h, probe) <= 5}
        assert found == expected


class TestRanking:
    def test_rrf_prefers_broad_agreement(self) -> None:
        ranked = {
            "text": [("a", 0.9), ("b", 0.8)],
            "image": [("b", 0.4), ("a", 0.3)],
            "lexical": [("b", 5.0), ("c", 1.0)],
        }
        fused = reciprocal_rank_fusion(ranked, k=60, adaptive=False)
        assert fused.order[0] == "b"  # 2xrank-1 beats one rank-1 + one rank-2

    def test_adaptive_weights_penalise_flat_lists(self) -> None:
        weights = adaptive_modality_weights(
            {"sharp": [("a", 0.95), ("b", 0.2), ("c", 0.1)], "flat": [("a", 0.5), ("b", 0.5), ("c", 0.5)]}
        )
        assert weights["sharp"] > weights["flat"]

    def test_entropy_bounds(self) -> None:
        assert normalized_entropy([0.5] * 6) == pytest.approx(1.0, abs=1e-3)
        assert normalized_entropy([0.99, 0.01, 0.01, 0.01]) < 0.1

    def test_mmr_diversifies(self) -> None:
        base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb = {
            "a": base,
            "b": base + np.array([0.01, 0.0, 0.0], dtype=np.float32),
            "c": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }
        chosen = mmr(["a", "b", "c"], {"a": 1.0, "b": 0.98, "c": 0.5}, emb, k=2, lambda_=0.5)
        assert chosen == ["a", "c"]  # near-duplicate b is skipped

    def test_temporal_diffusion_is_video_scoped(self) -> None:
        scores = {"a": 1.0, "b": 0.0, "x": 0.0}
        spans = {"a": (0.0, 10.0), "b": (11.0, 20.0), "x": (11.0, 20.0)}
        out = temporal_diffusion(scores, spans, decay_s=30, bonus=0.5, same_video={"a": "v1", "b": "v1", "x": "v2"})
        assert out["b"] > 0.0
        assert out["x"] == 0.0

    def test_ranking_metrics(self) -> None:
        assert ndcg_at_k(["a", "b"], {"a": 1.0, "b": 1.0}, k=2) == pytest.approx(1.0)
        assert ndcg_at_k(["b", "a"], {"a": 1.0}, k=2) < 1.0

        assert recall_at_k(["a", "b", "c"], ["a", "c"], k=3) == 1.0
        assert recall_at_k(["a", "b", "c"], ["a", "z"], k=3) == 0.5
        assert recall_at_k(["a"], [], k=1) == 0.0

        assert mean_reciprocal_rank(["a", "b"], ["a"]) == 1.0
        assert mean_reciprocal_rank(["b", "a"], ["a"]) == 0.5
        assert mean_reciprocal_rank(["b", "c"], ["a"]) == 0.0


class TestSelectCuts:
    def test_dp_beats_greedy_on_strength(self) -> None:
        times = np.array([0.0, 1.0, 5.0], dtype=np.float64)
        strengths = np.array([1.0, 9.0, 9.0], dtype=np.float64)
        # Greedy left-to-right takes t=0 then t=5 (total 10);
        # the DP takes t=1 and t=5 (total 18).
        assert select_cuts(times, strengths, min_gap=2.0) == [1, 2]


class TestSandbox:
    @pytest.mark.parametrize(
        "code",
        [
            "import os",
            "__import__('os')",
            "open('/etc/passwd')",
            "(1).__class__.__bases__",
            "eval('1+1')",
            "f = lambda: 1",
            "while True:\n    pass",
            "[0] * 10**9",
            "10**10**9",
            "[x for x in range(10**9)]",
        ],
    )
    def test_blocks_escapes(self, code: str) -> None:
        with pytest.raises(SandboxViolation):
            repl.run_sync(code, timeout=0.5)

    def test_allows_analysis(self) -> None:
        result = repl.run_sync(
            "vals=[42,51,68,91]\nround(pct_change(vals[0], vals[-1]), 2)"
        )
        assert result.ok and result.value == pytest.approx(116.67)

    def test_reports_runtime_errors_without_raising(self) -> None:
        result = repl.run_sync("1/0")
        assert not result.ok and "ZeroDivisionError" in result.error


class TestNumeric:
    def test_word_numbers(self) -> None:
        assert words_to_number(["forty", "two", "million"]) == 42_000_000
        assert words_to_number(["one", "hundred", "twenty", "three"]) == 123

    @pytest.mark.parametrize(
        "text",
        [
            "We closed forty two million in the first quarter and fifty one million in the second. "
            "The third quarter came in at sixty eight million and the fourth quarter reached ninety one million.",
            "Q1 revenue was $42M, Q2 $51M, Q3 $68M and Q4 $91M.",
            "In the first quarter we saw 42, second quarter 51, third quarter 68, fourth quarter 91.",
        ],
    )
    def test_series_alignment(self, text: str) -> None:
        series = build_series(extract_numbers(text))
        assert series.labels == ["Q1", "Q2", "Q3", "Q4"]
        assert [v / max(series.values) for v in series.values] == pytest.approx(
            [42 / 91, 51 / 91, 68 / 91, 1.0], abs=1e-6
        )

    def test_ignores_incidental_numbers(self) -> None:
        series = build_series(extract_numbers("The rollout took two weeks and saved $1,200."))
        assert not series.ok


class TestReferenceResolution:
    """Follow-ups must inherit context; new questions must not."""

    @staticmethod
    def _history() -> list[Turn]:
        return [
            Turn(
                query="When does the speaker show the architecture diagram?",
                answer="The diagram appears eight seconds in.",
                citations=[
                    Citation(
                        chunk_id="c1", video_id="v", start=8.5, end=16.0,
                        quote="Here is the system architecture diagram",
                    ),
                    # Ranked lower but earlier: the anchor must not drift to it.
                    Citation(
                        chunk_id="c0", video_id="v", start=4.0, end=8.0,
                        quote="Today I will walk through the agenda",
                    ),
                ],
                keywords=["architecture", "diagram"],
            )
        ]

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            # Pointer words have no subject of their own.
            ("What did they say right after that?", True),
            ("Tell me more about it", True),
            ("who said that", True),
            ("and before that?", True),
            ("the second one", True),
            ("why?", True),
            # These name what they are about, however tersely.
            ("what about the revenue numbers", False),
            ("what about kubernetes rollout timelines", False),
            ("summarise the main points", False),
            ("revenue", False),
        ],
    )
    def test_classification(self, query: str, expected: bool) -> None:
        assert resolve(query, self._history()).is_followup is expected

    def test_first_question_is_never_a_followup(self) -> None:
        assert resolve("what about that?", []).is_followup is False

    def test_anchor_follows_relevance_not_chronology(self) -> None:
        """"After that" means after what was discussed, not the earliest citation."""
        window = resolve("what happened right after that?", self._history()).time_range
        assert window is not None
        assert window.start == pytest.approx(8.5), "anchored on the wrong citation"
        assert window.end > window.start

    def test_a_new_topic_keeps_its_own_words(self) -> None:
        resolution = resolve("what about the revenue numbers", self._history())
        assert "diagram" not in resolution.query, "a new question inherited stale context"
        assert resolution.time_range is None, "a new question must not be time-boxed"

    def test_carried_words_name_a_subject(self) -> None:
        """Filler verbs in a citation must not become search terms."""
        words = subject_words("Let us look at the revenue chart, this is quarterly revenue.")
        assert "revenue" in words and "chart" in words
        assert not {"let", "look", "this"} & set(words), f"filler leaked through: {words}"

    def test_a_followup_carries_the_previous_subject(self) -> None:
        resolution = resolve("what did they say right after that?", self._history())
        assert "architecture" in resolution.query or "diagram" in resolution.query
        assert resolution.notes, "a resolution that changes the query must explain itself"
