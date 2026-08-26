"""Performance characteristics that are easy to regress silently.

These assert *complexity*, not wall-clock: absolute timings are meaningless on
shared CI runners, but a routine that was O(corpus) and becomes O(document)
again will fail the ratio checks below.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from app.core.hnsw import HNSW, l2_normalize
from app.ingest.scenes import _robust_z
from app.retrieval.lexical import BM25Index


def _build_index(n_docs: int, vocab_size: int) -> BM25Index:
    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(vocab_size)]
    index = BM25Index()
    for d in range(n_docs):
        index.add(f"c{d}", f"v{d % 10}", " ".join(rng.choice(vocab, 40)))
    return index


def _time_removals(index: BM25Index, count: int) -> float:
    start = time.perf_counter()
    for d in range(count):
        index.remove(f"c{d}")
    return time.perf_counter() - start


def test_bm25_removal_does_not_scale_with_vocabulary() -> None:
    """Deleting a document must touch only its own terms.

    The naive implementation walked every posting list, so a ten-fold larger
    vocabulary made each delete ten times slower, re-indexing a library
    degraded quadratically.
    """
    small = _time_removals(_build_index(400, 500), 100)
    large = _time_removals(_build_index(400, 20_000), 100)
    assert large < small * 6 + 0.05, f"removal scaled with vocabulary: {small:.4f}s -> {large:.4f}s"


def test_bm25_removal_is_correct() -> None:
    index = BM25Index()
    index.add("a", "v1", "kubernetes rollout finished early")
    index.add("b", "v1", "quarterly revenue reached ninety one million")
    assert index.search("kubernetes")[0][0] == "a"
    index.remove("a")
    assert index.search("kubernetes") == []
    assert index.search("revenue")[0][0] == "b"
    assert index.n_docs == 1


def test_robust_z_matches_reference_implementation() -> None:
    """The vectorised rolling median/MAD must agree with the obvious loop."""
    rng = np.random.default_rng(3)
    values = (rng.random(400) * 40).astype(np.float32)
    half = 7
    padded = np.pad(values, half, mode="reflect")
    expected = np.asarray(
        [
            (values[i] - np.median(padded[i : i + 2 * half + 1]))
            / max(float(np.median(np.abs(padded[i : i + 2 * half + 1] - np.median(padded[i : i + 2 * half + 1]))) * 1.4826), 1.0)
            for i in range(values.size)
        ],
        dtype=np.float32,
    )
    assert np.allclose(_robust_z(values), expected, atol=1e-4)


def test_hnsw_recall_stays_high_after_optimisation() -> None:
    """Neighbour selection was rewritten for speed; recall must not move."""
    rng = np.random.default_rng(0)
    vectors = l2_normalize(rng.normal(size=(1500, 64)).astype(np.float32))
    index = HNSW(64, m=16, ef_construction=150, ef_search=96)
    for i in range(vectors.shape[0]):
        index.add(f"k{i}", vectors[i])
    queries = l2_normalize(rng.normal(size=(30, 64)).astype(np.float32))
    recall = sum(
        len({k for k, _, _ in index.search(q, 10)} & {k for k, _ in index.brute_force(q, 10)}) / 10
        for q in queries
    ) / len(queries)
    assert recall > 0.95, f"recall regressed to {recall:.3f}"
    assert index.stats()["avg_degree"] > 8, "graph must stay well connected"


@pytest.mark.parametrize("n", [64, 512])
def test_hnsw_handles_degenerate_input(n: int) -> None:
    """Identical vectors are a classic way to break neighbour heuristics."""
    index = HNSW(8, m=8)
    same = np.ones(8, dtype=np.float32)
    for i in range(n):
        index.add(f"k{i}", same)
    assert len(index) == n
    assert len(index.search(same, 5)) == 5
