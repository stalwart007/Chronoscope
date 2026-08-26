"""Hierarchical Navigable Small World index (Malkov and Yashunin, 2018).

Used as the fallback vector backend when Qdrant is unavailable, and as a
deterministic index for tests. Vectors are L2-normalised on insert so cosine
similarity reduces to an inner product and distance is ``1 - ip``.

Neighbour selection uses the heuristic variant (Algorithm 4), which keeps
long-range links that naive top-M truncation would discard. Deletes are
tombstoned and reclaimed by ``compact()``.
"""

from __future__ import annotations

import heapq
import json
import math
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


@dataclass(slots=True)
class _Meta:
    key: str
    payload: dict[str, Any]
    deleted: bool = False


class HNSW:
    """Approximate nearest-neighbour index over unit vectors."""

    def __init__(
        self,
        dim: int,
        *,
        m: int = 24,
        ef_construction: int = 200,
        ef_search: int = 96,
        seed: int = 1337,
        capacity: int = 1024,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.M = m
        self.M0 = m * 2  # ground layer gets double connectivity
        self.ef_construction = max(ef_construction, m)
        self.ef_search = max(ef_search, 1)
        self.mL = 1.0 / math.log(max(2, m))
        self._rng = random.Random(seed)
        self._lock = threading.RLock()

        self._vectors = np.zeros((capacity, dim), dtype=np.float32)
        self._count = 0
        self._meta: list[_Meta] = []
        self._key_to_id: dict[str, int] = {}
        #: ``_graph[level][node] -> list[node]``
        self._graph: list[dict[int, list[int]]] = []
        self._entry: int | None = None
        self._max_level = -1
        self._deleted = 0

    # ------------------------------------------------------------- properties
    def __len__(self) -> int:
        return self._count - self._deleted

    @property
    def levels(self) -> int:
        return self._max_level + 1

    def stats(self) -> dict[str, Any]:
        deg = [len(v) for lvl in self._graph for v in lvl.values()]
        return {
            "size": len(self),
            "raw": self._count,
            "deleted": self._deleted,
            "levels": self.levels,
            "dim": self.dim,
            "avg_degree": round(float(np.mean(deg)), 2) if deg else 0.0,
            "M": self.M,
            "ef_search": self.ef_search,
        }

    # ------------------------------------------------------------------ store
    def _ensure_capacity(self, extra: int = 1) -> None:
        need = self._count + extra
        if need <= self._vectors.shape[0]:
            return
        new_cap = max(need, int(self._vectors.shape[0] * 1.7) + 8)
        grown = np.zeros((new_cap, self.dim), dtype=np.float32)
        grown[: self._count] = self._vectors[: self._count]
        self._vectors = grown

    def _random_level(self) -> int:
        return int(-math.log(max(self._rng.random(), 1e-12)) * self.mL)

    def _dist(self, q: np.ndarray, ids: list[int]) -> np.ndarray:
        """Batched cosine distance over a set of node ids."""
        return 1.0 - self._vectors[ids] @ q

    def _dist_one(self, q: np.ndarray, i: int) -> float:
        return float(1.0 - self._vectors[i] @ q)

    # ----------------------------------------------------------------- insert
    def add(self, key: str, vector: np.ndarray, payload: dict[str, Any] | None = None) -> int:
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vec.shape[0]}")
        vec = l2_normalize(vec)
        with self._lock:
            if key in self._key_to_id:  # upsert semantics
                self.remove(key)
            self._ensure_capacity()
            node = self._count
            self._vectors[node] = vec
            self._meta.append(_Meta(key=key, payload=payload or {}))
            self._key_to_id[key] = node
            self._count += 1

            level = self._random_level()
            while len(self._graph) <= level:
                self._graph.append({})
            for lv in range(level + 1):
                self._graph[lv].setdefault(node, [])

            if self._entry is None:
                self._entry, self._max_level = node, level
                return node

            ep = self._entry
            # Phase 1: greedy descent through layers above the new node.
            for lv in range(self._max_level, level, -1):
                ep = self._greedy(vec, ep, lv)
            # Phase 2: connect on every layer the node belongs to.
            for lv in range(min(level, self._max_level), -1, -1):
                candidates = self._search_layer(vec, [ep], self.ef_construction, lv)
                m = self.M0 if lv == 0 else self.M
                neighbours = self._select_heuristic(vec, candidates, m)
                self._graph[lv][node] = list(neighbours)
                for nb in neighbours:
                    adj = self._graph[lv].setdefault(nb, [])
                    adj.append(node)
                    if len(adj) > m:
                        # One batched distance computation for the whole
                        # adjacency list rather than one call per neighbour.
                        adj_d = self._dist(self._vectors[nb], adj)
                        pruned = self._select_heuristic(
                            self._vectors[nb], list(zip(adj_d.tolist(), adj, strict=True)), m
                        )
                        self._graph[lv][nb] = list(pruned)
                ep = candidates[0][1] if candidates else ep

            if level > self._max_level:
                self._max_level, self._entry = level, node
            return node

    def add_batch(self, keys: list[str], vectors: np.ndarray, payloads: list[dict[str, Any]] | None = None) -> None:
        payloads = payloads or [{} for _ in keys]
        self._ensure_capacity(len(keys))
        for k, v, p in zip(keys, vectors, payloads, strict=True):
            self.add(k, v, p)

    # ----------------------------------------------------------------- search
    def _greedy(self, q: np.ndarray, entry: int, level: int) -> int:
        """Zoom-in: walk downhill on one layer until no neighbour improves."""
        cur, cur_d = entry, self._dist_one(q, entry)
        improved = True
        while improved:
            improved = False
            adj = self._graph[level].get(cur, ())
            if not adj:
                break
            ds = self._dist(q, list(adj))
            j = int(np.argmin(ds))
            if ds[j] < cur_d:
                cur, cur_d, improved = adj[j], float(ds[j]), True
        return cur

    def _search_layer(self, q: np.ndarray, entries: list[int], ef: int, level: int) -> list[tuple[float, int]]:
        """Best-first beam search on one layer.

        ``cand`` is a min-heap of the frontier; ``top`` is a max-heap (negated)
        of the current ef best. Terminates when the closest frontier item is
        worse than the worst kept result, which is the HNSW stopping rule.
        """
        visited: set[int] = set(entries)
        cand: list[tuple[float, int]] = []
        top: list[tuple[float, int]] = []
        entry_d = self._dist(q, entries)
        for d, e in zip(entry_d.tolist(), entries, strict=True):
            heapq.heappush(cand, (d, e))
            heapq.heappush(top, (-d, e))
        graph = self._graph[level]
        while cand:
            d, c = heapq.heappop(cand)
            if top and d > -top[0][0] and len(top) >= ef:
                break
            unseen = [n for n in graph.get(c, ()) if n not in visited]
            if not unseen:
                continue
            visited.update(unseen)
            ds = self._dist(q, unseen)
            worst = -top[0][0] if top else float("inf")
            for nd, n in zip(ds.tolist(), unseen, strict=True):
                if len(top) < ef or nd < worst:
                    heapq.heappush(cand, (nd, n))
                    heapq.heappush(top, (-nd, n))
                    if len(top) > ef:
                        heapq.heappop(top)
                    worst = -top[0][0]
        return sorted(((-d, n) for d, n in top))

    def _select_heuristic(self, q: np.ndarray, candidates: list[tuple[float, int]], m: int) -> list[int]:
        """Algorithm 4: keep diverse, navigable links instead of nearest-M.

        The reference formulation compares each candidate against every kept
        neighbour one pair at a time, which is O(m^2) Python-level numpy calls
        and dominates construction time.

        Candidate sets are small (m is 16-48), so one BLAS call for the whole
        pairwise matrix followed by a plain Python loop over floats is faster
        than a vectorised loop whose per-iteration overhead exceeds the
        arithmetic it performs.
        """
        if not candidates:
            return []
        pool = sorted(candidates)
        ids: list[int] = []
        dists: list[float] = []
        seen: set[int] = set()
        for d, c in pool:  # dedupe, keeping the closest occurrence
            if c not in seen:
                seen.add(c)
                ids.append(c)
                dists.append(d)
        if len(ids) <= 1:
            return ids

        vecs = self._vectors[ids]
        # Single symmetric product; `n` is tiny so the n^2 memory is irrelevant.
        pairwise = (1.0 - vecs @ vecs.T).tolist()

        kept: list[int] = [0]
        nearest = list(pairwise[0])
        alive = [True] * len(ids)
        alive[0] = False
        while len(kept) < m:
            pick = -1
            for i, ok in enumerate(alive):
                # Closest remaining candidate that is nearer to the query than
                # to anything already kept: the diversity condition.
                if ok and nearest[i] > dists[i]:
                    pick = i
                    break
            if pick < 0:
                break
            kept.append(pick)
            alive[pick] = False
            row = pairwise[pick]
            for i in range(len(nearest)):
                nearest[i] = min(nearest[i], row[i])
        if len(kept) < m:  # backfill so the graph never fragments
            for i, ok in enumerate(alive):
                if ok:
                    kept.append(i)
                    if len(kept) >= m:
                        break
        return [ids[i] for i in kept]

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        *,
        ef: int | None = None,
        predicate: Any = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Return ``(key, similarity, payload)`` sorted by descending similarity.

        ``predicate`` is an optional ``(key, payload) -> bool`` filter applied
        post-hoc; ``ef`` is widened automatically so filtering does not starve
        the result set.
        """
        with self._lock:
            if self._entry is None or len(self) == 0:
                return []
            q = l2_normalize(np.asarray(query, dtype=np.float32).reshape(-1))
            if q.shape[0] != self.dim:
                raise ValueError(f"expected dim {self.dim}, got {q.shape[0]}")
            ef_eff = max(ef or self.ef_search, k)
            if predicate is not None:
                ef_eff = min(max(ef_eff * 4, 64), max(self._count, 1))
            ep = self._entry
            for lv in range(self._max_level, 0, -1):
                ep = self._greedy(q, ep, lv)
            found = self._search_layer(q, [ep], ef_eff, 0)
            out: list[tuple[str, float, dict[str, Any]]] = []
            for d, node in found:
                meta = self._meta[node]
                if meta.deleted:
                    continue
                if predicate is not None and not predicate(meta.key, meta.payload):
                    continue
                out.append((meta.key, round(1.0 - d, 6), meta.payload))
                if len(out) >= k:
                    break
            return out

    def brute_force(self, query: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        """Exact search, the ground truth used to measure recall in tests."""
        q = l2_normalize(np.asarray(query, dtype=np.float32).reshape(-1))
        sims = self._vectors[: self._count] @ q
        order = np.argsort(-sims)
        out = []
        for i in order:
            if self._meta[i].deleted:
                continue
            out.append((self._meta[i].key, float(sims[i])))
            if len(out) >= k:
                break
        return out

    # ---------------------------------------------------------------- mutate
    def remove(self, key: str) -> bool:
        with self._lock:
            node = self._key_to_id.pop(key, None)
            if node is None or self._meta[node].deleted:
                return False
            self._meta[node].deleted = True
            self._deleted += 1
            if self._deleted > 64 and self._deleted > 0.35 * self._count:
                self.compact()
            return True

    def get(self, key: str) -> np.ndarray | None:
        i = self._key_to_id.get(key)
        return None if i is None else self._vectors[i].copy()

    def compact(self) -> None:
        """Rebuild without tombstones. O(n log n) but amortised across deletes."""
        keys, vecs, payloads = [], [], []
        for i, meta in enumerate(self._meta):
            if not meta.deleted:
                keys.append(meta.key)
                vecs.append(self._vectors[i])
                payloads.append(meta.payload)
        self._reset()
        if keys:
            self.add_batch(keys, np.asarray(vecs, dtype=np.float32), payloads)

    def _reset(self) -> None:
        self._vectors = np.zeros((max(8, len(self._meta)), self.dim), dtype=np.float32)
        self._count = 0
        self._meta = []
        self._key_to_id = {}
        self._graph = []
        self._entry = None
        self._max_level = -1
        self._deleted = 0

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            np.savez_compressed(path.with_suffix(".npz"), vectors=self._vectors[: self._count])
            meta = {
                "dim": self.dim,
                "M": self.M,
                "ef_construction": self.ef_construction,
                "ef_search": self.ef_search,
                "entry": self._entry,
                "max_level": self._max_level,
                "deleted": self._deleted,
                "meta": [{"key": m.key, "payload": m.payload, "deleted": m.deleted} for m in self._meta],
                "graph": [{str(k): v for k, v in lvl.items()} for lvl in self._graph],
            }
            path.with_suffix(".json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: str | Path) -> HNSW:
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        idx = cls(
            meta["dim"], m=meta["M"], ef_construction=meta["ef_construction"], ef_search=meta["ef_search"]
        )
        vecs = np.load(path.with_suffix(".npz"))["vectors"].astype(np.float32)
        idx._vectors = vecs
        idx._count = vecs.shape[0]
        idx._meta = [_Meta(m["key"], m["payload"], m["deleted"]) for m in meta["meta"]]
        idx._key_to_id = {m.key: i for i, m in enumerate(idx._meta) if not m.deleted}
        idx._graph = [{int(k): list(v) for k, v in lvl.items()} for lvl in meta["graph"]]
        idx._entry = meta["entry"]
        idx._max_level = meta["max_level"]
        idx._deleted = meta["deleted"]
        return idx
