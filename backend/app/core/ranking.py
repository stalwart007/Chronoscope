"""Fusion, diversification and calibration for cross-modal retrieval.

One ANN search runs per modality. Their scores are not comparable (CLIP
cosines sit near 0.3 while a sentence encoder returns 0.8), so fusion happens
in rank space via Reciprocal Rank Fusion, with adaptive per-modality weights,
MMR diversification and a temporal kernel over neighbouring chunks.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-9


# --------------------------------------------------------------- calibration
def minmax(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < EPS:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    if not scores:
        return []
    arr = np.asarray(scores, dtype=np.float64) / max(temperature, EPS)
    arr -= arr.max()
    e = np.exp(arr)
    return (e / max(e.sum(), EPS)).tolist()


def normalized_entropy(scores: Sequence[float], temperature: float = 0.1) -> float:
    """Entropy of the softmax over scores, in [0, 1].

    0 => one item dominates (highly discriminative list).
    1 => perfectly flat list (the modality is telling us nothing).
    """
    n = len(scores)
    if n <= 1:
        return 0.0
    p = softmax(scores, temperature)
    h = -sum(pi * math.log(max(pi, EPS)) for pi in p)
    return float(min(1.0, max(0.0, h / math.log(n))))


def adaptive_modality_weights(
    ranked: Mapping[str, Sequence[tuple[str, float]]],
    *,
    prior: Mapping[str, float] | None = None,
    floor: float = 0.25,
) -> dict[str, float]:
    """Confidence weights per modality.

    ``w_m proportional to prior_m * (1 - H_m) * (1 + gap_m)`` where ``H_m`` is the normalised
    score entropy and ``gap_m`` the relative margin between rank 1 and the
    median. Weights are floored so a modality is attenuated, never silenced,
    then renormalised to sum to the number of modalities (keeping RRF's scale
    stable regardless of how many modalities took part).
    """
    prior = prior or {}
    raw: dict[str, float] = {}
    for name, items in ranked.items():
        scores = [s for _, s in items]
        if not scores:
            raw[name] = floor
            continue
        h = normalized_entropy(scores)
        top = max(scores)
        med = float(np.median(scores))
        gap = (top - med) / (abs(top) + EPS)
        raw[name] = max(floor, prior.get(name, 1.0) * (1.0 - h) * (1.0 + gap))
    total = sum(raw.values()) or 1.0
    scale = len(raw) / total
    return {k: round(v * scale, 6) for k, v in raw.items()}


# ---------------------------------------------------------------------- RRF
@dataclass(slots=True)
class FusionResult:
    order: list[str]
    scores: dict[str, float]
    ranks: dict[str, dict[str, int]] = field(default_factory=dict)
    raw: dict[str, dict[str, float]] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    contributions: dict[str, dict[str, float]] = field(default_factory=dict)


def reciprocal_rank_fusion(
    ranked: Mapping[str, Sequence[tuple[str, float]]],
    *,
    k: float = 60.0,
    weights: Mapping[str, float] | None = None,
    adaptive: bool = True,
    prior: Mapping[str, float] | None = None,
) -> FusionResult:
    r"""RRF over per-modality ranked lists.

    .. math::  \mathrm{RRF}(d) = \sum_{m \in M} \frac{w_m}{k + r_m(d)}

    ``k`` (default 60, from Cormack et al. 2009) damps the influence of the
    very top ranks so a single modality's rank-1 cannot overwhelm broad
    agreement further down the lists.
    """
    w = dict(weights) if weights else (adaptive_modality_weights(ranked, prior=prior) if adaptive else {})
    fused: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    raws: dict[str, dict[str, float]] = {}
    contrib: dict[str, dict[str, float]] = {}
    for modality, items in ranked.items():
        wm = float(w.get(modality, 1.0))
        for rank, (doc, score) in enumerate(items, start=1):
            c = wm / (k + rank)
            fused[doc] = fused.get(doc, 0.0) + c
            ranks.setdefault(doc, {})[modality] = rank
            raws.setdefault(doc, {})[modality] = float(score)
            contrib.setdefault(doc, {})[modality] = c
    order = sorted(fused, key=lambda d: (-fused[d], min(ranks[d].values()), d))
    return FusionResult(order=order, scores=fused, ranks=ranks, raw=raws, weights=w, contributions=contrib)


# ----------------------------------------------------------------------- MMR
def mmr(
    doc_ids: Sequence[str],
    relevance: Mapping[str, float],
    embeddings: Mapping[str, np.ndarray],
    *,
    k: int = 8,
    lambda_: float = 0.72,
) -> list[str]:
    r"""Maximal Marginal Relevance.

    .. math::
        \mathrm{MMR} = \arg\max_{d \in R \setminus S}
        \bigl[\lambda\,\mathrm{rel}(d) - (1-\lambda)\max_{s \in S}\mathrm{sim}(d,s)\bigr]

    The running ``max sim`` per candidate is cached and updated with the single
    newest selection, so the loop is O(k*n) rather than O(k^2*n).
    """
    if not doc_ids:
        return []
    k = min(k, len(doc_ids))
    pool = [d for d in doc_ids if d in embeddings]
    missing = [d for d in doc_ids if d not in embeddings]
    if not pool:
        return list(doc_ids)[:k]

    mat = np.stack([embeddings[d] for d in pool]).astype(np.float32)
    mat /= np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), EPS)
    rel_raw = [relevance.get(d, 0.0) for d in pool]
    rel = np.asarray(minmax(rel_raw), dtype=np.float32)

    selected: list[int] = [int(np.argmax(rel))]
    max_sim = mat @ mat[selected[0]]
    chosen = {selected[0]}
    while len(selected) < k and len(chosen) < len(pool):
        obj = lambda_ * rel - (1.0 - lambda_) * max_sim
        obj[list(chosen)] = -np.inf
        nxt = int(np.argmax(obj))
        if not np.isfinite(obj[nxt]):
            break
        selected.append(nxt)
        chosen.add(nxt)
        max_sim = np.maximum(max_sim, mat @ mat[nxt])
    out = [pool[i] for i in selected]
    out.extend(d for d in missing if d not in out)
    return out[:k]


# ------------------------------------------------------------- temporal glue
def temporal_diffusion(
    scores: Mapping[str, float],
    spans: Mapping[str, tuple[float, float]],
    *,
    decay_s: float = 45.0,
    bonus: float = 0.18,
    same_video: Mapping[str, str] | None = None,
) -> dict[str, float]:
    r"""Propagate confidence to temporal neighbours.

    Each chunk receives ``bonus * sum(s_j * exp(-gap_j / decay))`` from other
    chunks of the same video, where the gap is measured between span midpoints.
    Evidence for a question is usually spread across adjacent windows rather
    than confined to one.

    A sorted-midpoint sweep with a two-pointer window of radius ``4 * decay``
    gives O(n log n) instead of the O(n^2) all-pairs kernel.
    """
    if not scores:
        return {}
    items = [(cid, (spans[cid][0] + spans[cid][1]) / 2.0, sc) for cid, sc in scores.items() if cid in spans]
    if not items:
        return dict(scores)
    items.sort(key=lambda x: x[1])
    radius = 4.0 * decay_s
    out = {cid: sc for cid, _, sc in items}
    n = len(items)
    lo = 0
    for i in range(n):
        cid, mid, _ = items[i]
        while items[lo][1] < mid - radius:
            lo += 1
        acc = 0.0
        j = lo
        while j < n and items[j][1] <= mid + radius:
            if j != i:
                ocid, omid, osc = items[j]
                if same_video is None or same_video.get(ocid) == same_video.get(cid):
                    acc += osc * math.exp(-abs(omid - mid) / max(decay_s, EPS))
            j += 1
        out[cid] += bonus * acc
    return out


# ------------------------------------------------------------------- metrics
def dcg(gains: Iterable[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int = 10) -> float:
    gains = [relevant.get(d, 0.0) for d in ranked[:k]]
    ideal = sorted(relevant.values(), reverse=True)[:k]
    denom = dcg(ideal)
    return round(dcg(gains) / denom, 6) if denom > EPS else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    return round(len(set(ranked[:k]) & rel) / len(rel), 6)


def mean_reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    rel = set(relevant)
    for i, d in enumerate(ranked, start=1):
        if d in rel:
            return round(1.0 / i, 6)
    return 0.0
