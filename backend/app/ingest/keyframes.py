"""Keyframe selection: budget allocation, quality scoring and dedup.

A global budget is apportioned across scenes by information demand using the
largest-remainder method, so long static scenes do not crowd out short busy
ones. Candidates are scored on sharpness, entropy, exposure and text density,
spread within the scene by dynamic programming, then de-duplicated through a
BK-tree over dHash.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import settings
from app.core.bktree import BKTree, dhash
from app.core.types import Keyframe, Scene, stable_id
from app.ingest.decode import grab_frames, save_frame
from app.ingest.scenes import ContentSignal, select_cuts
from app.logging_conf import get_logger

log = get_logger(__name__)

_LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


@dataclass(slots=True)
class FrameScore:
    timestamp: float
    sharpness: float
    entropy: float
    exposure: float
    text_density: float
    quality: float
    phash: int

    @property
    def is_slide(self) -> bool:
        return self.text_density > 0.055 and self.entropy < 6.2


def _conv2_valid(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Small-kernel 2-D convolution via stride tricks (no SciPy dependency)."""
    kh, kw = kernel.shape
    win = np.lib.stride_tricks.sliding_window_view(img, (kh, kw))
    return np.einsum("ijkl,kl->ij", win, kernel)


def score_frame(rgb: np.ndarray, timestamp: float) -> FrameScore:
    f = rgb.astype(np.float32) / 255.0
    luma = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    small = luma[:: max(1, luma.shape[0] // 240), :: max(1, luma.shape[1] // 320)]

    lap = _conv2_valid(small, _LAPLACIAN)
    sharpness = float(lap.var())

    hist, _ = np.histogram(small, bins=64, range=(0.0, 1.0))
    p = hist / max(hist.sum(), 1)
    nz = p[p > 0]
    entropy = float(-(nz * np.log2(nz)).sum())

    mean = float(small.mean())
    exposure = 1.0 - min(1.0, abs(mean - 0.5) * 2.4)

    # Text/diagram proxy: many thin, high-contrast strokes at fine scale.
    gy, gx = np.gradient(small)
    mag = np.hypot(gx, gy)
    strong = mag > max(0.14, float(np.percentile(mag, 92)))
    text_density = float(strong.mean())

    quality = float(
        0.34 * math.tanh(sharpness * 220.0)
        + 0.24 * min(1.0, entropy / 7.5)
        + 0.20 * exposure
        + 0.22 * min(1.0, text_density * 9.0)
    )
    return FrameScore(
        timestamp=timestamp,
        sharpness=round(sharpness, 6),
        entropy=round(entropy, 4),
        exposure=round(exposure, 4),
        text_density=round(text_density, 5),
        quality=round(quality, 5),
        phash=dhash(small * 255.0),
    )


def scene_activity(signal: ContentSignal, scene: Scene) -> float:
    mask = (signal.times >= scene.span.start) & (signal.times < scene.span.end)
    win = signal.values[mask]
    if win.size == 0:
        return 0.0
    return float(np.clip(win.mean() / 12.0, 0.0, 3.0))


def allocate_budget(scenes: list[Scene], signal: ContentSignal, budget: int) -> dict[int, int]:
    r"""Largest-remainder apportionment of ``budget`` frames across scenes.

    demand_i = sqrtduration_i * (0.35 + activity_i) * (1.15 - static_ratio_i)

    The square root damps very long scenes (a 20-minute Q&A does not deserve
    20x the frames of a 1-minute demo), while activity and non-staticness push
    budget toward scenes where the picture actually changes.
    """
    if not scenes:
        return {}
    budget = max(len(scenes), min(budget, settings.max_keyframes))
    demands: dict[int, float] = {}
    for sc in scenes:
        act = scene_activity(signal, sc)
        demands[sc.index] = math.sqrt(max(sc.span.duration, 0.25)) * (0.35 + act) * (1.15 - sc.static_ratio)
    total = sum(demands.values()) or 1.0
    free = budget - len(scenes)  # one guaranteed frame each
    exact = {i: free * d / total for i, d in demands.items()}
    alloc = {i: math.floor(v) for i, v in exact.items()}
    remainder = free - sum(alloc.values())
    for i, _ in sorted(exact.items(), key=lambda kv: kv[1] - math.floor(kv[1]), reverse=True)[: max(0, remainder)]:
        alloc[i] += 1
    return {i: alloc.get(i, 0) + 1 for i in demands}


def candidate_timestamps(scene: Scene, n_wanted: int, *, max_candidates: int = 9) -> list[float]:
    """Oversample within the scene so scoring has something to choose from."""
    n = min(max_candidates, max(2, n_wanted * 3))
    lo, hi = scene.span.start, scene.span.end
    if hi - lo < 0.2:
        return [max(0.0, (lo + hi) / 2.0)]
    pad = min(0.35, (hi - lo) * 0.12)  # skip the transition itself
    return list(np.linspace(lo + pad, hi - pad, n))


def extract_keyframes(
    video_path: str,
    video_id: str,
    scenes: list[Scene],
    signal: ContentSignal,
    out_dir: Path,
    *,
    budget: int | None = None,
    progress: object = None,
) -> list[Keyframe]:
    budget = budget or settings.max_keyframes
    alloc = allocate_budget(scenes, signal, budget)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1, decode every candidate in one sorted seek pass and score it.
    wanted: list[tuple[float, int]] = []
    for sc in scenes:
        for ts in candidate_timestamps(sc, alloc.get(sc.index, 1)):
            wanted.append((float(ts), sc.index))
    wanted.sort()
    frames = grab_frames(video_path, [t for t, _ in wanted], max_dim=settings.frame_max_dim)
    if not frames:
        log.warning("no frames decoded for %s", video_path)
        return []

    # Map each decoded frame back to the scene that requested it. Both lists
    # are sorted, so a binary search avoids an O(n*m) nearest-neighbour scan.
    want_times = [t for t, _ in wanted]
    scene_of: list[int] = []
    for t, _ in frames:
        j = bisect.bisect_left(want_times, t)
        best, best_d = wanted[min(j, len(wanted) - 1)][1], float("inf")
        for cand in (j - 1, j, j + 1):
            if 0 <= cand < len(wanted):
                d = abs(want_times[cand] - t)
                if d < best_d:
                    best, best_d = wanted[cand][1], d
        scene_of.append(best)

    scored: dict[int, list[tuple[FrameScore, np.ndarray]]] = {}
    for (t, arr), sidx in zip(frames, scene_of, strict=True):
        scored.setdefault(sidx, []).append((score_frame(arr, t), arr))

    # Phase 2, per-scene selection: max total quality with a minimum gap.
    picked: list[tuple[int, FrameScore, np.ndarray]] = []
    for sc in scenes:
        items = sorted(scored.get(sc.index, []), key=lambda p: p[0].timestamp)
        if not items:
            continue
        k = max(1, alloc.get(sc.index, 1))
        times = np.asarray([p[0].timestamp for p in items], dtype=np.float64)
        quals = np.asarray([p[0].quality for p in items], dtype=np.float64)
        min_gap = max(0.6, sc.span.duration / (k + 1)) if k > 1 else 0.0
        keep = select_cuts(times, quals, min_gap) if k > 1 else [int(np.argmax(quals))]
        if len(keep) > k:  # DP may exceed the quota, trim to the best k
            keep = sorted(sorted(keep, key=lambda i: -quals[i])[:k])
        for i in keep:
            picked.append((sc.index, items[i][0], items[i][1]))

    # Phase 3, global perceptual dedup (BK-tree over dHash).
    tree: BKTree = BKTree()
    kept: list[tuple[int, FrameScore, np.ndarray]] = []
    radius = settings.keyframe_dedupe_hamming
    best_of_scene: dict[int, float] = {}
    for sidx, fs, _ in picked:
        if fs.quality > best_of_scene.get(sidx, -1.0):
            best_of_scene[sidx] = fs.quality
    protected = {(sidx, best_of_scene[sidx]) for sidx in best_of_scene}
    for sidx, fs, arr in sorted(picked, key=lambda p: p[1].timestamp):
        is_scene_rep = (sidx, fs.quality) in protected
        if not is_scene_rep and tree.nearest(fs.phash, radius) is not None:
            continue
        if is_scene_rep:
            protected.discard((sidx, fs.quality))  # only the first match wins
        tree.add(fs.phash, fs.timestamp)
        kept.append((sidx, fs, arr))

    # Phase 4, persist.
    out: list[Keyframe] = []
    for sidx, fs, arr in kept:
        kf_id = stable_id(video_id, "kf", round(fs.timestamp, 3))
        rel = Path(video_id) / f"{kf_id}.jpg"
        w, h = save_frame(arr, out_dir / rel)
        out.append(
            Keyframe(
                id=kf_id,
                scene_index=sidx,
                timestamp=round(fs.timestamp, 3),
                path=str(rel),
                width=w,
                height=h,
                phash=fs.phash,
                quality=fs.quality,
                sharpness=fs.sharpness,
                entropy=fs.entropy,
                text_density=fs.text_density,
                is_slide=fs.is_slide,
            )
        )
        if callable(progress):
            progress(len(out), len(kept))
    log.info("kept %d/%d candidate keyframes (budget %d)", len(out), len(frames), budget)
    return out
