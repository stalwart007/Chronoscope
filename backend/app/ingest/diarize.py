"""Speaker diarisation.

Uses pyannote.audio when a HuggingFace token is configured. The fallback is a
spectral pipeline: MFCC and delta features from a numpy STFT, segment
embeddings, a refined cosine affinity matrix, speaker count from the largest
eigengap of the normalised Laplacian, then k-means on the leading eigenvectors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.config import settings
from app.core.interval_tree import IntervalTree
from app.core.types import SpeakerTurn, TimeSpan, TranscriptSegment
from app.logging_conf import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class DiarizationResult:
    turns: list[SpeakerTurn]
    speakers: list[str]
    source: str
    degraded: bool = False


# ------------------------------------------------------------------ features
def _hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_mels: int, n_fft: int, sr: int, fmin: float = 60.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sr / 2
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    freqs = _mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        if mid == lo:
            mid = lo + 1
        if hi == mid:
            hi = mid + 1
        hi = min(hi, fb.shape[1] - 1)
        if lo >= fb.shape[1] - 1:
            break
        fb[i, lo:mid] = np.linspace(0, 1, max(1, mid - lo), endpoint=False)
        fb[i, mid:hi] = np.linspace(1, 0, max(1, hi - mid), endpoint=False)
    return fb


def mfcc(audio: np.ndarray, sr: int = 16000, *, n_mfcc: int = 20, n_mels: int = 40,
         win_s: float = 0.025, hop_s: float = 0.010) -> np.ndarray:
    """(frames, n_mfcc*2) MFCC + delta features via a strided STFT."""
    n_fft = 1 << math.ceil(math.log2(max(2, int(sr * win_s))))
    hop = max(1, int(sr * hop_s))
    if audio.size < n_fft:
        return np.zeros((0, n_mfcc * 2), dtype=np.float32)
    pre = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])  # pre-emphasis
    n_frames = 1 + (pre.size - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        pre, shape=(n_frames, n_fft), strides=(pre.strides[0] * hop, pre.strides[0]), writeable=False
    )
    window = np.hanning(n_fft).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    fb = mel_filterbank(n_mels, n_fft, sr)
    mel = np.log(np.maximum(spec @ fb.T, 1e-10))
    # DCT-II (orthonormal): the cepstral transform.
    k = np.arange(n_mels)
    basis = np.cos(np.pi / n_mels * (k[None, :] + 0.5) * np.arange(n_mfcc)[:, None])
    cep = (mel @ basis.T) * math.sqrt(2.0 / n_mels)
    delta = np.gradient(cep, axis=0) if cep.shape[0] > 1 else np.zeros_like(cep)
    return np.concatenate([cep, delta], axis=1).astype(np.float32)


def segment_embeddings(audio: np.ndarray, spans: list[TimeSpan], sr: int = 16000) -> np.ndarray:
    feats = mfcc(audio, sr)
    if feats.shape[0] == 0:
        return np.zeros((len(spans), 1), dtype=np.float32)
    feats = feats - feats.mean(axis=0, keepdims=True)  # cepstral mean normalisation
    hop_s = 0.010
    rows = []
    for sp in spans:
        i0, i1 = int(sp.start / hop_s), int(sp.end / hop_s)
        window = feats[max(0, i0) : min(feats.shape[0], max(i1, i0 + 2))]
        if window.shape[0] < 2:
            rows.append(np.zeros(feats.shape[1] * 2, dtype=np.float32))
            continue
        rows.append(np.concatenate([window.mean(axis=0), window.std(axis=0)]))
    emb = np.asarray(rows, dtype=np.float32)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    if emb.shape[0] > 2:
        # Centre on the recording's own mean voice. Without this every pair of
        # segments sits at cosine ~ 0.99 (they share the room, mic and codec)
        # and the between-speaker signal is buried under the common component.
        emb = emb - emb.mean(axis=0, keepdims=True)
        emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    return emb


# ------------------------------------------------------------------ spectral
def refine_affinity(emb: np.ndarray, *, max_speakers: int | None = None) -> np.ndarray:
    n = emb.shape[0]
    a = emb @ emb.T
    np.fill_diagonal(a, 1.0)
    a = (a + 1.0) / 2.0  # cosine  in  [-1,1] -> affinity  in  [0,1]
    if n > 3:
        # Keep roughly one cluster's worth of neighbours per row. A fixed high
        # percentile prunes everything but the diagonal on short recordings.
        keep = max(2, min(n - 1, round(n / max(2, min(max_speakers or 4, n - 1))) + 1))
        thr = np.partition(a, n - keep, axis=1)[:, n - keep][:, None]
        a = np.where(a >= thr, a, a * 0.01)  # row-wise thresholding
    a = (a + a.T) / 2.0
    a = a @ a  # diffusion sharpens the block structure
    a /= max(a.max(), 1e-9)
    return a


def estimate_speakers(affinity: np.ndarray, max_speakers: int) -> tuple[int, np.ndarray, np.ndarray]:
    """Eigengap heuristic on the normalised Laplacian ``L = I - D^-1/2 A D^-1/2``."""
    n = affinity.shape[0]
    d = np.maximum(affinity.sum(axis=1), 1e-9)
    dinv = np.diag(1.0 / np.sqrt(d))
    lap = np.eye(n) - dinv @ affinity @ dinv
    vals, vecs = np.linalg.eigh(lap)
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    limit = min(max_speakers, n - 1)
    if limit < 2:
        return 1, vals, vecs
    gaps = np.diff(vals[: limit + 1])
    k = int(np.argmax(gaps)) + 1
    return max(1, min(k, limit)), vals, vecs


def kmeans(x: np.ndarray, k: int, *, iters: int = 60, restarts: int = 8, seed: int = 17) -> np.ndarray:
    """k-means with k-means++ seeding; returns the best-inertia labelling."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    if k <= 1 or n <= k:
        return np.zeros(n, dtype=np.int32) if k <= 1 else np.arange(n, dtype=np.int32) % k
    best_labels, best_inertia = np.zeros(n, dtype=np.int32), np.inf
    for _ in range(restarts):
        centers = [x[rng.integers(n)]]
        for _ in range(1, k):
            d2 = np.min(((x[:, None, :] - np.asarray(centers)[None, :, :]) ** 2).sum(-1), axis=1)
            probs = d2 / max(d2.sum(), 1e-12)
            centers.append(x[rng.choice(n, p=probs)])
        c = np.asarray(centers)
        labels = np.zeros(n, dtype=np.int32)
        for _ in range(iters):
            dist = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
            new = np.argmin(dist, axis=1).astype(np.int32)
            if np.array_equal(new, labels):
                break
            labels = new
            for j in range(k):
                pts = x[labels == j]
                if pts.size:
                    c[j] = pts.mean(axis=0)
        inertia = float(((x - c[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_labels, best_inertia = labels, inertia
    return best_labels


def spectral_cluster(emb: np.ndarray, max_speakers: int) -> tuple[np.ndarray, int]:
    n = emb.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.int32), 1
    affinity = refine_affinity(emb, max_speakers=max_speakers)
    k, _vals, vecs = estimate_speakers(affinity, max_speakers)
    if k <= 1:
        return np.zeros(n, dtype=np.int32), 1
    features = vecs[:, :k]
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-9)
    return kmeans(features, k), k


# ---------------------------------------------------------------- entrypoints
def diarize_pyannote(audio_path: str) -> DiarizationResult | None:
    if not settings.diarization_enabled or not settings.hf_token:
        return None
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        return None
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=settings.hf_token
        )
        annotation = pipeline(audio_path)
        turns = [
            SpeakerTurn(speaker=str(label), span=TimeSpan(start=round(seg.start, 3), end=round(seg.end, 3)))
            for seg, _, label in annotation.itertracks(yield_label=True)
        ]
        return DiarizationResult(turns=turns, speakers=sorted({t.speaker for t in turns}), source="pyannote")
    except Exception as exc:
        log.warning("pyannote diarization failed: %s", exc)
        return None


def diarize_spectral(audio: np.ndarray, spans: list[TimeSpan], sr: int = 16000) -> DiarizationResult:
    if not spans:
        return DiarizationResult(turns=[], speakers=[], source="spectral", degraded=True)
    emb = segment_embeddings(audio, spans, sr)
    labels, k = spectral_cluster(emb, settings.max_speakers)
    # Rename clusters by first appearance so SPEAKER_00 is whoever talks first.
    order: dict[int, str] = {}
    turns: list[SpeakerTurn] = []
    centroids = {int(j): emb[labels == j].mean(axis=0) for j in set(labels.tolist())}
    for i, (sp, lab) in enumerate(zip(spans, labels.tolist(), strict=True)):
        if lab not in order:
            order[lab] = f"SPEAKER_{len(order):02d}"
        cen = centroids[lab]
        conf = float(np.clip((emb[i] @ cen + 1) / 2, 0, 1)) if emb.shape[1] > 1 else 0.5
        turns.append(SpeakerTurn(speaker=order[lab], span=sp, confidence=round(conf, 4)))
    log.info("spectral diarization: %d speakers over %d regions", k, len(spans))
    return DiarizationResult(
        turns=turns, speakers=sorted(order.values()), source="spectral-mfcc", degraded=True
    )


def assign_speakers(segments: list[TranscriptSegment], turns: list[SpeakerTurn]) -> list[str]:
    """Attach a speaker to every transcript segment via interval overlap.

    Built on the interval tree: for each segment we take the turn with maximum
    temporal overlap. O(log n + k) per segment instead of scanning all turns.
    """
    if not turns:
        return []
    tree = IntervalTree((t.span.start, t.span.end, t) for t in turns)
    seen: list[str] = []
    for seg in segments:
        if seg.speaker:  # already labelled (e.g. sidecar with names)
            if seg.speaker not in seen:
                seen.append(seg.speaker)
            continue
        best = tree.best_overlap(seg.span.start, seg.span.end)
        if best is None:
            mid = seg.span.mid
            stabbed = tree.stab(mid)
            best = stabbed[0] if stabbed else None
        if best is not None:
            seg.speaker = best.payload.speaker
            if seg.speaker not in seen:
                seen.append(seg.speaker)
    return seen
