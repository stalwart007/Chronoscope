"""Lightweight encoders used when torch and CLIP are unavailable.

``HashingTextEncoder`` applies signed feature hashing over word unigrams,
bigrams and character 4-grams with sub-linear term-frequency damping, giving a
Johnson-Lindenstrauss sketch of a TF vector.

``ColorLayoutImageEncoder`` builds an MPEG-7 style descriptor from per-cell HSV
histograms on a 4x4 grid plus gradient energy. Text queries are projected
through a small colour and structure lexicon. Both are reported as degraded.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections import Counter

import numpy as np

from app.embed.base import ImageEncoder, TextEncoder, l2

_WORD = re.compile(r"[a-z0-9']+")
_STOP = frozenset(
    ["the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on", "at", "for", "with", "by", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that", "these", "those", "as", "from", "into", "about", "over", "under", "he", "she", "they", "we", "you", "i", "not", "no", "do", "does", "did", "so", "such", "than"]
)


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    """Signed feature hashing, the sign cancels collision bias in expectation."""
    h = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    v = int.from_bytes(h, "big")
    return v % dim, 1.0 if (v >> 63) & 1 else -1.0


class HashingTextEncoder(TextEncoder):
    name = "hashing-tfidf-sketch"
    degraded = True

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _features(self, text: str) -> Counter[str]:
        toks = _tokens(text)
        feats: Counter[str] = Counter()
        feats.update(f"w:{t}" for t in toks)
        feats.update(f"b:{a}_{b}" for a, b in itertools.pairwise(toks))
        squashed = " ".join(toks)
        feats.update(f"c:{squashed[i : i + 4]}" for i in range(0, max(0, len(squashed) - 3), 2))
        return feats

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, tf in self._features(text or "").items():
                idx, sign = _bucket(feat, self.dim)
                weight = 1.0 + math.log(tf)  # sub-linear TF damping
                if feat.startswith("c:"):
                    weight *= 0.45  # char-grams inform, they should not dominate
                elif feat.startswith("b:"):
                    weight *= 1.3  # bigrams carry more signal than unigrams
                out[row, idx] += sign * weight
        return l2(out)


_COLOR_WORDS = {
    "diagram": (0.15, 0.12, 0.85), "chart": (0.18, 0.20, 0.80), "graph": (0.16, 0.18, 0.82),
    "slide": (0.10, 0.08, 0.90), "text": (0.08, 0.05, 0.88), "code": (0.12, 0.10, 0.30),
    "face": (0.55, 0.45, 0.55), "person": (0.52, 0.42, 0.52), "speaker": (0.54, 0.44, 0.54),
    "dark": (0.30, 0.30, 0.15), "bright": (0.40, 0.30, 0.92), "table": (0.14, 0.10, 0.86),
}


class ColorLayoutImageEncoder(ImageEncoder):
    name = "color-layout-descriptor"
    degraded = True

    def __init__(self, grid: int = 4, hue_bins: int = 8) -> None:
        self.grid = grid
        self.hue_bins = hue_bins
        self.dim = grid * grid * (hue_bins + 4)

    def _describe(self, arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape[:2]
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        rgb = arr[:, :, :3].astype(np.float32) / 255.0
        mx, mn = rgb.max(axis=2), rgb.min(axis=2)
        val, chroma = mx, mx - mn
        sat = np.where(mx > 1e-6, chroma / np.maximum(mx, 1e-6), 0.0)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        hue = np.zeros_like(val)
        nz = chroma > 1e-6
        with np.errstate(invalid="ignore"):
            hue = np.where(nz & (mx == r), ((g - b) / np.maximum(chroma, 1e-6)) % 6, hue)
            hue = np.where(nz & (mx == g), (b - r) / np.maximum(chroma, 1e-6) + 2, hue)
            hue = np.where(nz & (mx == b), (r - g) / np.maximum(chroma, 1e-6) + 4, hue)
        hue = (hue / 6.0) % 1.0
        gy, gx = np.gradient(val)
        mag = np.hypot(gx, gy)

        feats = []
        ys = np.linspace(0, h, self.grid + 1).astype(int)
        xs = np.linspace(0, w, self.grid + 1).astype(int)
        for i in range(self.grid):
            for j in range(self.grid):
                cy, cx = slice(ys[i], ys[i + 1]), slice(xs[j], xs[j + 1])
                cell_h, cell_s, cell_v, cell_m = hue[cy, cx], sat[cy, cx], val[cy, cx], mag[cy, cx]
                hist, _ = np.histogram(cell_h, bins=self.hue_bins, range=(0, 1), weights=cell_s + 0.05)
                hist = hist / max(hist.sum(), 1e-6)
                feats.extend(hist.tolist())
                feats.extend(
                    [float(cell_v.mean()), float(cell_v.std()), float(cell_m.mean()), float((cell_m > 0.12).mean())]
                )
        return np.asarray(feats, dtype=np.float32)

    def encode_images(self, paths: list[str]) -> np.ndarray:
        from PIL import Image

        rows = []
        for p in paths:
            try:
                with Image.open(p) as handle:
                    frame = handle.convert("RGB").resize((160, 120))
                    rows.append(self._describe(np.asarray(frame)))
            except Exception:
                rows.append(np.zeros(self.dim, dtype=np.float32))
        return l2(np.stack(rows)) if rows else np.zeros((0, self.dim), dtype=np.float32)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Project text onto the descriptor space via a colour/structure lexicon."""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        cell = self.hue_bins + 4
        for row, text in enumerate(texts):
            toks = set(_tokens(text))
            sat_t, val_t, struct_t = 0.35, 0.5, 0.4
            hits = [_COLOR_WORDS[t] for t in toks if t in _COLOR_WORDS]
            if hits:
                sat_t = float(np.mean([h[0] for h in hits]))
                val_t = float(np.mean([h[1] for h in hits]))
                struct_t = float(np.mean([h[2] for h in hits]))
            for c in range(self.grid * self.grid):
                base = c * cell
                out[row, base : base + self.hue_bins] = sat_t / self.hue_bins
                out[row, base + self.hue_bins] = val_t
                out[row, base + self.hue_bins + 1] = 0.25
                out[row, base + self.hue_bins + 2] = struct_t * 0.3
                out[row, base + self.hue_bins + 3] = struct_t
        return l2(out)
