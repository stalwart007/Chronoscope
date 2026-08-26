"""Perceptual hashing and a BK-tree for near-duplicate frame suppression.

Hamming distance is a metric, so the triangle inequality lets a BK-tree prune
most of the tree during a radius search instead of comparing every pair.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

_POPCOUNT = bytes(bin(i).count("1") for i in range(256))


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


@dataclass(slots=True)
class _BKNode:
    key: int
    value: Any
    children: dict[int, _BKNode] = field(default_factory=dict)


class BKTree:
    """Metric tree over integer hashes under Hamming distance."""

    __slots__ = ("_root", "_size")

    def __init__(self, items: Iterable[tuple[int, Any]] = ()) -> None:
        self._root: _BKNode | None = None
        self._size = 0
        for h, v in items:
            self.add(h, v)

    def __len__(self) -> int:
        return self._size

    def add(self, key: int, value: Any = None) -> None:
        if self._root is None:
            self._root = _BKNode(key, value)
            self._size = 1
            return
        node = self._root
        while True:
            d = hamming(key, node.key)
            if d == 0:
                return  # exact duplicate, nothing to store
            child = node.children.get(d)
            if child is None:
                node.children[d] = _BKNode(key, value)
                self._size += 1
                return
            node = child

    def find(self, key: int, radius: int) -> list[tuple[int, int, Any]]:
        """``(distance, hash, value)`` for every entry within ``radius``."""
        if self._root is None:
            return []
        out: list[tuple[int, int, Any]] = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            d = hamming(key, node.key)
            if d <= radius:
                out.append((d, node.key, node.value))
            lo, hi = d - radius, d + radius
            stack.extend(c for dist, c in node.children.items() if lo <= dist <= hi)
        out.sort(key=lambda t: t[0])
        return out

    def nearest(self, key: int, radius: int = 8) -> tuple[int, int, Any] | None:
        hits = self.find(key, radius)
        return hits[0] if hits else None


def dhash(gray: Any, size: int = 8) -> int:
    """Difference hash of a 2-D uint8 array (numpy), returned as a 64-bit int.

    dHash compares horizontally adjacent pixels of a downscaled image, so it is
    invariant to brightness/contrast shifts and mild compression, exactly the
    noise that separates two "identical" slide frames.
    """
    import numpy as np

    arr = np.asarray(gray, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    h, w = arr.shape
    ys = (np.linspace(0, h - 1, size)).astype(np.int32)
    xs = (np.linspace(0, w - 1, size + 1)).astype(np.int32)
    small = arr[np.ix_(ys, xs)]
    diff = small[:, 1:] > small[:, :-1]
    bits = diff.flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


