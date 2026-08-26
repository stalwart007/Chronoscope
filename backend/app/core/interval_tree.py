"""Height-balanced interval tree and sweep-line helpers.

Aligning transcript segments against scenes, speaker turns and keyframes is an
interval-overlap problem. Nested loops are O(N*M); this answers an overlap
query in O(log n + k). Nodes are augmented with ``max_end`` so subtrees that
cannot overlap the query are pruned immediately.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class _Node(Generic[T]):
    start: float
    end: float
    payload: T
    left: _Node[T] | None = None
    right: _Node[T] | None = None
    height: int = 1
    max_end: float = 0.0

    def __post_init__(self) -> None:
        self.max_end = self.end


def _h(n: _Node[T] | None) -> int:
    return n.height if n else 0


def _me(n: _Node[T] | None) -> float:
    return n.max_end if n else float("-inf")


def _pull(n: _Node[T]) -> None:
    n.height = 1 + max(_h(n.left), _h(n.right))
    n.max_end = max(n.end, _me(n.left), _me(n.right))


def _rotate_right(y: _Node[T]) -> _Node[T]:
    x = y.left
    assert x is not None
    y.left, x.right = x.right, y
    _pull(y)
    _pull(x)
    return x


def _rotate_left(x: _Node[T]) -> _Node[T]:
    y = x.right
    assert y is not None
    x.right, y.left = y.left, x
    _pull(x)
    _pull(y)
    return y


def _balance(n: _Node[T]) -> _Node[T]:
    _pull(n)
    bf = _h(n.left) - _h(n.right)
    if bf > 1:
        assert n.left is not None
        if _h(n.left.left) < _h(n.left.right):
            n.left = _rotate_left(n.left)
        return _rotate_right(n)
    if bf < -1:
        assert n.right is not None
        if _h(n.right.right) < _h(n.right.left):
            n.right = _rotate_right(n.right)
        return _rotate_left(n)
    return n


@dataclass(slots=True)
class Interval(Generic[T]):
    start: float
    end: float
    payload: T

    def overlap(self, lo: float, hi: float) -> float:
        return max(0.0, min(self.end, hi) - max(self.start, lo))


class IntervalTree(Generic[T]):
    """Balanced interval tree over half-open ``[start, end)`` intervals."""

    __slots__ = ("_root", "_size")

    def __init__(self, items: Iterable[tuple[float, float, T]] | None = None) -> None:
        self._root: _Node[T] | None = None
        self._size = 0
        if items:
            # Bulk build from a sorted array => perfectly balanced, O(n log n).
            arr = sorted(items, key=lambda it: (it[0], it[1]))
            self._root = self._build(arr, 0, len(arr) - 1)
            self._size = len(arr)

    def _build(self, arr: list[tuple[float, float, T]], lo: int, hi: int) -> _Node[T] | None:
        if lo > hi:
            return None
        mid = (lo + hi) // 2
        s, e, p = arr[mid]
        node = _Node(s, e, p)
        node.left = self._build(arr, lo, mid - 1)
        node.right = self._build(arr, mid + 1, hi)
        _pull(node)
        return node

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0

    @property
    def height(self) -> int:
        return _h(self._root)

    def insert(self, start: float, end: float, payload: T) -> None:
        def go(n: _Node[T] | None) -> _Node[T]:
            if n is None:
                return _Node(start, max(start, end), payload)
            if (start, end) < (n.start, n.end):
                n.left = go(n.left)
            else:
                n.right = go(n.right)
            return _balance(n)

        self._root = go(self._root)
        self._size += 1

    def query(self, lo: float, hi: float) -> list[Interval[T]]:
        """All intervals overlapping ``[lo, hi)``. O(log n + k)."""
        out: list[Interval[T]] = []
        stack: list[_Node[T]] = [self._root] if self._root else []
        while stack:
            n = stack.pop()
            if n.max_end <= lo:  # entire subtree ends before the window
                continue
            if n.left is not None:
                stack.append(n.left)
            if n.start < hi:
                if n.end > lo:
                    out.append(Interval(n.start, n.end, n.payload))
                if n.right is not None:
                    stack.append(n.right)
        out.sort(key=lambda i: (i.start, i.end))
        return out

    def stab(self, t: float) -> list[Interval[T]]:
        """All intervals containing instant ``t``."""
        return self.query(t, t + 1e-9)

    def best_overlap(self, lo: float, hi: float) -> Interval[T] | None:
        """The single interval sharing the most time with ``[lo, hi)``."""
        best: Interval[T] | None = None
        best_ov = 0.0
        for iv in self.query(lo, hi):
            ov = iv.overlap(lo, hi)
            if ov > best_ov:
                best, best_ov = iv, ov
        return best

    def __iter__(self) -> Iterator[Interval[T]]:
        def walk(n: _Node[T] | None) -> Iterator[Interval[T]]:
            if n is None:
                return
            yield from walk(n.left)
            yield Interval(n.start, n.end, n.payload)
            yield from walk(n.right)

        return walk(self._root)


# ---------------------------------------------------------------- sweep line
@dataclass(slots=True)
class Coverage:
    """Union-of-intervals accumulator (sweep line), used for speech coverage."""

    _events: list[tuple[float, int]] = field(default_factory=list)

    def add(self, start: float, end: float) -> None:
        if end > start:
            self._events.append((start, 1))
            self._events.append((end, -1))

    def merged(self) -> list[tuple[float, float]]:
        if not self._events:
            return []
        self._events.sort()
        out: list[tuple[float, float]] = []
        depth = 0
        cur_start = 0.0
        for t, delta in self._events:
            if depth == 0 and delta == 1:
                cur_start = t
            depth += delta
            if depth == 0:
                out.append((cur_start, t))
        return out

    def total(self) -> float:
        return sum(e - s for s, e in self.merged())


