"""BM25 sparse retrieval over chunk text.

Dense embeddings handle paraphrase well and rare literals poorly: a 384-d
vector does not reliably separate "sixty eight million" from "ninety one
million". An inverted index covers exactly those cases, so it joins fusion as a
peer channel.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field

_WORD = re.compile(r"[a-z0-9][a-z0-9'\-]*")
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60",
    "seventy": "70", "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
    "million": "1e6", "billion": "1e9",
}

K1 = 1.4
B = 0.72


def tokenize(text: str) -> list[str]:
    toks = _WORD.findall((text or "").lower())
    out: list[str] = []
    for t in toks:
        out.append(t)
        # Index spelled-out numerals under their digit form too, so "68" finds
        # "sixty eight" and vice versa.
        if t in _NUMBER_WORDS:
            out.append(_NUMBER_WORDS[t])
    return out


@dataclass(slots=True)
class _Doc:
    chunk_id: str
    video_id: str
    length: int
    start: float = 0.0
    end: float = 0.0
    #: The document's own vocabulary. Without it, deleting one document means
    #: scanning every posting list in the index. O(|V|) per delete, and |V|
    #: grows with the corpus, so re-indexing a library degrades quadratically.
    terms: tuple[str, ...] = ()


@dataclass
class BM25Index:
    postings: dict[str, dict[int, int]] = field(default_factory=lambda: defaultdict(dict))
    docs: list[_Doc] = field(default_factory=list)
    doc_of_chunk: dict[str, int] = field(default_factory=dict)
    total_len: int = 0
    _lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def n_docs(self) -> int:
        return sum(1 for d in self.docs if d.length >= 0)

    @property
    def avg_len(self) -> float:
        n = self.n_docs
        return self.total_len / n if n else 1.0

    def add(
        self, chunk_id: str, video_id: str, text: str, *, start: float = 0.0, end: float = 0.0
    ) -> None:
        with self._lock:
            if chunk_id in self.doc_of_chunk:
                self.remove(chunk_id)
            toks = tokenize(text)
            counts = Counter(toks)
            doc_id = len(self.docs)
            self.docs.append(_Doc(chunk_id, video_id, len(toks), start, end, tuple(counts)))
            self.doc_of_chunk[chunk_id] = doc_id
            self.total_len += len(toks)
            for term, tf in counts.items():
                self.postings[term][doc_id] = tf

    def remove(self, chunk_id: str) -> None:
        with self._lock:
            doc_id = self.doc_of_chunk.pop(chunk_id, None)
            if doc_id is None:
                return
            doc = self.docs[doc_id]
            self.total_len -= max(doc.length, 0)
            doc.length = -1  # tombstone
            for term in doc.terms:  # O(|doc|), not O(|vocabulary|)
                posting = self.postings.get(term)
                if posting is not None:
                    posting.pop(doc_id, None)
                    if not posting:
                        del self.postings[term]
            doc.terms = ()

    def remove_video(self, video_id: str) -> None:
        for d in list(self.docs):
            if d.video_id == video_id and d.length >= 0:
                self.remove(d.chunk_id)

    def has_video(self, video_id: str) -> bool:
        return any(d.video_id == video_id and d.length >= 0 for d in self.docs)

    def search(
        self,
        query: str,
        *,
        limit: int = 32,
        video_ids: list[str] | None = None,
        start: float | None = None,
        end: float | None = None,
    ) -> list[tuple[str, float]]:
        with self._lock:
            terms = tokenize(query)
            if not terms or not self.n_docs:
                return []
            n, avg = self.n_docs, self.avg_len
            allowed = set(video_ids) if video_ids else None
            scores: dict[int, float] = defaultdict(float)
            for term, qtf in Counter(terms).items():
                posting = self.postings.get(term)
                if not posting:
                    continue
                df = len(posting)
                idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                for doc_id, tf in posting.items():
                    doc = self.docs[doc_id]
                    if doc.length < 0 or (allowed and doc.video_id not in allowed):
                        continue
                    # Sparse retrieval must honour the same span filter as the
                    # dense channels, or fusion reintroduces chunks the user
                    # explicitly excluded.
                    if start is not None and doc.end and doc.end < start:
                        continue
                    if end is not None and doc.start > end:
                        continue
                    denom = tf + K1 * (1.0 - B + B * doc.length / avg)
                    scores[doc_id] += idf * (tf * (K1 + 1.0)) / denom * (1.0 + 0.15 * math.log(qtf + 1))
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:limit]
            return [(self.docs[d].chunk_id, round(s, 6)) for d, s in ranked]

    def stats(self) -> dict[str, float | int]:
        return {
            "documents": self.n_docs,
            "terms": len(self.postings),
            "avg_doc_len": round(self.avg_len, 2),
            "postings": sum(len(p) for p in self.postings.values()),
        }


index = BM25Index()
