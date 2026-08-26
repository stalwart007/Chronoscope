"""Numeric extraction from speech and slide text.

Speakers say "forty two million"; slides render "$42M". Both must become a
number before any arithmetic, and the label has to travel with the value. The
word-number parser is a state machine over the units/teens/tens/scales grammar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60,
        "seventy": 70, "eighty": 80, "ninety": 90}
SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000,
          "trillion": 1_000_000_000_000}
SUFFIX = {"k": 1e3, "m": 1e6, "bn": 1e9, "b": 1e9, "t": 1e12}

ORDINAL_LABEL = re.compile(
    r"\b(first|second|third|fourth|1st|2nd|3rd|4th|q[1-4]|quarter\s+(?:one|two|three|four|[1-4]))\b", re.I
)
_NUM_TOKEN = re.compile(r"[a-z]+|\d+(?:[.,]\d+)?|%|\$", re.I)
_DIGIT = re.compile(
    r"(?P<currency>[$€£])?\s*(?P<num>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*"
    r"(?P<suffix>k|m|bn|b|t|%|percent|million|billion|thousand|trillion)?",
    re.I,
)


@dataclass(slots=True)
class NumericFact:
    value: float
    raw: str
    label: str = ""
    unit: str = ""
    position: int = 0
    source: str = "transcript"
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value, "raw": self.raw, "label": self.label,
            "unit": self.unit, "source": self.source, "context": self.context[:160],
        }


def words_to_number(tokens: list[str]) -> float | None:
    """Parse a spelled-out cardinal. Returns ``None`` if nothing parses."""
    total, current, seen = 0.0, 0.0, False
    for tok in tokens:
        t = tok.lower()
        if t in UNITS:
            current += UNITS[t]
            seen = True
        elif t in TENS:
            current += TENS[t]
            seen = True
        elif t == "hundred":
            current = max(current, 1.0) * 100
            seen = True
        elif t in SCALES:
            total += max(current, 1.0) * SCALES[t]
            current = 0.0
            seen = True
        elif t in {"and", "-"}:
            continue
        else:
            break
    return (total + current) if seen else None


def extract_numbers(text: str, *, source: str = "transcript") -> list[NumericFact]:
    facts: list[NumericFact] = []
    if not text:
        return facts

    # --- digit forms: "$42M", "68 million", "23.5%", "1,200"
    for m in _DIGIT.finditer(text):
        raw_num = m.group("num").replace(",", "")
        try:
            value: float = float(raw_num)
        except ValueError:
            continue
        suffix = (m.group("suffix") or "").lower()
        unit = m.group("currency") or ""
        if suffix in {"%", "percent"}:
            unit = "%"
        elif suffix in SUFFIX:
            value *= SUFFIX[suffix]
        elif suffix in SCALES:
            value *= SCALES[suffix]
        if m.start("num") > 0 and text[m.start("num") - 1].lower() == "q":
            continue  # the "1" in "Q1" is a label, not a measurement
        lo, hi = max(0, m.start() - 60), min(len(text), m.end() + 60)
        facts.append(
            NumericFact(
                value=value, raw=m.group(0).strip(), unit=unit, position=m.start(),
                source=source, context=text[lo:hi],
                label=_nearest_label(text, m.start()),
            )
        )

    # --- word forms: "forty two million"
    tokens = [(t.group(0), t.start()) for t in _NUM_TOKEN.finditer(text)]
    i = 0
    while i < len(tokens):
        tok = tokens[i][0].lower()
        if tok in UNITS or tok in TENS:
            j = i
            words: list[str] = []
            while j < len(tokens) and (
                tokens[j][0].lower() in UNITS or tokens[j][0].lower() in TENS
                or tokens[j][0].lower() in SCALES or tokens[j][0].lower() == "and"
            ):
                words.append(tokens[j][0])
                j += 1
            while words and words[-1].lower() == "and":
                words.pop()
                j -= 1
            parsed = words_to_number(words)
            if parsed is not None and parsed > 0 and len(words) >= 1:
                start = tokens[i][1]
                lo, hi = max(0, start - 60), min(len(text), start + 80)
                facts.append(
                    NumericFact(
                        value=parsed, raw=" ".join(words), position=start, source=source,
                        context=text[lo:hi], label=_nearest_label(text, start),
                        unit="$" if re.search(r"\bdollars?\b", text[start : start + 90], re.I) else "",
                    )
                )
            i = max(j, i + 1)
        else:
            i += 1

    facts.sort(key=lambda f: f.position)
    facts = _dedupe(facts)
    assign_labels(text, facts)
    return facts


def _nearest_label(text: str, pos: int) -> str:
    """Provisional label, refined globally by :func:`assign_labels`."""
    best, best_d = "", float("inf")
    for m in ORDINAL_LABEL.finditer(text):
        raw = abs(m.start() - pos)
        if raw >= 90:
            continue
        d = raw * (0.7 if m.start() >= pos else 1.0)
        if d < best_d:
            best, best_d = m.group(0).lower(), d
    return _canonical_label(best)


def assign_labels(text: str, facts: list[NumericFact], *, max_distance: float = 110.0) -> None:
    """Match labels to numbers with order-preserving alignment.

    Nearest-label assignment fails on alternating phrasing such as "the third
    quarter came in at 68 ... the fourth quarter reached 91", where the label
    after 68 is closer than the one before it but belongs to 91.

    Speech is monotone: labels and their values appear in the same relative
    order. That makes this sequence alignment, solved by the Needleman-Wunsch
    recurrence:

    ``dp[i][j] = min(dp[i-1][j] + skip, dp[i][j-1] + skip, dp[i-1][j-1] + d(i,j))``

    ``d`` is positional distance, weighted so labels following their value
    count as closer, and ``skip`` prices leaving one side unmatched. O(n*m).
    """
    labels = [(m.start(), _canonical_label(m.group(0).lower())) for m in ORDINAL_LABEL.finditer(text)]
    for f in facts:
        f.label = ""
    if not labels or not facts:
        return
    n, m = len(labels), len(facts)
    skip = max_distance * 0.85
    INF = float("inf")

    def d(i: int, j: int) -> float:
        raw = abs(labels[i][0] - facts[j].position)
        if raw > max_distance:
            return INF
        return raw * (0.7 if labels[i][0] >= facts[j].position else 1.0)

    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]  # 1=match 2=skip-label 3=skip-fact
    dp[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i and dp[i - 1][j] + skip < dp[i][j]:
                dp[i][j], back[i][j] = dp[i - 1][j] + skip, 2
            if j and dp[i][j - 1] + skip < dp[i][j]:
                dp[i][j], back[i][j] = dp[i][j - 1] + skip, 3
            if i and j:
                cost = d(i - 1, j - 1)
                if cost < INF and dp[i - 1][j - 1] + cost < dp[i][j]:
                    dp[i][j], back[i][j] = dp[i - 1][j - 1] + cost, 1
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move == 1:
            facts[j - 1].label = labels[i - 1][1]
            i, j = i - 1, j - 1
        elif move == 2:
            i -= 1
        elif move == 3:
            j -= 1
        else:
            break


def _canonical_label(label: str) -> str:
    if not label:
        return ""
    key = label.lower().replace("quarter ", "q").strip()
    mapping = {
        "first": "Q1", "1st": "Q1", "q1": "Q1", "qone": "Q1",
        "second": "Q2", "2nd": "Q2", "q2": "Q2", "qtwo": "Q2",
        "third": "Q3", "3rd": "Q3", "q3": "Q3", "qthree": "Q3",
        "fourth": "Q4", "4th": "Q4", "q4": "Q4", "qfour": "Q4",
    }
    return mapping.get(key, label.upper() if len(label) <= 3 else label.title())


def _dedupe(facts: list[NumericFact]) -> list[NumericFact]:
    """Drop the word-form duplicate when a digit form covers the same span."""
    out: list[NumericFact] = []
    for f in facts:
        if any(abs(f.position - g.position) < 6 and abs(f.value - g.value) < 1e-6 for g in out):
            continue
        out.append(f)
    return out


@dataclass(slots=True)
class Series:
    labels: list[str] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    unit: str = ""

    @property
    def ok(self) -> bool:
        return len(self.values) >= 2

    def to_dict(self) -> dict[str, Any]:
        return {"labels": self.labels, "values": self.values, "unit": self.unit}


def build_series(facts: list[NumericFact], *, min_len: int = 2) -> Series:
    """Assemble a labelled series from facts that share a unit and scale.

    Values within one order of magnitude of the median are kept, that filters
    stray numbers ("two weeks early", "all three regions") out of a revenue
    series without needing a model.
    """
    labelled = [f for f in facts if f.label]
    pool = labelled if len(labelled) >= min_len else facts
    if len(pool) < min_len:
        return Series()
    values = sorted(f.value for f in pool)
    median = values[len(values) // 2]
    lo, hi = median / 12.0, median * 12.0
    kept = [f for f in pool if lo <= f.value <= hi]
    if len(kept) < min_len:
        return Series()
    seen: dict[str, NumericFact] = {}
    for f in kept:
        key = f.label or f"#{len(seen) + 1}"
        if key not in seen:
            seen[key] = f
    ordered = sorted(seen.items(), key=lambda kv: kv[1].position)
    return Series(
        labels=[k for k, _ in ordered],
        values=[v.value for _, v in ordered],
        unit=next((f.unit for f in kept if f.unit), ""),
    )
