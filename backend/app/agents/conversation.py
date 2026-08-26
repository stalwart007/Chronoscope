"""Turning a follow-up into a standalone question.

Each query is answered on its own, which makes the obvious second question
impossible: "when do they show the architecture diagram?" followed by "what did
they say right after that?" leaves the second query with no idea the first
happened. Retrieval sees "what did they say right after that", which matches
nothing in particular.

Resolution happens before planning, so everything downstream keeps working on a
single self-contained question. Three kinds of reference are handled, all
without a model, because the deterministic path is the default:

topical    "what else did they say about it" -> the subject of the last turn is
           substituted in from the vocabulary that actually matched.
temporal   "right after that" -> a time window anchored on the previous
           answer's citations, so retrieval is restricted rather than reworded.
ordinal    "the second one" -> the numbered hit from the previous turn becomes
           the anchor.

The resolved query and the reason for each substitution are both returned, so
the interface can show what it assumed rather than silently rewriting the
user's words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.types import Citation, TimeSpan

#: Words that point at something already said rather than naming it.
PRONOUNS = re.compile(
    r"\b(it|its|that|this|those|these|they|them|their|there|he|she|him|her|his|hers|same)\b", re.I
)
TEMPORAL = re.compile(
    r"\b(after|before|then|next|earlier|later|previously|following|preceding|since|until)\b", re.I
)
ORDINAL = re.compile(
    r"\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|last|final|previous|next)\b", re.I
)
CONTINUATION = re.compile(
    r"^\s*(and|also|what about|how about|why|and then|ok|okay|so)\b", re.I
)

ORDINAL_VALUES = {
    "first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3, "fifth": 4, "5th": 4, "last": -1, "final": -1,
}

#: How far around an anchor "after that" or "before that" should look.
WINDOW_AFTER = 90.0
WINDOW_BEFORE = 60.0

_WORD = re.compile(r"[a-z0-9][a-z0-9'-]{2,}")
_STOP = frozenset(
    """the a an and or but if then of to in on at for with by is are was were be been being it its this that
    these those as from into about over under what when where which who how why does did do said say show
    shows tell me you your their there here more most some any all can could would should will other else
    again same thing things one ones part parts""".split()
)


@dataclass(slots=True)
class Turn:
    """One completed exchange, kept as context for the next."""

    query: str
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    @property
    def span(self) -> TimeSpan | None:
        """The full extent of what the previous answer cited."""
        if not self.citations:
            return None
        return TimeSpan(
            start=min(c.start for c in self.citations),
            end=max(c.end for c in self.citations),
        )

    @property
    def focus(self) -> float | None:
        """The moment the previous answer was mostly about.

        "after that" means after the thing just described, which is the
        strongest citation rather than whichever happened to occur earliest.
        Citations arrive ordered by relevance.
        """
        if not self.citations:
            return None
        return self.citations[0].start


@dataclass(slots=True)
class Resolution:
    query: str
    original: str
    time_range: TimeSpan | None = None
    is_followup: bool = False
    notes: list[str] = field(default_factory=list)


def content_words(text: str) -> list[str]:
    return [w for w in _WORD.findall((text or "").lower()) if w not in _STOP]


#: Verbs and adverbs that frame a question without naming its subject. A query
#: built only from these is pointing at the previous turn.
SCAFFOLDING = frozenset(
    """say said says mention mentioned talk talked tell told speak spoke discuss discussed cover covered
    happen happened come came go goes went right just exactly specifically more else again about
    thing things one ones part parts bit next""".split()
)


#: Presentation filler. These carry a sentence without naming its subject, so
#: carrying them into the next question adds noise and dilutes the real terms.
_FILLER = frozenset(
    """let us look looking see seeing watch here now well okay like get got make made take took put
    want need think know going want lets going first second third next last also just really very
    actually basically simply quite bit lot really""".split()
)


def subject_words(text: str) -> list[str]:
    """The words in a sentence that name what it is about.

    Stricter than :func:`content_words`: a citation reading "Let us look at the
    revenue chart" is about revenue and charts, not about looking.
    """
    return [
        w
        for w in content_words(text)
        if w not in SCAFFOLDING and w not in _FILLER and len(w) > 2
    ]


def looks_like_followup(query: str, history: list[Turn]) -> bool:
    """Whether the question leans on something already established.

    Deliberately conservative. Substituting the previous subject into an
    unrelated question makes the answer worse, so "and what about the
    Kubernetes rollout" is treated as a new topic despite the "what about":
    it names its own subject, so it needs no antecedent. Only questions that
    are genuinely incomplete on their own get rewritten.
    """
    if not history:
        return False
    words = content_words(query)
    substantive = [w for w in words if w not in SCAFFOLDING]

    # A pointer word is the only reliable signal. An opener like "what about"
    # is not: "what about the revenue numbers" names its own subject and must
    # be answered on its own terms.
    if PRONOUNS.search(query):
        return True
    if ORDINAL.search(query) and len(substantive) <= 1:
        return True
    # Nothing to search for at all, e.g. "why?" or "and then?".
    return not substantive


def _anchor(history: list[Turn], query: str) -> tuple[float | None, str]:
    """Pick the moment a relative reference points at."""
    last = history[-1]
    match = ORDINAL.search(query)
    if match and last.citations:
        index = ORDINAL_VALUES.get(match.group(1).lower())
        if index is not None:
            ordered = sorted(last.citations, key=lambda c: c.start)
            try:
                cite = ordered[index]
                return cite.start, f'"{match.group(1)}" resolved to the citation at {cite.start:.0f}s'
            except IndexError:
                pass
    focus = last.focus
    if focus is not None:
        return focus, f"anchored on the previous answer at {focus:.0f}s"
    return None, ""


def _ordinal_window(history: list[Turn], query: str) -> tuple[TimeSpan | None, str]:
    """An ordinal reference points at one specific result, so scope to it."""
    match = ORDINAL.search(query)
    if not match or not history[-1].citations:
        return None, ""
    index = ORDINAL_VALUES.get(match.group(1).lower())
    if index is None:
        return None, ""
    ordered = sorted(history[-1].citations, key=lambda c: c.start)
    try:
        cite = ordered[index]
    except IndexError:
        return None, ""
    return (
        TimeSpan(start=max(0.0, cite.start - 5.0), end=cite.end + 30.0),
        f'"{match.group(1)}" resolved to the moment at {cite.start:.0f}s',
    )


def resolve(query: str, history: list[Turn]) -> Resolution:
    """Rewrite a follow-up into a question that stands on its own."""
    original = query.strip()
    if not looks_like_followup(original, history):
        return Resolution(query=original, original=original)

    notes: list[str] = []
    last = history[-1]
    time_range: TimeSpan | None = None

    ordinal_span, ordinal_note = _ordinal_window(history, original)
    if ordinal_span is not None:
        time_range = ordinal_span
        notes.append(ordinal_note)

    anchor, anchor_note = _anchor(history, original)
    relative = TEMPORAL.search(original)
    if relative and anchor is not None and time_range is None:
        word = relative.group(1).lower()
        if word in {"after", "then", "next", "later", "following", "since"}:
            time_range = TimeSpan(start=anchor, end=anchor + WINDOW_AFTER)
            notes.append(f"limited to the {WINDOW_AFTER:.0f}s after {anchor:.0f}s")
        elif word in {"before", "earlier", "previously", "preceding", "until"}:
            time_range = TimeSpan(start=max(0.0, anchor - WINDOW_BEFORE), end=anchor)
            notes.append(f"limited to the {WINDOW_BEFORE:.0f}s before {anchor:.0f}s")
        if anchor_note:
            notes.append(anchor_note)

    # Carry the previous subject forward. Terms come from what actually matched
    # last time, not from the wording of the previous question, so the
    # substitution reflects the recording rather than the user's guess at it.
    subject = [k for k in last.keywords if k not in content_words(original)][:4]
    if not subject:
        subject = [w for w in content_words(last.query) if w not in content_words(original)][:4]

    rewritten = original
    if subject:
        rewritten = f"{original} {' '.join(subject)}"
        notes.append(f"carried forward: {', '.join(subject)}")

    return Resolution(
        query=rewritten,
        original=original,
        time_range=time_range,
        is_followup=True,
        notes=notes,
    )


def summarise_history(history: list[Turn], *, limit: int = 3, budget: int = 900) -> str:
    """Recent turns, compactly, for a model prompt."""
    lines: list[str] = []
    used = 0
    for turn in history[-limit:]:
        entry = f"Q: {turn.query}\nA: {turn.answer.strip()[:300]}"
        if used + len(entry) > budget:
            break
        lines.append(entry)
        used += len(entry)
    return "\n\n".join(lines)
