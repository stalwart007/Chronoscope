"""In-process publish/subscribe with replay.

Progress is streamed to the browser over SSE, and clients subscribe after the
request that starts a job. Each topic keeps a bounded ring buffer so a
subscriber can replay from a cursor and then follow live. Per-subscriber queues
are bounded and drop the oldest event, so a slow consumer cannot stall a
producer.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from app.core.types import utcnow


@dataclass(slots=True)
class Event:
    seq: int
    topic: str
    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=lambda: utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "topic": self.topic, "kind": self.kind, "data": self.data, "ts": self.ts}


class _Subscriber:
    __slots__ = ("dropped", "queue", "topic")

    def __init__(self, topic: str, maxsize: int) -> None:
        self.topic = topic
        self.queue: asyncio.Queue[Event | None] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    def offer(self, ev: Event) -> None:
        try:
            self.queue.put_nowait(ev)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self.queue.get_nowait()
                self.dropped += 1
            with contextlib.suppress(asyncio.QueueFull):
                self.queue.put_nowait(ev)


class EventBus:
    def __init__(self, *, history: int = 512, queue_size: int = 256) -> None:
        self._history_size = history
        self._queue_size = queue_size
        self._buffers: dict[str, deque[Event]] = {}
        self._subs: dict[str, set[_Subscriber]] = {}
        self._seq = 0
        self._closed: set[str] = set()
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------- publishing
    def publish(self, topic: str, kind: str, **data: Any) -> Event:
        self._seq += 1
        ev = Event(seq=self._seq, topic=topic, kind=kind, data=data)
        buf = self._buffers.setdefault(topic, deque(maxlen=self._history_size))
        buf.append(ev)
        for sub in tuple(self._subs.get(topic, ())):
            sub.offer(ev)
        return ev

    def close_topic(self, topic: str) -> None:
        self._closed.add(topic)
        for sub in tuple(self._subs.get(topic, ())):
            with contextlib.suppress(asyncio.QueueFull):
                sub.queue.put_nowait(None)

    def is_closed(self, topic: str) -> bool:
        return topic in self._closed

    def history(self, topic: str, after: int = 0) -> list[Event]:
        return [e for e in self._buffers.get(topic, ()) if e.seq > after]

    # ------------------------------------------------------------ subscribing
    async def subscribe(self, topic: str, *, after: int = 0) -> AsyncIterator[Event]:
        sub = _Subscriber(topic, self._queue_size)
        async with self._lock:
            self._subs.setdefault(topic, set()).add(sub)
        try:
            for ev in self.history(topic, after):
                yield ev
                after = ev.seq
            if self.is_closed(topic):
                return
            while True:
                item = await sub.queue.get()
                if item is None:
                    return
                ev = item
                if ev.seq <= after:
                    continue
                after = ev.seq
                yield ev
        finally:
            async with self._lock:
                subs = self._subs.get(topic)
                if subs:
                    subs.discard(sub)
                    if not subs:
                        self._subs.pop(topic, None)

    def stats(self) -> dict[str, Any]:
        return {
            "topics": len(self._buffers),
            "subscribers": sum(len(s) for s in self._subs.values()),
            "events": self._seq,
            "closed": len(self._closed),
        }

    def drop_topic(self, topic: str) -> None:
        self._buffers.pop(topic, None)
        self._closed.discard(topic)


bus = EventBus()
