"""Background job runner: a priority queue with bounded workers.

Ingestion must not run inside the upload request, and a task per upload would
thrash the model cache. Jobs are queued shortest-first, so a short clip is not
stuck behind a long keynote, and the registry supports cancellation and status.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.core.types import JobStatus
from app.logging_conf import get_logger

log = get_logger(__name__)


@dataclass(order=True)
class _QueueItem:
    priority: float
    seq: int
    job_id: str = field(compare=False)
    fn: Callable[[], Awaitable[Any]] = field(compare=False)
    kind: str = field(compare=False, default="ingest")


@dataclass
class JobRecord:
    job_id: str
    kind: str
    status: JobStatus = JobStatus.PENDING
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str = ""
    task: asyncio.Task[Any] | None = field(default=None, repr=False)

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status.value,
            "queued_s": round((self.started_at or time.time()) - self.submitted_at, 2),
            "runtime_s": round((self.finished_at or time.time()) - self.started_at, 2)
            if self.started_at
            else 0.0,
            "error": self.error[:300],
        }


class JobRunner:
    def __init__(self, concurrency: int | None = None) -> None:
        self.concurrency = concurrency or settings.ingest_concurrency
        self._queue: asyncio.PriorityQueue[_QueueItem] = asyncio.PriorityQueue()
        self._jobs: dict[str, JobRecord] = {}
        self._workers: list[asyncio.Task[None]] = []
        self._seq = 0
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._workers = [asyncio.create_task(self._worker(i), name=f"worker-{i}") for i in range(self.concurrency)]
        log.info("job runner started with %d worker(s)", self.concurrency)

    async def stop(self, *, drain_timeout: float = 10.0) -> None:
        self._running = False
        for w in self._workers:
            w.cancel()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.gather(*self._workers, return_exceptions=True), timeout=drain_timeout)
        self._workers.clear()

    def submit(
        self, job_id: str, fn: Callable[[], Awaitable[Any]], *, priority: float = 100.0, kind: str = "ingest"
    ) -> JobRecord:
        # A resubmission supersedes whatever is in flight for the same id.
        # Without this the old run finishes after the new one is queued and
        # overwrites the record with stale status and artefacts.
        previous = self._jobs.get(job_id)
        if previous is not None and previous.status in {JobStatus.PENDING, JobStatus.RUNNING}:
            self.cancel(job_id)
        self._seq += 1
        record = JobRecord(job_id=job_id, kind=kind)
        self._jobs[job_id] = record
        self._queue.put_nowait(_QueueItem(priority=priority, seq=self._seq, job_id=job_id, fn=fn, kind=kind))
        return record

    async def _worker(self, index: int) -> None:
        while True:
            item = await self._queue.get()
            record = self._jobs.get(item.job_id)
            if record is None or record.status is JobStatus.CANCELLED:
                self._queue.task_done()
                continue
            record.status = JobStatus.RUNNING
            record.started_at = time.time()
            task: asyncio.Task[Any] = asyncio.ensure_future(item.fn())
            record.task = task
            try:
                await task
                record.status = JobStatus.COMPLETED
            except asyncio.CancelledError:
                record.status = JobStatus.CANCELLED
                log.info("job %s cancelled", item.job_id)
            except Exception as exc:
                record.status = JobStatus.FAILED
                record.error = f"{type(exc).__name__}: {exc}"
                log.exception("job %s failed", item.job_id)
            finally:
                record.finished_at = time.time()
                record.task = None
                self._queue.task_done()

    def cancel(self, job_id: str) -> bool:
        record = self._jobs.get(job_id)
        if record is None:
            return False
        if record.task is not None and not record.task.done():
            record.task.cancel()
            return True
        if record.status is JobStatus.PENDING:
            record.status = JobStatus.CANCELLED
            return True
        return False

    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def stats(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        for r in self._jobs.values():
            by_status[r.status.value] = by_status.get(r.status.value, 0) + 1
        return {
            "workers": len(self._workers),
            "queued": self._queue.qsize(),
            "jobs": len(self._jobs),
            "by_status": by_status,
            "running": [r.snapshot() for r in self._jobs.values() if r.status is JobStatus.RUNNING],
        }

    async def wait_idle(self, timeout: float = 300.0) -> bool:
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except TimeoutError:
            return False
        for _ in range(int(timeout * 10)):
            if not any(r.status is JobStatus.RUNNING for r in self._jobs.values()):
                return True
            await asyncio.sleep(0.1)
        return False


runner = JobRunner()


def size_priority(size_bytes: int) -> float:
    """Shortest-job-first: smaller uploads jump the queue."""
    return round(size_bytes / (1024 * 1024), 3)
