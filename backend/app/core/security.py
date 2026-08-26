"""Authentication, rate limiting, input validation and output redaction.

Uploaded media, request identifiers and query text are all treated as
untrusted. Containers are sniffed by magic bytes and probed before indexing,
identifiers are pattern-checked before reaching the filesystem, and anything
credential-shaped is stripped from responses and logs.

Authentication is optional: setting ``CS_API_KEY`` enables constant-time bearer
checks. It is off by default because the common deployment is localhost.
"""

from __future__ import annotations

import hmac
import re
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from fastapi import Header, Request
from fastapi import Path as PathParam

from app.config import settings
from app.core.errors import BadRequest, ChronoscopeError, UnsupportedMedia
from app.logging_conf import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------- identifiers
#: Ids are blake2b digests rendered as 32 lowercase hex characters. Anything
#: else is rejected before it can reach a path join or a database lookup.
ID_PATTERN = r"^[0-9a-f]{32}$"
ID_RE = re.compile(ID_PATTERN)

VideoId = Annotated[str, PathParam(pattern=ID_PATTERN, description="32-character hex video id")]


def is_valid_id(value: str) -> bool:
    return bool(ID_RE.match(value or ""))


def safe_join(root: Path, *parts: str) -> Path:
    """Join under ``root`` and verify the result stayed inside it.

    ``Path`` resolves ``..`` segments, and an absolute component discards
    everything before it, so the resolved path is re-checked against the root.

    Backslashes and NUL bytes are rejected rather than resolved, since a
    backslash is an ordinary filename character on POSIX and a separator on
    Windows.
    """
    for part in parts:
        if "\\" in part or "\x00" in part:
            raise BadRequest("path contains illegal characters", code="path_traversal")
    root = root.resolve()
    candidate = root.joinpath(*parts).resolve()
    if candidate != root and root not in candidate.parents:
        raise BadRequest("path escapes its root directory", code="path_traversal")
    return candidate


class Unauthorized(ChronoscopeError):
    status_code = 401
    code = "unauthorized"


class RateLimited(ChronoscopeError):
    status_code = 429
    code = "rate_limited"


# ------------------------------------------------------------------- secrets
_SECRET_PATTERNS = (
    re.compile(r"\b(sk-or-v1-|sk-|gsk_|hf_|ghp_|xox[baprs]-)[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\b(api[_-]?key|authorization|bearer|token|password|secret)\b\s*[:=]\s*\S+"),
)


def redact(text: str, *, limit: int = 400) -> str:
    """Strip anything credential-shaped before it reaches a log or a client."""
    if not text:
        return ""
    out = text[:limit]
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub("[redacted]", out)
    return out


def client_key(request: Request) -> str:
    """Best-effort client identity for rate limiting.

    ``X-Forwarded-For`` is honoured only when the app is explicitly told it is
    behind a proxy; otherwise a client could spoof the header and mint a fresh
    bucket for every request.
    """
    if settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()[:64]
    return (request.client.host if request.client else "unknown")[:64]


# -------------------------------------------------------------- rate limiting
@dataclass(slots=True)
class _Bucket:
    tokens: float
    updated: float


@dataclass
class RateLimiter:
    """Token bucket per client, with an LRU cap so memory stays bounded.

    A token bucket (rather than a fixed window) lets a user burst, opening a
    page fires several requests at once, while still bounding the sustained
    rate. The LRU cap matters: an unbounded dict keyed by client IP is itself a
    memory-exhaustion vector.
    """

    rate: float
    burst: float
    max_clients: int = 8192
    _buckets: OrderedDict[str, _Bucket] = field(default_factory=OrderedDict, init=False)

    def check(self, key: str, cost: float = 1.0) -> tuple[bool, float]:
        now = time.monotonic()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=self.burst, updated=now)
            self._buckets[key] = bucket
            while len(self._buckets) > self.max_clients:
                self._buckets.popitem(last=False)
        else:
            self._buckets.move_to_end(key)
            bucket.tokens = min(self.burst, bucket.tokens + (now - bucket.updated) * self.rate)
            bucket.updated = now
        if bucket.tokens >= cost:
            bucket.tokens -= cost
            return True, 0.0
        return False, round((cost - bucket.tokens) / max(self.rate, 1e-6), 2)

    def reset(self) -> None:
        self._buckets.clear()


#: Separate budgets per class of work. An expensive ingest should not share a
#: bucket with a cheap poll of the library.
LIMITERS: dict[str, RateLimiter] = {
    "upload": RateLimiter(rate=settings.rl_upload_per_min / 60.0, burst=max(5.0, settings.rl_upload_per_min / 4)),
    "query": RateLimiter(rate=settings.rl_query_per_min / 60.0, burst=max(4.0, settings.rl_query_per_min / 4)),
    "stream": RateLimiter(rate=settings.rl_stream_per_min / 60.0, burst=max(4.0, settings.rl_stream_per_min / 4)),
    "default": RateLimiter(rate=settings.rl_default_per_min / 60.0, burst=max(20.0, settings.rl_default_per_min / 3)),
}


def classify(path: str, method: str) -> str:
    if path.startswith("/api/videos") and method == "POST" and path.count("/") == 2:
        return "upload"
    if path.endswith(("/events", "/stream")):
        return "stream"
    if path.startswith(("/api/query", "/api/search")):
        return "query"
    return "default"


#: Runtime toggle seeded from configuration. Kept separate from the frozen
#: settings object so limits can be suspended without a restart, useful during
#: a bulk import, and required for deterministic tests.
_rate_limiting_enabled = settings.rate_limit_enabled


def set_rate_limiting(enabled: bool) -> None:
    global _rate_limiting_enabled
    _rate_limiting_enabled = enabled


def enforce_rate_limit(request: Request) -> None:
    if not _rate_limiting_enabled:
        return
    bucket = classify(request.url.path, request.method)
    ok, retry_after = LIMITERS[bucket].check(client_key(request))
    if not ok:
        raise RateLimited(
            f"too many {bucket} requests, retry in {retry_after:g}s",
            detail={"bucket": bucket, "retry_after": retry_after},
        )


def reset_rate_limits() -> None:
    for limiter in LIMITERS.values():
        limiter.reset()


# ------------------------------------------------------------ authentication
#: Liveness probes must work without credentials. Container healthchecks, load
#: balancers and orchestrators cannot present a bearer token, so gating this
#: path would mark a perfectly healthy process as failed. It exposes nothing, #: the detailed `/api/system/health` (models, providers, capabilities) stays
#: authenticated.
PUBLIC_PATHS = frozenset({"/healthz"})


def require_api_key(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """Constant-time bearer check. A no-op unless ``CS_API_KEY`` is set."""
    expected = settings.api_key
    if not expected or request.url.path in PUBLIC_PATHS or request.method == "OPTIONS":
        return
    presented = ""
    if authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()
    elif x_api_key:
        presented = x_api_key.strip()
    if not presented or not hmac.compare_digest(presented, expected):
        raise Unauthorized("a valid API key is required")


# ---------------------------------------------------------------- media type
#: (offset, magic bytes) -> container family. Enough to reject a renamed
#: executable or archive before it is handed to a decoder.
_MAGIC: tuple[tuple[int, bytes, str], ...] = (
    (4, b"ftyp", "mp4"),          # ISO-BMFF: mp4, m4v, mov
    (0, b"\x1a\x45\xdf\xa3", "matroska"),  # mkv, webm
    (0, b"RIFF", "avi"),
    (0, b"OggS", "ogg"),
    (0, b"FLV\x01", "flv"),
    (0, b"\x30\x26\xb2\x75", "asf"),       # wmv
    (0, b"\x00\x00\x01\xba", "mpeg-ps"),
    (0, b"\x00\x00\x01\xb3", "mpeg-vs"),
    (0, b"\x47", "mpeg-ts"),
)

_FORBIDDEN_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"PK\x03\x04", "zip archive"),
    (b"\x7fELF", "ELF executable"),
    (b"MZ", "Windows executable"),
    (b"#!", "script"),
    (b"\xca\xfe\xba\xbe", "Mach-O / Java class"),
    (b"\xcf\xfa\xed\xfe", "Mach-O executable"),
    (b"%PDF", "PDF document"),
    (b"<?xml", "XML document"),
    (b"<", "markup document"),
)


def sniff_container(header: bytes) -> str | None:
    """Identify a media container from its first bytes, or ``None``."""
    for offset, magic, name in _MAGIC:
        if header[offset : offset + len(magic)] == magic:
            return name
    return None


def validate_media_header(header: bytes, *, filename: str) -> str:
    """Reject anything that is provably not a media container.

    Extension checks are trivially bypassed; this looks at the actual bytes.
    Explicitly-dangerous formats produce a precise error, and unknown formats
    are rejected too, the decoder is the largest attack surface in the
    system, so nothing unrecognised reaches it.
    """
    if len(header) < 12:
        raise UnsupportedMedia("file is too small to be a video", detail={"filename": filename})
    for magic, label in _FORBIDDEN_MAGIC:
        if header.startswith(magic):
            raise UnsupportedMedia(
                f"this is a {label}, not a video", detail={"filename": filename, "detected": label}
            )
    container = sniff_container(header)
    if container is None:
        raise UnsupportedMedia(
            "the file's contents are not a recognised video container",
            detail={"filename": filename, "hint": "re-encode with `ffmpeg -i input output.mp4`"},
        )
    return container


_SUBTITLE_HINTS = ("-->", "WEBVTT", "{", "[")


def validate_subtitle(data: bytes, *, filename: str) -> None:
    if len(data) > settings.max_transcript_kb * 1024:
        raise BadRequest(f"transcript exceeds {settings.max_transcript_kb} KB", detail={"filename": filename})
    for magic, label in _FORBIDDEN_MAGIC[:6]:  # binary formats only; markup is fine in captions
        if data.startswith(magic):
            raise UnsupportedMedia(f"transcript looks like a {label}", detail={"filename": filename})
    try:
        text = data[:4096].decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise BadRequest("transcript must be UTF-8 text", detail={"filename": filename}) from exc
    if not any(hint in text for hint in _SUBTITLE_HINTS):
        raise BadRequest(
            "transcript does not look like SRT, WebVTT or JSON",
            detail={"filename": filename},
        )


def sanitize_title(raw: str, *, limit: int = 160) -> str:
    """Titles are rendered in a browser; strip control characters and markup."""
    cleaned = re.sub(r"[\x00-\x1f\x7f<>]", " ", raw or "").strip()
    return re.sub(r"\s+", " ", cleaned)[:limit]


# -------------------------------------------------------------------- quotas
def disk_usage(path: Path) -> tuple[int, int]:
    """``(used_bytes, free_bytes)`` for the data volume."""
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                continue
    free = shutil.disk_usage(path).free
    return total, free


def check_quota(incoming_bytes: int) -> None:
    """Refuse an upload that would breach the configured storage budget.

    Derived artefacts (frames, audio, index) roughly match the source size, so
    the projection reserves twice the incoming bytes.
    """
    quota = settings.storage_quota_gb * 1024**3
    if quota <= 0:
        return
    used, free = disk_usage(settings.data_dir)
    projected = used + incoming_bytes * 2
    if projected > quota:
        raise BadRequest(
            f"storage quota reached ({used / 1024**3:.1f} GB of {settings.storage_quota_gb} GB used)",
            code="quota_exceeded",
            detail={"used_bytes": used, "quota_bytes": quota},
        )
    if free < incoming_bytes * 2:
        raise BadRequest("not enough free disk space for this upload", code="disk_full")


def safe_public_error(exc: BaseException) -> dict[str, Any]:
    """Client-facing shape for an unhandled error.

    In production the message is generic: exception text routinely contains
    paths, SQL fragments and occasionally credentials. Developers keep the
    detail because losing it makes local debugging miserable.
    """
    if settings.env == "prod":
        return {"code": "internal_error", "message": "An internal error occurred.", "detail": None}
    return {"code": "internal_error", "message": redact(str(exc)), "detail": type(exc).__name__}
