"""Fetching media from a URL, with SSRF protection.

A server that retrieves arbitrary URLs on request is a confused deputy: it sits
inside the network perimeter and will happily read whatever the caller names.
The classic targets are cloud metadata endpoints (169.254.169.254), loopback
services, and private-range hosts that are unreachable from outside.

The defences here, in order:

* Scheme allowlist. Only http and https, so file://, gopher:// and friends
  cannot be reached at all.
* Address validation. The hostname is resolved and *every* returned address is
  checked against loopback, private, link-local, multicast and reserved ranges.
  Checking only the first address leaves an obvious bypass.
* Redirects are followed manually, and each hop is validated again. A redirect
  to an internal address is the most common way naive checks are defeated.
* Response limits. Size and wall-clock caps, plus the same magic-byte check the
  upload path uses, applied to the first chunk before the rest is written.

Residual risk: DNS rebinding. The name is resolved for validation and again by
the connection, so a record with a very low TTL could return a public address
to the check and a private one to the connect. Closing that fully means pinning
the connection to the validated address, which conflicts with TLS SNI and
certificate verification. For a self-hosted tool behind an API key the exposure
is small, but it is a real gap and the deployment notes say so.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from app.config import settings
from app.core.errors import BadRequest, UnsupportedMedia
from app.core.security import validate_media_header
from app.logging_conf import get_logger

log = get_logger(__name__)

ALLOWED_SCHEMES = {"http", "https"}
MAX_REDIRECTS = 4

#: Hosts that commonly expose credentials or internal services.
BLOCKED_NETWORKS = [
    ipaddress.ip_network(n)
    for n in (
        "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16",
        "172.16.0.0/12", "192.0.0.0/24", "192.168.0.0/16", "198.18.0.0/15",
        "224.0.0.0/4", "240.0.0.0/4", "255.255.255.255/32",
        "::1/128", "fc00::/7", "fe80::/10", "ff00::/8", "::/128",
    )
]

#: Pages, not media. Downloading these yields HTML, so the failure is named
#: rather than left to surface as "not a recognised video container".
EXTRACTOR_HOSTS = (
    "youtube.com", "youtu.be", "vimeo.com", "dailymotion.com", "twitch.tv",
    "tiktok.com", "instagram.com", "facebook.com", "x.com", "twitter.com",
)


@dataclass(slots=True)
class FetchResult:
    path: Path
    size: int
    filename: str
    final_url: str


def _is_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return any(ip in net for net in BLOCKED_NETWORKS)


def validate_url(raw: str) -> str:
    """Check scheme and destination address. Returns the normalised URL."""
    raw = (raw or "").strip()
    if not raw:
        raise BadRequest("no URL supplied")
    if len(raw) > 2048:
        raise BadRequest("URL is too long")

    parsed = urlparse(raw)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise BadRequest(
            f"unsupported scheme {parsed.scheme or '(none)'!r}; only http and https are allowed",
            code="url_scheme",
        )
    host = parsed.hostname
    if not host:
        raise BadRequest("URL has no host")

    bare = host.lower().removeprefix("www.")
    if any(bare == h or bare.endswith(f".{h}") for h in EXTRACTOR_HOSTS):
        raise BadRequest(
            f"{bare} serves a web page, not a media file. Download the video first, "
            "then upload it, or paste a direct link to the media.",
            code="needs_extractor",
        )

    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise BadRequest(f"cannot resolve {host}", code="dns_failure") from exc

    addresses = {ipaddress.ip_address(info[4][0]) for info in infos}
    if not addresses:
        raise BadRequest(f"cannot resolve {host}", code="dns_failure")
    # Every address must be acceptable: one public and one private record would
    # otherwise let the connection pick the private one.
    for ip in addresses:
        if _is_blocked(ip):
            raise BadRequest(
                f"{host} resolves to a private or reserved address, which is not permitted",
                code="blocked_address",
            )
    return raw


def filename_from(url: str, content_type: str | None) -> str:
    name = Path(unquote(urlparse(url).path)).name
    if name and Path(name).suffix:
        return name[:255]
    suffix = {
        "video/mp4": ".mp4", "video/quicktime": ".mov", "video/x-matroska": ".mkv",
        "video/webm": ".webm", "video/x-msvideo": ".avi",
    }.get((content_type or "").split(";")[0].strip().lower(), ".mp4")
    return f"download{suffix}"


async def fetch_media(url: str, destination: Path, *, max_bytes: int, timeout: float = 60.0) -> FetchResult:
    """Stream a media file to ``destination``, validating every redirect hop."""
    current = validate_url(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0

    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=httpx.Timeout(timeout, connect=10.0),
        headers={"User-Agent": "Chronoscope/1.0", "Accept": "video/*,application/octet-stream;q=0.9,*/*;q=0.5"},
    ) as client:
        for hop in range(MAX_REDIRECTS + 1):
            async with client.stream("GET", current) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise BadRequest("redirect without a destination", code="bad_redirect")
                    current = validate_url(str(response.url.join(location)))
                    if hop == MAX_REDIRECTS:
                        raise BadRequest("too many redirects", code="too_many_redirects")
                    continue

                if response.status_code >= 400:
                    raise BadRequest(
                        f"the server returned {response.status_code} for that URL",
                        code="fetch_failed",
                    )

                declared = response.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > max_bytes:
                    raise BadRequest(
                        f"the file is {int(declared) / 1024**2:.0f} MB, over the "
                        f"{max_bytes / 1024**2:.0f} MB limit",
                        code="too_large",
                    )

                first = True
                try:
                    with destination.open("wb") as out:
                        async for chunk in response.aiter_bytes(1 << 20):
                            if first:
                                # Same check as the upload path: prove it is media
                                # before writing the rest of the body to disk.
                                validate_media_header(chunk[:64], filename=filename_from(current, response.headers.get("content-type")))
                                first = False
                            size += len(chunk)
                            if size > max_bytes:
                                raise BadRequest(
                                    f"the download exceeded the {max_bytes / 1024**2:.0f} MB limit",
                                    code="too_large",
                                )
                            out.write(chunk)
                except BaseException:
                    destination.unlink(missing_ok=True)
                    raise

                if size == 0:
                    destination.unlink(missing_ok=True)
                    raise UnsupportedMedia("the URL returned an empty response")

                log.info("fetched %.1f MB from %s", size / 1e6, urlparse(current).netloc)
                return FetchResult(
                    path=destination,
                    size=size,
                    filename=filename_from(current, response.headers.get("content-type")),
                    final_url=current,
                )

    raise BadRequest("too many redirects", code="too_many_redirects")


def max_download_bytes() -> int:
    return settings.max_upload_mb * 1024 * 1024
