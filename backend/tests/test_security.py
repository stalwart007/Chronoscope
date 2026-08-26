"""Security tests, each asserts an attack is *blocked*, not merely handled."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.errors import ChronoscopeError
from app.core.security import (
    RateLimiter,
    is_valid_id,
    redact,
    safe_join,
    sanitize_title,
    validate_media_header,
    validate_subtitle,
)


class TestInputValidation:
    @pytest.mark.parametrize(
        ("blob", "why"),
        [
            (b"PK\x03\x04" + b"\x00" * 40, "zip archive"),
            (b"\x7fELF\x02\x01\x01" + b"\x00" * 40, "ELF binary"),
            (b"MZ\x90\x00" + b"\x00" * 40, "Windows PE"),
            (b"#!/bin/sh\nrm -rf /\n" + b" " * 40, "shell script"),
            (b"<html><script>alert(1)</script></html>", "HTML"),
            (b"%PDF-1.7\n" + b"\x00" * 40, "PDF"),
            (b"\xde\xad\xbe\xef" * 12, "unrecognised bytes"),
            (b"tiny", "truncated file"),
        ],
    )
    def test_non_video_uploads_are_rejected(self, blob: bytes, why: str) -> None:
        with pytest.raises(ChronoscopeError):
            validate_media_header(blob, filename=f"payload.mp4  ({why})")

    def test_real_container_is_accepted(self, sample_video: Path) -> None:
        assert validate_media_header(sample_video.read_bytes()[:64], filename="ok.mp4") == "mp4"

    @pytest.mark.parametrize(
        "attack",
        ["../../etc", "../..", "/etc/passwd", "a/../../../../root", "..\\..\\windows", "a\x00b"],
    )
    def test_path_traversal_is_blocked(self, attack: str, tmp_path: Path) -> None:
        with pytest.raises(ChronoscopeError):
            safe_join(tmp_path, attack)

    def test_safe_join_allows_legitimate_children(self, tmp_path: Path) -> None:
        assert safe_join(tmp_path, "a" * 32, "frame.jpg").name == "frame.jpg"

    @pytest.mark.parametrize(
        ("value", "ok"),
        [("a" * 32, True), ("0123456789abcdef" * 2, True), ("A" * 32, False),
         ("../../etc", False), ("", False), ("a" * 31, False), ("a" * 33, False)],
    )
    def test_id_pattern(self, value: str, ok: bool) -> None:
        assert is_valid_id(value) is ok

    def test_titles_are_stripped_of_markup_and_control_chars(self) -> None:
        cleaned = sanitize_title("<script>alert(1)</script>\x00\x07 my talk")
        assert "<" not in cleaned and "\x00" not in cleaned
        assert "my talk" in cleaned

    def test_subtitle_validation(self) -> None:
        validate_subtitle(b"1\n00:00:01,000 --> 00:00:02,000\nhi\n", filename="a.srt")
        with pytest.raises(ChronoscopeError):
            validate_subtitle(b"\x7fELF" + b"\x00" * 40, filename="a.srt")
        with pytest.raises(ChronoscopeError):
            validate_subtitle(b"just some prose with no cues", filename="a.srt")


class TestSecretHandling:
    @pytest.mark.parametrize(
        "leak",
        [
            "connection failed for sk-or-v1-abcdef1234567890",
            "Authorization: Bearer gsk_ABCDEFGH12345678",
            "api_key=hf_QQQQQQQQQQQQQQQQ failed",
        ],
    )
    def test_credentials_never_survive_redaction(self, leak: str) -> None:
        cleaned = redact(leak)
        assert "[redacted]" in cleaned
        for token in ("sk-or-v1-", "gsk_ABCDEF", "hf_QQQQ"):
            assert token not in cleaned


class TestApiKey:
    """`CS_API_KEY` is optional, but when set it must actually gate access."""

    def _patch_key(self, monkeypatch, key: str | None) -> None:
        from types import SimpleNamespace

        import app.core.security as sec

        monkeypatch.setattr(sec, "settings", SimpleNamespace(api_key=key))

    @staticmethod
    def _request(path: str = "/api/videos", method: str = "GET"):  # type: ignore[no-untyped-def]
        """Minimal stand-in for the Request the dependency inspects."""
        from types import SimpleNamespace

        return SimpleNamespace(url=SimpleNamespace(path=path), method=method)

    def test_open_when_unset(self, monkeypatch) -> None:
        from app.core.security import require_api_key

        self._patch_key(monkeypatch, None)
        require_api_key(self._request(), authorization=None, x_api_key=None)  # must not raise

    @pytest.mark.parametrize(
        ("auth", "key"),
        [(None, None), ("Bearer wrong", None), (None, "wrong"), ("Basic c2VjcmV0", None), ("Bearer ", None)],
    )
    def test_rejects_missing_or_wrong_credentials(self, monkeypatch, auth, key) -> None:
        from app.core.security import require_api_key

        self._patch_key(monkeypatch, "s3cret-value")
        with pytest.raises(ChronoscopeError) as err:
            require_api_key(self._request(), authorization=auth, x_api_key=key)
        assert err.value.status_code == 401

    @pytest.mark.parametrize("header", ["bearer", "Bearer", "BEARER"])
    def test_accepts_valid_bearer_case_insensitively(self, monkeypatch, header) -> None:
        from app.core.security import require_api_key

        self._patch_key(monkeypatch, "s3cret-value")
        require_api_key(self._request(), authorization=f"{header} s3cret-value", x_api_key=None)

    def test_accepts_x_api_key(self, monkeypatch) -> None:
        from app.core.security import require_api_key

        self._patch_key(monkeypatch, "s3cret-value")
        require_api_key(self._request(), authorization=None, x_api_key="s3cret-value")

    def test_liveness_probe_stays_public(self, monkeypatch) -> None:
        """Container healthchecks cannot present a bearer token."""
        from app.core.security import require_api_key

        self._patch_key(monkeypatch, "s3cret-value")
        require_api_key(self._request("/healthz"), authorization=None, x_api_key=None)

    def test_detailed_health_still_requires_the_key(self, monkeypatch) -> None:
        from app.core.security import require_api_key

        self._patch_key(monkeypatch, "s3cret-value")
        with pytest.raises(ChronoscopeError):
            require_api_key(self._request("/api/system/health"), authorization=None, x_api_key=None)


class TestRateLimiter:
    def test_bucket_refills_over_time(self) -> None:
        limiter = RateLimiter(rate=100.0, burst=2)
        assert limiter.check("a")[0]
        assert limiter.check("a")[0]
        allowed, retry = limiter.check("a")
        assert not allowed and retry > 0

    def test_clients_are_isolated(self) -> None:
        limiter = RateLimiter(rate=0.01, burst=1)
        assert limiter.check("client-a")[0]
        assert not limiter.check("client-a")[0]
        assert limiter.check("client-b")[0], "one client must not exhaust another's budget"

    def test_memory_is_bounded(self) -> None:
        limiter = RateLimiter(rate=1, burst=1, max_clients=64)
        for i in range(5000):
            limiter.check(f"ip-{i}")
        assert len(limiter._buckets) <= 64


@pytest.mark.asyncio(loop_scope="module")
class TestHttpSurface:
    async def test_security_headers_are_present(self, client) -> None:  # type: ignore[no-untyped-def]
        res = await client.get("/api")
        for header in ("X-Content-Type-Options", "X-Frame-Options", "Content-Security-Policy",
                       "Referrer-Policy", "Permissions-Policy"):
            assert header in res.headers, header
        assert "frame-ancestors 'none'" in res.headers["Content-Security-Policy"]
        assert res.headers["X-Frame-Options"] == "DENY"

    async def test_malformed_video_id_is_rejected_before_any_lookup(self, client) -> None:  # type: ignore[no-untyped-def]
        for bad in ["..%2F..%2Fetc", "not-an-id", "A" * 32, "%2e%2e%2f"]:
            res = await client.get(f"/api/videos/{bad}")
            assert res.status_code in {404, 422}, bad
            assert "error" in res.json()

    async def test_delete_rejects_traversal_ids(self, client) -> None:  # type: ignore[no-untyped-def]
        res = await client.delete("/api/videos/..%2F..%2Fetc")
        assert res.status_code in {404, 422}

    async def test_non_video_upload_is_refused(self, client) -> None:  # type: ignore[no-untyped-def]
        res = await client.post(
            "/api/videos", files={"file": ("evil.mp4", b"PK\x03\x04" + b"\x00" * 64, "video/mp4")}
        )
        assert res.status_code == 415
        assert "zip archive" in res.json()["error"]["message"]

    async def test_oversized_json_body_is_refused(self, client) -> None:  # type: ignore[no-untyped-def]
        res = await client.post("/api/search", json={"query": "x" * 4000})
        assert res.status_code == 422  # query length is bounded by the schema

    async def test_query_filters_are_sanitised(self, client) -> None:  # type: ignore[no-untyped-def]
        res = await client.post(
            "/api/search",
            json={"query": "test", "video_ids": ["../../etc/passwd", "a" * 32], "speakers": ["<img src=x>"]},
        )
        assert res.status_code == 200

    async def test_traversal_through_static_frames_is_blocked(self, client) -> None:  # type: ignore[no-untyped-def]
        for attack in ["../../chronoscope.db", "..%2f..%2fchronoscope.db", "/etc/passwd"]:
            res = await client.get(f"/frames/{attack}")
            assert res.status_code in {403, 404}, attack

    async def test_docs_and_health_are_reachable(self, client) -> None:  # type: ignore[no-untyped-def]
        assert (await client.get("/healthz")).status_code == 200
        assert (await client.get("/api/openapi.json")).status_code == 200

    async def test_every_failure_uses_one_error_envelope(self, client) -> None:  # type: ignore[no-untyped-def]
        for path in ("/api/nope", "/api/videos/zzz", f"/api/videos/{'0' * 32}"):
            body = (await client.get(path)).json()
            assert set(body) == {"error"}, path
            assert {"code", "message", "detail"} <= set(body["error"]), path

    async def test_validation_errors_do_not_reflect_input(self, client) -> None:  # type: ignore[no-untyped-def]
        payload = {"query": "", "top_k": 999, "video_ids": ["<script>alert(1)</script>"]}
        body = (await client.post("/api/search", json=payload)).json()
        assert "<script>" not in str(body), "attacker-controlled input must not be echoed"

    async def test_rate_limiting_engages_when_enabled(self, client, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        from app.core.security import (
            LIMITERS,
            RateLimiter,
            reset_rate_limits,
            set_rate_limiting,
        )

        monkeypatch.setitem(LIMITERS, "default", RateLimiter(rate=0.01, burst=2))
        set_rate_limiting(True)
        try:
            statuses = [(await client.get("/api")).status_code for _ in range(5)]
        finally:
            set_rate_limiting(False)
            reset_rate_limits()
        assert statuses[:2] == [200, 200], "a short burst must be allowed"
        assert 429 in statuses, statuses


class TestUrlFetching:
    """Fetching a caller-supplied URL is a confused-deputy risk; each of these
    is a real technique for reaching inside the perimeter."""

    @pytest.mark.parametrize(
        ("url", "why"),
        [
            ("http://169.254.169.254/latest/meta-data/", "cloud metadata"),
            ("http://metadata.google.internal/computeMetadata/v1/", "gcp metadata"),
            ("http://127.0.0.1:8000/api/videos", "loopback"),
            ("http://localhost:6333/collections", "loopback by name"),
            ("http://10.1.2.3/a.mp4", "private class A"),
            ("http://172.16.5.5/a.mp4", "private class B"),
            ("http://192.168.0.10/a.mp4", "private class C"),
            ("http://[::1]/a.mp4", "ipv6 loopback"),
            ("http://0.0.0.0/a.mp4", "unspecified"),
            ("file:///etc/passwd", "file scheme"),
            ("gopher://internal/x", "gopher scheme"),
            ("ftp://host/a.mp4", "ftp scheme"),
        ],
    )
    def test_internal_and_non_http_targets_are_refused(self, url: str, why: str) -> None:
        from app.ingest.fetch import validate_url

        with pytest.raises(ChronoscopeError):
            validate_url(url)

    @pytest.mark.parametrize(
        "url",
        ["https://youtube.com/watch?v=x", "https://www.youtu.be/x", "https://vimeo.com/123",
         "https://twitter.com/i/status/1", "https://www.tiktok.com/@a/video/1"],
    )
    def test_platform_pages_are_named_not_downloaded(self, url: str) -> None:
        """These serve HTML. Failing with a specific message beats failing later
        with "not a recognised video container"."""
        from app.ingest.fetch import validate_url

        with pytest.raises(ChronoscopeError) as err:
            validate_url(url)
        assert err.value.code == "needs_extractor"

    def test_empty_and_oversized_urls(self) -> None:
        from app.ingest.fetch import validate_url

        for bad in ["", "   ", "https://example.com/" + "a" * 3000]:
            with pytest.raises(ChronoscopeError):
                validate_url(bad)

    def test_filename_derivation(self) -> None:
        from app.ingest.fetch import filename_from

        assert filename_from("https://h/x/talk.mp4", None) == "talk.mp4"
        assert filename_from("https://h/x/talk%20two.mkv", None) == "talk two.mkv"
        assert filename_from("https://h/stream", "video/webm").endswith(".webm")
        assert filename_from("https://h/", None).endswith(".mp4")
