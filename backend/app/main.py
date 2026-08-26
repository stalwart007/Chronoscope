"""ASGI application: lifespan, middleware, routing and static artefacts."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app import __version__
from app.api import routes_query, routes_system, routes_videos
from app.config import settings
from app.core.errors import ChronoscopeError
from app.core.security import (
    enforce_rate_limit,
    redact,
    require_api_key,
    safe_public_error,
    set_rate_limiting,
)
from app.embed import registry as embeds
from app.llm.providers import close_client
from app.logging_conf import configure_logging, get_logger, request_id
from app.store.db import dispose_db, init_db
from app.store.registry import get_store, reset_store
from app.workers.runner import runner

log = get_logger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging(settings.log_level, settings.log_json)
    settings.ensure_dirs()
    log.info("starting %s v%s (env=%s)", settings.app_name, __version__, settings.env)
    set_rate_limiting(settings.rate_limit_enabled)
    await init_db()
    await runner.start()
    if settings.env == "prod" and not settings.api_key:
        log.warning(
            "running in prod with no CS_API_KEY, the API is open to anyone who can reach it"
        )

    async def warm() -> None:
        # Model loading is slow; do it off the critical path so /health answers
        # immediately and the UI can show "warming up" instead of hanging.
        try:
            info = await embeds.warm_up()
            await get_store(await embeds.dims())
            log.info("encoders ready: %s", info)
        except Exception as exc:
            log.exception("warm-up failed: %s", exc)

    async def checkpoint() -> None:
        """Persist the in-process index periodically.

        Upserts mark the store dirty rather than rewriting every vector inline;
        this coalesces those writes. A crash between checkpoints loses only the
        ANN index, which is derived data, the relational store is the source
        of truth and a re-index rebuilds it.
        """
        while True:
            await asyncio.sleep(settings.checkpoint_interval_s)
            try:
                store = await get_store()
                flush = getattr(store, "flush", None)
                if flush is not None:
                    await flush()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("index checkpoint failed: %s", exc)

    warm_task = asyncio.create_task(warm(), name="warmup")
    checkpoint_task = asyncio.create_task(checkpoint(), name="checkpoint")
    try:
        yield
    finally:
        for task in (warm_task, checkpoint_task):
            task.cancel()
            # CancelledError derives from BaseException, so suppressing
            # `Exception` alone lets teardown raise out of the lifespan.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await runner.stop()
        await close_client()
        await reset_store()
        await dispose_db()
        log.info("shutdown complete")


app = FastAPI(
    title="Chronoscope",
    version=__version__,
    summary="Multimodal video analytics engine, scene-aware ingestion, cross-modal retrieval, agentic Q&A.",
    lifespan=lifespan,
    docs_url="/api/docs" if settings.enable_docs else None,
    openapi_url="/api/openapi.json" if settings.enable_docs else None,
    redoc_url=None,
    dependencies=[Depends(require_api_key)],
    # Match routes exactly. Implicit slash redirects turn `%2e%2e%2f` into a
    # 307 toward a normalised path, harmless here, but path normalisation is
    # a recurring source of bypasses and this endpoint set does not need it.
    redirect_slashes=False,
)

app.add_middleware(GZipMiddleware, minimum_size=1024)

# Credentialed CORS and a wildcard origin are mutually exclusive. Browsers
# reject the combination, so allowing it in config would produce a deployment
# that appears protected but is not.
_origins = settings.cors_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=bool(_origins),
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID", "Last-Event-ID"],
    expose_headers=["X-Request-ID", "X-Elapsed-Ms", "Retry-After"],
    max_age=600,
)

if settings.allowed_host_list != ["*"]:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_host_list)


#: Static, framework-independent hardening. The CSP is strict because the SPA
#: ships no inline scripts; `frame-ancestors 'none'` blocks clickjacking, and
#: `media-src`/`img-src 'self'` keep derived artefacts same-origin.
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-site",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "img-src 'self' data: blob:; media-src 'self' blob:; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com data:; "
        "script-src 'self'; connect-src 'self'; "
        "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
}


@app.middleware("http")
async def harden(request: Request, call_next: Any) -> Any:
    # Bound non-upload bodies before they are buffered. Uploads stream to disk
    # under their own limit and must not be capped here.
    if request.method in {"POST", "PUT", "PATCH"} and not request.url.path.startswith("/api/videos"):
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > settings.max_request_bytes:
            return JSONResponse(
                status_code=413,
                content={"error": {"code": "payload_too_large", "message": "request body is too large", "detail": None}},
            )
    try:
        enforce_rate_limit(request)
    except ChronoscopeError as exc:
        retry = (exc.detail or {}).get("retry_after", 1) if isinstance(exc.detail, dict) else 1
        response = JSONResponse(status_code=exc.status_code, content=exc.to_payload())
        response.headers["Retry-After"] = str(max(1, int(float(retry))))
        return response

    response = await call_next(request)
    for header, value in _SECURITY_HEADERS.items():
        response.headers.setdefault(header, value)
    if settings.env == "prod":
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response


@app.middleware("http")
async def correlate(request: Request, call_next: Any) -> Any:
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request_id.set(rid)
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = rid
    response.headers["X-Elapsed-Ms"] = f"{elapsed:.1f}"
    if elapsed > 2000 and not request.url.path.endswith(("/events", "/stream")):
        log.warning("slow request %s %s took %.0f ms", request.method, request.url.path, elapsed)
    return response


@app.exception_handler(ChronoscopeError)
async def domain_error(_request: Request, exc: ChronoscopeError) -> JSONResponse:
    payload = exc.to_payload()
    payload["error"]["message"] = redact(payload["error"]["message"])
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.exception_handler(StarletteHTTPException)
async def http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Normalise framework errors into the app's envelope.

    Starlette answers unmatched routes with ``{"detail": "Not Found"}``, so a
    client would otherwise need two parsers for the same failure class.
    """
    codes = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found",
             405: "method_not_allowed", 413: "payload_too_large", 429: "rate_limited"}
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": codes.get(exc.status_code, "http_error"),
                "message": redact(str(exc.detail)),
                "detail": None,
            }
        },
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    # Validation errors carry the offending input; echoing it back verbatim
    # reflects attacker-controlled content, so only the location and rule are
    # returned.
    problems = [
        {"field": ".".join(str(p) for p in err.get("loc", ())), "rule": err.get("type", "invalid")}
        for err in exc.errors()[:10]
    ]
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "validation_error", "message": "invalid request", "detail": problems}},
    )


@app.exception_handler(Exception)
async def unhandled(_request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"error": safe_public_error(exc)})


app.include_router(routes_videos.router)
app.include_router(routes_query.router)
app.include_router(routes_system.router)

settings.ensure_dirs()
(settings.artifact_dir / "frames").mkdir(parents=True, exist_ok=True)
# StaticFiles resolves and prefix-checks every path, so traversal via `..` or
# an absolute component is rejected before it touches the filesystem.
app.mount("/artifacts", StaticFiles(directory=str(settings.artifact_dir)), name="artifacts")
app.mount("/frames", StaticFiles(directory=str(settings.artifact_dir / "frames")), name="frames")


@app.get("/api", tags=["system"])
async def root() -> dict[str, Any]:
    return {
        "name": settings.app_name,
        "version": __version__,
        "docs": "/api/docs",
        "endpoints": ["/api/videos", "/api/query", "/api/search", "/api/system/health"],
    }


@app.get("/healthz", tags=["system"], include_in_schema=False)
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
