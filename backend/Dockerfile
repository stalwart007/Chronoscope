# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------- builder ---
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build
COPY pyproject.toml ./
COPY app ./app

# INSTALL_ML=1 builds the full-fidelity image (torch, CLIP, Whisper, several
# GB larger). The default image ships the deterministic fallbacks (~1.1 GB).
ARG INSTALL_ML=0
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv && \
    if [ "$INSTALL_ML" = "1" ]; then \
      VIRTUAL_ENV=/opt/venv uv pip install --python /opt/venv/bin/python ".[ml]"; \
    else \
      VIRTUAL_ENV=/opt/venv uv pip install --python /opt/venv/bin/python "."; \
    fi

# ---------------------------------------------------------------- runtime ---
FROM python:3.12-slim AS runtime

# ffmpeg is optional (PyAV bundles its own libraries) but useful for exotic
# containers; curl provides the container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 chrono

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/data/models \
    TORCH_HOME=/data/models \
    CS_DATA_DIR=/data

COPY --from=builder /opt/venv /opt/venv
WORKDIR /srv
COPY app ./app
COPY pyproject.toml ./

RUN mkdir -p /data/uploads /data/artifacts /data/models && chown -R chrono:chrono /data /srv
USER chrono

EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=4s --start-period=40s --retries=5 \
  CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "75"]
