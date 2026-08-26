"""Shared fixtures.

Tests run fully offline against the degraded encoders and the in-process
vector store, so CI needs no models, no Qdrant and no network.

The data directory is redirected in ``pytest_configure``, *before* any test
module is imported. ``app.config`` builds its settings singleton at import
time, so setting ``CS_DATA_DIR`` from a fixture would be too late for any
module that imports the app during collection, and the suite would silently
read and write the developer's real ``data/`` directory.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

_TMP_ROOT: Path | None = None


def pytest_configure(config: object) -> None:
    global _TMP_ROOT
    _TMP_ROOT = Path(tempfile.mkdtemp(prefix="chronoscope-test-"))
    os.environ["CS_DATA_DIR"] = str(_TMP_ROOT)
    os.environ.setdefault("CS_ENV", "test")
    os.environ.setdefault("CS_VECTOR_BACKEND", "memory")
    os.environ.setdefault("CS_LLM_PROVIDER_CHAIN", "")
    os.environ.setdefault("CS_LOG_LEVEL", "WARNING")
    # Deterministic tests: the limiter is exercised directly in test_security.
    os.environ.setdefault("CS_RATE_LIMIT_ENABLED", "false")
    os.environ.setdefault("CS_DIARIZATION_ENABLED", "true")


def pytest_unconfigure(config: object) -> None:
    if _TMP_ROOT is not None:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)


import pytest
import pytest_asyncio


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
def data_root() -> Path:
    from app.config import settings

    assert str(settings.data_dir).startswith(tempfile.gettempdir()), (
        f"tests must not touch the real data dir (got {settings.data_dir})"
    )
    settings.ensure_dirs()
    return settings.data_dir


@pytest.fixture(scope="session")
def sample_video(data_root: Path) -> Path:
    """Render the synthetic talk once per session."""
    import subprocess

    out = data_root / "uploads" / "sample_talk.mp4"
    if not out.exists():
        script = BACKEND.parent / "scripts" / "make_sample.py"
        subprocess.run(
            [sys.executable, str(script), "--out", str(out)], check=True, capture_output=True
        )
    return out


@pytest.fixture(scope="session")
def brief_events_video(data_root: Path) -> tuple[Path, list[float]]:
    """A recording whose only distinct content is five short events.

    Uniform sampling misses these once the stride grows, which is what the
    adaptive sampler exists to prevent.
    """
    import math

    import av
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    out = data_root / "uploads" / "brief_events.mp4"
    events = [(30.0, 1.2), (70.0, 0.8), (110.0, 1.5), (160.0, 0.6), (210.0, 1.0)]
    edges = [t for start, dur in events for t in (start, start + dur)]
    if out.exists():
        return out, edges

    # Short enough to encode quickly; the strides under test represent long
    # recordings, so the fixture's own length does not need to.
    W, H, FPS, DURATION = 480, 270, 25, 240.0

    def font(size: int):
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    def render(t: float) -> Image.Image:
        for i, (start, dur) in enumerate(events):
            if start <= t < start + dur:
                img = Image.new("RGB", (W, H), (250, 250, 252))
                d = ImageDraw.Draw(img)
                d.text((40, 40), f"CHART {i}", font=font(46), fill=(15, 20, 40))
                for k in range(4):
                    d.rectangle([60 + k * 130, 300 - (k + 1) * 45, 160 + k * 130, 300], fill=(46, 120, 220))
                return img
        img = Image.new("RGB", (W, H), (18, 20, 30))
        d = ImageDraw.Draw(img)
        d.text((40, 160), "presenter talking", font=font(28), fill=(120, 130, 160))
        d.ellipse([480, 120, 580, 220], fill=(60 + int(20 * math.sin(t)), 70, 110))
        return img

    out.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(out), mode="w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    stream.options = {"crf": "28", "preset": "ultrafast"}
    for i in range(int(DURATION * FPS)):
        frame = av.VideoFrame.from_ndarray(np.asarray(render(i / FPS)), format="rgb24")
        frame.pts = i
        frame.time_base = Fraction(1, FPS)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return out, edges


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def client(data_root: Path):  # type: ignore[no-untyped-def]
    import httpx

    from app.main import app

    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=120
    ) as c:
        yield c
