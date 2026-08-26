"""Media decoding via PyAV, with the ffmpeg CLI as a fallback.

PyAV links its own FFmpeg libraries, so no system ffmpeg install is required.
``sample_frames`` does a single forward decode pass on a time grid; ``grab_frames``
seeks to specific timestamps, sorted to avoid backward seeks.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import wave
from collections.abc import Iterator, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings
from app.core.errors import DependencyUnavailable, PipelineError
from app.core.types import VideoProbe
from app.logging_conf import get_logger

log = get_logger(__name__)


def _av() -> Any:
    try:
        import av

        av.logging.set_level(av.logging.ERROR)
        return av
    except ImportError as exc:  # pragma: no cover
        raise DependencyUnavailable(
            "PyAV is required for media decoding (pip install av)", detail=str(exc)
        ) from exc


def has_ffmpeg() -> bool:
    return shutil.which(settings.ffmpeg_bin) is not None


# ---------------------------------------------------------------------- probe
def probe_video(path: str | Path) -> VideoProbe:
    av = _av()
    path = str(path)
    try:
        with av.open(path) as container:
            vstreams = container.streams.video
            if not vstreams:
                raise PipelineError("probe", "file contains no video stream")
            v = vstreams[0]
            astreams = container.streams.audio
            a = astreams[0] if astreams else None
            duration = float(container.duration / av.time_base) if container.duration else 0.0
            if not duration and v.duration and v.time_base:
                duration = float(v.duration * v.time_base)
            fps = float(v.average_rate or v.guessed_rate or Fraction(25, 1))
            return VideoProbe(
                duration=round(duration, 3),
                width=int(v.codec_context.width or 0),
                height=int(v.codec_context.height or 0),
                fps=round(fps, 4),
                codec=v.codec_context.name or "unknown",
                has_audio=a is not None,
                audio_channels=int(getattr(a.codec_context, "channels", 0) or 0) if a else 0,
                audio_sample_rate=int(getattr(a.codec_context, "sample_rate", 0) or 0) if a else 0,
                bit_rate=int(container.bit_rate or 0),
                container=container.format.name if container.format else "",
            )
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError("probe", f"cannot open media: {exc}") from exc


# --------------------------------------------------------------------- frames
def _resize(arr: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = arr.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return arr
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    ys = (np.arange(nh) * (h / nh)).astype(np.int32)
    xs = (np.arange(nw) * (w / nw)).astype(np.int32)
    return arr[np.ix_(ys, xs)]


def sample_frames(
    path: str | Path, *, stride_s: float = 0.5, max_dim: int = 320, start: float = 0.0, end: float | None = None
) -> Iterator[tuple[float, np.ndarray]]:
    """Single forward decode pass, emitting ~one frame per ``stride_s``."""
    av = _av()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if start > 0:
            with contextlib.suppress(Exception):
                container.seek(int(start / av.time_base), stream=None)
        next_t = start
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            if t + 1e-6 < next_t:
                continue
            if end is not None and t > end:
                break
            arr = frame.to_ndarray(format="rgb24")
            yield t, _resize(arr, max_dim)
            next_t = t + stride_s


def grab_frames(
    path: str | Path, timestamps: Sequence[float], *, max_dim: int = 896
) -> list[tuple[float, np.ndarray]]:
    """Seek-and-decode a specific set of timestamps (sorted internally)."""
    av = _av()
    out: list[tuple[float, np.ndarray]] = []
    wanted = sorted({round(float(t), 3) for t in timestamps})
    if not wanted:
        return out
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = stream.time_base or Fraction(1, 1000)
        for target in wanted:
            try:
                container.seek(max(0, int(target / tb)), stream=stream, any_frame=False, backward=True)
            except Exception as exc:  # unseekable point, skip it
                log.debug("seek to %.2fs failed: %s", target, exc)
                continue
            best: tuple[float, np.ndarray] | None = None
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                t = float(frame.pts * tb)
                arr = frame.to_ndarray(format="rgb24")
                best = (t, arr)
                if t >= target:
                    break
            if best is not None:
                out.append((best[0], _resize(best[1], max_dim)))
    return out


def thumb_path(path: str | Path) -> Path:
    """Companion thumbnail path for a keyframe (``x.jpg`` -> ``x.thumb.jpg``)."""
    p = Path(path)
    return p.with_suffix(f".thumb{p.suffix}")


def save_frame(
    arr: np.ndarray, path: str | Path, *, quality: int = 88, thumbnail: int | None = 384
) -> tuple[int, int]:
    """Write a keyframe and, by default, a small companion thumbnail.

    The UI paints dozens of frames at once in the filmstrip, the evidence list
    and the hover preview, all at well under 200 px. Serving the full-size
    frame for those turns a timeline scrub into megabytes of transfer and
    decode, so a thumbnail is written once at ingest and the full frame is
    reserved for the vision model and full-screen viewing.
    """
    from PIL import Image

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(p, format="JPEG", quality=quality, optimize=True)
    if thumbnail and max(img.size) > thumbnail:
        small = img.copy()
        small.thumbnail((thumbnail, thumbnail), Image.Resampling.LANCZOS)
        small.save(thumb_path(p), format="JPEG", quality=72, optimize=True)
    return img.width, img.height


# ---------------------------------------------------------------------- audio
def extract_audio(path: str | Path, out_wav: str | Path, *, sample_rate: int = 16000) -> np.ndarray:
    """Decode + resample to mono float32 @ ``sample_rate``, also writing a WAV.

    Whisper wants 16 kHz mono; doing the resample once here means the ASR stage
    never re-decodes the container and the diarizer reuses the same array.
    """
    av = _av()
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[np.ndarray] = []
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise PipelineError("audio", "no audio stream")
            stream = container.streams.audio[0]
            stream.thread_type = "AUTO"
            resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray().reshape(-1))
            for out in resampler.resample(None):  # flush
                chunks.append(out.to_ndarray().reshape(-1))
    except PipelineError:
        raise
    except Exception as exc:
        if has_ffmpeg():
            log.warning("PyAV audio decode failed (%s), retrying with ffmpeg CLI", exc)
            return _ffmpeg_audio(path, out_wav, sample_rate)
        raise PipelineError("audio", f"audio decode failed: {exc}") from exc

    pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.astype(np.int16).tobytes())
    return pcm.astype(np.float32) / 32768.0


def _ffmpeg_audio(path: str | Path, out_wav: Path, sample_rate: int) -> np.ndarray:
    cmd = [
        settings.ffmpeg_bin, "-nostdin", "-y", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(out_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        raise PipelineError("audio", f"ffmpeg failed: {proc.stderr.decode()[-400:]}")
    with wave.open(str(out_wav), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


async def probe_async(path: str | Path) -> VideoProbe:
    from app.core.concurrency import run_blocking

    return await run_blocking(probe_video, path)


async def make_poster(path: str | Path, out: str | Path, *, at: float = 1.0) -> bool:
    """Best-effort poster image for the library grid."""
    from app.core.concurrency import run_blocking

    def work() -> bool:
        frames = grab_frames(path, [at], max_dim=640) or grab_frames(path, [0.0], max_dim=640)
        if not frames:
            return False
        save_frame(frames[0][1], out, quality=82, thumbnail=None)
        return True

    with contextlib.suppress(Exception):
        return await run_blocking(work)
    return False


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def estimate_stride(duration: float, *, target_samples: int = 1600, floor: float = 0.25) -> float:
    """Adaptive sampling stride: constant work regardless of video length."""
    if duration <= 0:
        return floor
    return max(floor, duration / max(target_samples, 1))


