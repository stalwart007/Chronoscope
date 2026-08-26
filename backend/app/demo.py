"""Generates a synthetic demo talk with no external downloads.

Five scenes (title, architecture diagram, speaker, revenue chart, closing) with
a two-speaker audio track whose spectral character differs per speaker, plus a
scripted sidecar transcript. Used by the demo endpoint, ``make sample`` and the
test suite, which asserts exact ground truth against it.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 960, 540, 12, 16000

SCENES = [
    (0.0, 8.0, "title"),
    (8.0, 22.0, "architecture"),
    (22.0, 34.0, "speaker"),
    (34.0, 48.0, "chart"),
    (48.0, 58.0, "closing"),
]

# (start, end, speaker), speaker 0 is the presenter, speaker 1 the co-host.
TURNS = [
    (0.5, 7.5, 0), (8.5, 21.0, 0), (22.5, 27.0, 1),
    (27.5, 33.5, 0), (34.5, 47.0, 1), (48.5, 57.0, 0),
]


def font(size: int) -> Any:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10
        return ImageFont.load_default()


def frame_title(t: float) -> Image.Image:
    img = Image.new("RGB", (W, H), (12, 16, 34))
    d = ImageDraw.Draw(img)
    for y in range(H):
        d.line([(0, y), (W, y)], fill=(12 + y // 24, 16 + y // 30, 34 + y // 12))
    d.text((70, 180), "Scaling Multimodal Retrieval", font=font(44), fill=(240, 246, 255))
    d.text((70, 250), "Chronoscope Engineering Review", font=font(26), fill=(120, 200, 255))
    d.text((70, 300), "Q3 Architecture + Revenue Update", font=font(20), fill=(150, 160, 190))
    pulse = int(120 + 60 * math.sin(t * 2))
    d.ellipse([W - 180, H - 180, W - 60, H - 60], outline=(60, pulse, 220), width=6)
    return img


def frame_architecture(t: float) -> Image.Image:
    img = Image.new("RGB", (W, H), (250, 250, 252))
    d = ImageDraw.Draw(img)
    d.text((60, 40), "System Architecture Diagram", font=font(30), fill=(20, 24, 40))
    boxes = [
        (80, 150, "Ingest\nFFmpeg", (66, 133, 244)),
        (300, 150, "Embed\nCLIP", (219, 68, 55)),
        (520, 150, "Qdrant\nVectors", (15, 157, 88)),
        (740, 150, "Agent\nSwarm", (244, 160, 0)),
    ]
    for x, y, label, color in boxes:
        d.rounded_rectangle([x, y, x + 150, y + 110], radius=14, outline=color, width=4)
        d.multiline_text((x + 22, y + 30), label, font=font(19), fill=(30, 34, 50), spacing=6)
    for x in (230, 450, 670):
        d.line([(x, 205), (x + 70, 205)], fill=(90, 96, 120), width=3)
        d.polygon([(x + 70, 199), (x + 82, 205), (x + 70, 211)], fill=(90, 96, 120))
    d.text((80, 330), "Retrieval fuses text + image ranks with RRF", font=font(20), fill=(70, 76, 96))
    d.text((80, 370), f"latency budget: {180 + int(20 * math.sin(t)):d} ms p95", font=font(18), fill=(120, 126, 150))
    return img


def frame_speaker(t: float) -> Image.Image:
    img = Image.new("RGB", (W, H), (28, 24, 38))
    d = ImageDraw.Draw(img)
    for y in range(H):
        d.line([(0, y), (W, y)], fill=(28 + y // 40, 24 + y // 50, 38 + y // 30))
    cx, cy = W // 2 + int(12 * math.sin(t * 1.6)), H // 2 - 20
    d.ellipse([cx - 110, cy - 130, cx + 110, cy + 90], fill=(224, 186, 158))
    d.ellipse([cx - 130, cy + 70, cx + 130, cy + 300], fill=(46, 62, 108))
    d.ellipse([cx - 55, cy - 40, cx - 25, cy - 10], fill=(30, 30, 40))
    d.ellipse([cx + 25, cy - 40, cx + 55, cy - 10], fill=(30, 30, 40))
    mouth = 8 + int(10 * abs(math.sin(t * 7)))
    d.ellipse([cx - 30, cy + 30, cx + 30, cy + 30 + mouth], fill=(120, 50, 60))
    d.text((40, H - 60), "Live Q&A. Kubernetes rollout", font=font(22), fill=(220, 226, 240))
    return img


def frame_chart(t: float) -> Image.Image:
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((60, 36), "Quarterly Revenue Chart ($M)", font=font(30), fill=(18, 22, 36))
    data = [("Q1", 42), ("Q2", 51), ("Q3", 68), ("Q4", 91)]
    base_y, bar_w = 430, 110
    d.line([(120, base_y), (880, base_y)], fill=(180, 186, 200), width=2)
    for i, (label, val) in enumerate(data):
        x = 160 + i * 180
        h = int(val * 3.2)
        d.rectangle([x, base_y - h, x + bar_w, base_y], fill=(46, 120, 220))
        d.text((x + 30, base_y - h - 30), str(val), font=font(24), fill=(20, 24, 40))
        d.text((x + 36, base_y + 14), label, font=font(22), fill=(70, 76, 96))
    d.text((60, 480), "YoY growth accelerating; Q4 guidance raised", font=font(20), fill=(90, 96, 116))
    return img


def frame_closing(t: float) -> Image.Image:
    img = Image.new("RGB", (W, H), (8, 12, 24))
    d = ImageDraw.Draw(img)
    d.text((80, 200), "Thank you", font=font(46), fill=(240, 246, 255))
    d.text((80, 270), "Questions? chronoscope@example.com", font=font(22), fill=(110, 190, 250))
    for i in range(20):
        x = (i * 61 + int(t * 40)) % W
        d.ellipse([x, 420 + (i % 5) * 12, x + 6, 426 + (i % 5) * 12], fill=(40, 70, 130))
    return img


RENDER = {
    "title": frame_title, "architecture": frame_architecture, "speaker": frame_speaker,
    "chart": frame_chart, "closing": frame_closing,
}


def build_audio(duration: float) -> np.ndarray:
    n = int(duration * SR)
    t = np.arange(n) / SR
    audio = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(11)
    for start, end, spk in TURNS:
        i0, i1 = int(start * SR), min(n, int(end * SR))
        seg_t = t[i0:i1]
        f0 = 118.0 if spk == 0 else 196.0          # pitch separates the voices
        formants = (1.0, 2.1, 3.4) if spk == 0 else (1.0, 2.9, 4.6)
        voice = sum(a * np.sin(2 * np.pi * f0 * m * seg_t) for m, a in zip(formants, (0.6, 0.28, 0.14), strict=True))
        syllable = 0.5 + 0.5 * np.sin(2 * np.pi * (3.6 if spk == 0 else 4.8) * seg_t)
        breath = rng.normal(0, 0.02, seg_t.shape).astype(np.float32)
        audio[i0:i1] = (voice * syllable * 0.36 + breath).astype(np.float32)
    fade = int(0.02 * SR)
    audio[:fade] *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    return np.clip(audio, -0.95, 0.95)


# Cue script: (start, end, speaker, text). Written next to the media as a
# sidecar .srt so the demo has a real transcript without downloading ASR
# weights, exactly the production path for footage that ships with captions.
SCRIPT = [
    (0.6, 4.0, 0, "Welcome everyone to the Chronoscope engineering review for the third quarter."),
    (4.1, 7.6, 0, "Today I will walk through the architecture, the rollout, and the revenue picture."),
    (8.6, 12.4, 0, "Here is the system architecture diagram for the multimodal retrieval pipeline."),
    (12.5, 16.2, 0, "Ingestion uses FFmpeg and scene detection, then CLIP embeds every keyframe."),
    (16.3, 21.0, 0, "Qdrant stores the vectors and the agent swarm fuses text and image ranks with reciprocal rank fusion."),
    (22.6, 26.8, 1, "Quick question on the rollout, how did the Kubernetes migration actually go in practice?"),
    (27.6, 30.9, 0, "The Kubernetes rollout finished two weeks early with zero downtime across all regions."),
    (31.0, 33.4, 0, "We now autoscale the embedding workers based on queue depth."),
    (34.6, 38.9, 1, "Let us look at the revenue chart, this is quarterly revenue in millions of dollars."),
    (39.0, 43.2, 1, "We closed forty two million in the first quarter and fifty one million in the second."),
    (43.3, 46.9, 1, "The third quarter came in at sixty eight million and the fourth quarter reached ninety one million."),
    (48.6, 52.4, 0, "That is a strong finish, year over year growth is accelerating into next year."),
    (52.5, 56.9, 0, "Thank you all for joining, please send any follow up questions to the team."),
]


def srt_timestamp(t: float) -> str:
    ms = round(t * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: Path) -> None:
    lines = []
    for i, (start, end, spk, text) in enumerate(SCRIPT, start=1):
        lines.append(str(i))
        lines.append(f"{srt_timestamp(start)} --> {srt_timestamp(end)}")
        lines.append(f"[SPEAKER_{spk:02d}] {text}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")




def generate(out: Path) -> Path:
    """Render the demo video and its sidecar transcript. Returns the video path."""
    import av

    out.parent.mkdir(parents=True, exist_ok=True)
    duration = SCENES[-1][1]

    container = av.open(str(out), mode="w")
    vstream = container.add_stream("libx264", rate=FPS)
    vstream.width, vstream.height, vstream.pix_fmt = W, H, "yuv420p"
    vstream.options = {"crf": "23", "preset": "veryfast"}
    astream = container.add_stream("aac", rate=SR)
    astream.layout = "mono"

    for i in range(int(duration * FPS)):
        t = i / FPS
        kind = next(k for s, e, k in SCENES if s <= t < e)
        frame = av.VideoFrame.from_ndarray(np.asarray(RENDER[kind](t)), format="rgb24")
        frame.pts = i
        frame.time_base = Fraction(1, FPS)
        for packet in vstream.encode(frame):
            container.mux(packet)

    pcm = (build_audio(duration) * 32767).astype(np.int16)
    block = 1024
    for i in range(0, len(pcm), block):
        chunk = pcm[i : i + block]
        if len(chunk) < block:
            chunk = np.pad(chunk, (0, block - len(chunk)))
        aframe = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
        aframe.sample_rate = SR
        aframe.pts = i
        aframe.time_base = Fraction(1, SR)
        for packet in astream.encode(aframe):
            container.mux(packet)

    for packet in vstream.encode():
        container.mux(packet)
    for packet in astream.encode():
        container.mux(packet)
    container.close()

    write_srt(out.with_suffix(".srt"))
    return out
