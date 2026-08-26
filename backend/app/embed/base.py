"""Encoder interfaces and device selection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from app.logging_conf import get_logger

log = get_logger(__name__)


def pick_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = np.linalg.norm(x)
        return x / max(float(n), 1e-12)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class TextEncoder(ABC):
    name: str = "text"
    dim: int = 0
    degraded: bool = False

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray: ...

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "dim": self.dim, "degraded": self.degraded}


class ImageEncoder(ABC):
    """A joint image/text encoder. CLIP's text tower lives in image space, so
    the same object answers both, which is what makes text->frame search work."""

    name: str = "image"
    dim: int = 0
    degraded: bool = False

    @abstractmethod
    def encode_images(self, paths: list[str]) -> np.ndarray: ...

    @abstractmethod
    def encode_text(self, texts: list[str]) -> np.ndarray: ...

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "dim": self.dim, "degraded": self.degraded}
