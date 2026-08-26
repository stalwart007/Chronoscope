"""Model-backed encoders: sentence-transformers for text, open_clip for vision.

Both are imported inside the constructor so a deployment without torch still
boots and falls back to the lightweight encoders.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.config import settings
from app.embed.base import ImageEncoder, TextEncoder, l2, pick_device
from app.logging_conf import get_logger

log = get_logger(__name__)


class SentenceTransformerEncoder(TextEncoder):
    """Dense sentence encoder (default: all-MiniLM-L6-v2, 384-d, CPU-friendly).

    Set ``CS_TEXT_MODEL=nomic-ai/nomic-embed-text-v1.5`` for the stronger
    768-d model recommended in the spec, it needs ``trust_remote_code`` and
    task prefixes, both handled below.
    """

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or settings.text_model
        self.device = device or pick_device(settings.device)
        self.name = self.model_name
        kwargs: dict[str, Any] = {"device": self.device}
        if "nomic" in self.model_name.lower():
            kwargs["trust_remote_code"] = True
        self._model = SentenceTransformer(self.model_name, **kwargs)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self._nomic = "nomic" in self.model_name.lower()
        log.info("text encoder %s (dim=%d, device=%s)", self.model_name, self.dim, self.device)

    def _prefix(self, texts: list[str], task: str) -> list[str]:
        if not self._nomic:
            return texts
        return [f"{task}: {t}" for t in texts]

    def encode(self, texts: list[str], *, task: str = "search_document") -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(
            self._prefix(texts, task),
            batch_size=settings.embed_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts, task="search_query")


class ClipEncoder(ImageEncoder):
    """open_clip image+text towers sharing one embedding space."""

    def __init__(self, model: str | None = None, pretrained: str | None = None, device: str | None = None) -> None:
        import open_clip
        import torch

        self._torch = torch
        self.model_name = model or settings.clip_model
        self.pretrained = pretrained or settings.clip_pretrained
        self.device = device or pick_device(settings.device)
        self.name = f"{self.model_name}/{self.pretrained}"
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device
        )
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        with torch.inference_mode():
            probe = self._model.encode_text(self._tokenizer(["probe"]).to(self.device))
        self.dim = int(probe.shape[-1])
        log.info("clip encoder %s (dim=%d, device=%s)", self.name, self.dim, self.device)

    def encode_images(self, paths: list[str]) -> np.ndarray:
        from PIL import Image

        if not paths:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[np.ndarray] = []
        bs = settings.embed_batch_size
        for i in range(0, len(paths), bs):
            tensors = []
            for p in paths[i : i + bs]:
                try:
                    with Image.open(p) as im:
                        tensors.append(self._preprocess(im.convert("RGB")))
                except Exception as exc:
                    log.warning("frame %s unreadable: %s", p, exc)
                    tensors.append(self._torch.zeros(3, 224, 224))
            batch = self._torch.stack(tensors).to(self.device)
            with self._torch.inference_mode():
                feats = self._model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            out.append(feats.float().cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # CLIP was trained on caption-like prompts; wrapping bare queries in a
        # template measurably improves zero-shot retrieval.
        prompts = [t if len(t.split()) > 6 else f"a photo of {t}" for t in texts]
        tokens = self._tokenizer(prompts).to(self.device)
        with self._torch.inference_mode():
            feats = self._model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return l2(feats.float().cpu().numpy())
