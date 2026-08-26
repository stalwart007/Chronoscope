"""Language model providers.

Ollama runs locally with no key. OpenRouter and Groq have free tiers. All three
speak an OpenAI-shaped JSON dialect, so they differ only in endpoint, auth
header and how JSON mode is requested.
"""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import re
import time
from pathlib import Path
from typing import Any

import httpx

from app.config import settings
from app.core.errors import LLMError
from app.llm.base import LLMProvider, LLMResponse, Message
from app.logging_conf import get_logger

log = get_logger(__name__)

_CLIENT: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    global _CLIENT
    if _CLIENT is None or _CLIENT.is_closed:
        _CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.llm_timeout_s, connect=8.0),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
            follow_redirects=True,
        )
    return _CLIENT


async def close_client() -> None:
    global _CLIENT
    if _CLIENT is not None and not _CLIENT.is_closed:
        await _CLIENT.aclose()
    _CLIENT = None


def encode_image(path: str | Path, *, max_dim: int = 1024) -> str:
    """Local image -> ``data:`` URL, downscaled so prompts stay small."""
    p = Path(path)
    try:
        from PIL import Image

        with Image.open(p) as handle:
            img = handle.convert("RGB")
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                img = img.resize((int(img.width * ratio), int(img.height * ratio)))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=82)
            data = buf.getvalue()
        mime = "image/jpeg"
    except Exception:
        data = p.read_bytes()
        mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


class OllamaProvider(LLMProvider):
    name = "ollama"
    supports_vision = True

    def __init__(self) -> None:
        self.url = settings.ollama_url.rstrip("/")
        self.model = settings.ollama_model
        self.vision_model = settings.ollama_vision_model

    async def available(self) -> bool:
        try:
            r = await client().get(f"{self.url}/api/tags", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False

    async def installed_models(self) -> list[str]:
        try:
            r = await client().get(f"{self.url}/api/tags", timeout=3.0)
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    async def chat(
        self, messages: list[Message], *, temperature: float = 0.15, max_tokens: int = 1024,
        json_mode: bool = False, model: str | None = None,
    ) -> LLMResponse:
        has_images = any(m.images for m in messages)
        target = model or (self.vision_model if has_images else self.model)
        payload: dict[str, Any] = {
            "model": target,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    **({"images": [i.split(",", 1)[-1] for i in m.images]} if m.images else {}),
                }
                for m in messages
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        t0 = time.perf_counter()
        r = await client().post(f"{self.url}/api/chat", json=payload)
        if r.status_code >= 400:
            raise LLMError(f"ollama {r.status_code}: {r.text[:300]}")
        data = r.json()
        return LLMResponse(
            text=(data.get("message") or {}).get("content", ""),
            model=target,
            provider=self.name,
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=int(data.get("eval_count", 0)),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


class OpenAICompatProvider(LLMProvider):
    """Shared implementation for OpenRouter / Groq / any OpenAI-shaped API."""

    endpoint: str = ""
    api_key: str | None = None
    model: str = ""
    vision_model: str = ""
    extra_headers: dict[str, str]

    def __init__(self) -> None:  # pragma: no cover - overridden by subclasses
        self.extra_headers = {}

    async def available(self) -> bool:
        return bool(self.api_key)

    async def chat(
        self, messages: list[Message], *, temperature: float = 0.15, max_tokens: int = 1024,
        json_mode: bool = False, model: str | None = None,
    ) -> LLMResponse:
        if not self.api_key:
            raise LLMError(f"{self.name}: no API key configured")
        has_images = any(m.images for m in messages)
        target = model or (self.vision_model if has_images and self.vision_model else self.model)
        payload: dict[str, Any] = {
            "model": target,
            "messages": [m.to_openai() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}", **self.extra_headers}
        t0 = time.perf_counter()
        r = await client().post(self.endpoint, json=payload, headers=headers)
        if r.status_code == 429:
            raise LLMError(f"{self.name}: rate limited", code="rate_limited")
        if r.status_code >= 400:
            raise LLMError(f"{self.name} {r.status_code}: {r.text[:300]}")
        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        return LLMResponse(
            text=(choice.get("message") or {}).get("content", "") or "",
            model=data.get("model", target),
            provider=self.name,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


class OpenRouterProvider(OpenAICompatProvider):
    name = "openrouter"
    supports_vision = True

    def __init__(self) -> None:
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self.api_key = settings.openrouter_key
        self.model = settings.openrouter_model
        self.vision_model = settings.openrouter_vision_model
        self.extra_headers = {
            "HTTP-Referer": "https://github.com/chronoscope",
            "X-Title": "Chronoscope",
        }


class GroqProvider(OpenAICompatProvider):
    name = "groq"
    supports_vision = False

    def __init__(self) -> None:
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self.api_key = settings.groq_key
        self.model = settings.groq_model
        self.vision_model = ""


PROVIDERS: dict[str, type[LLMProvider]] = {
    "ollama": OllamaProvider,
    "openrouter": OpenRouterProvider,
    "groq": GroqProvider,
}


def _balanced_slice(s: str, start: int, opener: str, closer: str) -> str | None:
    """Slice out the balanced ``opener...closer`` region beginning at ``start``."""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def extract_json(text: str) -> dict[str, Any] | list[Any] | None:
    """Recover a JSON value from a chatty completion.

    Small free models wrap JSON in prose or fences, and occasionally emit a
    trailing comma or single quotes. We strip fences, take whichever of ``{``
    or ``[`` appears first, slice the balanced region with a depth counter that
    respects string literals, then apply cheap repairs before giving up.
    """
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s.lstrip("`")
        if s[:4].lower() == "json":
            s = s[4:]
        s = s.strip()

    positions = [(s.find(o), o, c) for o, c in (("{", "}"), ("[", "]")) if s.find(o) >= 0]
    for start, opener, closer in sorted(positions):
        blob = _balanced_slice(s, start, opener, closer)
        if blob is None:
            continue
        repairs = (
            blob,
            re.sub(r",\s*([}\]])", r"\1", blob),                       # trailing commas
            re.sub(r",\s*([}\]])", r"\1", blob.replace("'", '"')),      # single quotes
        )
        for attempt in repairs:
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None
