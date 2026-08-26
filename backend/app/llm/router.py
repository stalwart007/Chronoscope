"""Provider routing: ordered fallback, circuit breaking, retries, JSON coercion.

``CS_LLM_PROVIDER_CHAIN`` defines preference order. Providers whose circuit is
open are skipped, transient failures are retried with jittered backoff, and
per-provider health is recorded for the health endpoint. If nothing answers,
the agent layer switches to its deterministic path.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.core.concurrency import CircuitBreaker, TokenBucket, with_retries
from app.core.errors import LLMError
from app.core.security import redact
from app.llm.base import LLMProvider, LLMResponse, Message
from app.llm.providers import PROVIDERS, extract_json
from app.logging_conf import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class ProviderState:
    provider: LLMProvider
    breaker: CircuitBreaker = field(default_factory=lambda: CircuitBreaker(threshold=3, cooldown=45.0))
    bucket: TokenBucket = field(default_factory=lambda: TokenBucket(rate=1.5, capacity=6))
    calls: int = 0
    failures: int = 0
    last_error: str = ""
    available: bool | None = None
    tokens_in: int = 0
    tokens_out: int = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.provider.name,
            "vision": self.provider.supports_vision,
            "state": self.breaker.state,
            "available": self.available,
            "calls": self.calls,
            "failures": self.failures,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "last_error": redact(self.last_error, limit=200),
        }


class LLMRouter:
    def __init__(self) -> None:
        self._states: list[ProviderState] = []
        self._probed = False
        self._lock = asyncio.Lock()

    async def _ensure(self) -> list[ProviderState]:
        async with self._lock:
            if not self._states:
                for name in settings.provider_chain:
                    cls = PROVIDERS.get(name)
                    if cls is None:
                        continue
                    try:
                        self._states.append(ProviderState(provider=cls()))
                    except Exception as exc:
                        log.warning("provider %s could not be constructed: %s", name, exc)
            if not self._probed:
                self._probed = True
                results = await asyncio.gather(
                    *(s.provider.available() for s in self._states), return_exceptions=True
                )
                for state, ok in zip(self._states, results, strict=True):
                    state.available = bool(ok) if not isinstance(ok, BaseException) else False
                usable = [s.provider.name for s in self._states if s.available]
                log.info("LLM providers available: %s", usable or "none (heuristic mode)")
        return self._states

    async def any_available(self) -> bool:
        return any(s.available for s in await self._ensure())

    async def health(self) -> dict[str, Any]:
        states = await self._ensure()
        return {
            "chain": settings.provider_chain,
            "any_available": any(s.available for s in states),
            "providers": [s.snapshot() for s in states],
        }

    async def refresh(self) -> None:
        self._probed = False
        await self._ensure()

    # ------------------------------------------------------------------ calls
    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int = 1024,
        json_mode: bool = False,
        require_vision: bool = False,
    ) -> LLMResponse:
        states = await self._ensure()
        candidates = [
            s for s in states
            if s.available and (s.provider.supports_vision or not require_vision) and s.breaker.allow()
        ]
        if not candidates:
            open_circuits = [s.provider.name for s in states if s.available and not s.breaker.allow()]
            raise LLMError(
                "no LLM provider available"
                + (f" (circuit open: {', '.join(open_circuits)})" if open_circuits else ""),
                detail={"chain": settings.provider_chain, "vision_required": require_vision},
            )
        last: Exception | None = None
        for state in candidates:
            try:
                await state.bucket.take()

                async def call(st: ProviderState = state) -> LLMResponse:
                    return await st.provider.chat(
                        messages,
                        temperature=settings.llm_temperature if temperature is None else temperature,
                        max_tokens=max_tokens,
                        json_mode=json_mode,
                    )

                res = await with_retries(call, attempts=settings.llm_max_retries + 1, base_delay=0.6)
                state.calls += 1
                state.tokens_in += res.prompt_tokens
                state.tokens_out += res.completion_tokens
                state.breaker.record_success()
                return res
            except Exception as exc:
                state.failures += 1
                state.last_error = str(exc)
                state.breaker.record_failure()
                last = exc
                log.warning("provider %s failed: %s", state.provider.name, exc)
        raise LLMError(f"all providers failed: {last}")

    async def chat_json(
        self,
        messages: list[Message],
        *,
        schema_hint: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        require_vision: bool = False,
        repair: bool = True,
    ) -> dict[str, Any]:
        """Chat expecting JSON, with one repair round-trip on malformed output."""
        msgs = list(messages)
        if schema_hint:
            msgs.insert(
                0,
                Message(
                    role="system",
                    content=(
                        "You output ONLY valid minified JSON. No prose, no markdown fences.\n"
                        f"Conform exactly to this shape:\n{schema_hint}"
                    ),
                ),
            )
        res = await self.chat(
            msgs, temperature=temperature, max_tokens=max_tokens, json_mode=True, require_vision=require_vision
        )
        parsed = extract_json(res.text)
        if isinstance(parsed, dict):
            parsed.setdefault("_meta", {})["model"] = f"{res.provider}/{res.model}"
            return parsed
        if isinstance(parsed, list):
            return {"items": parsed, "_meta": {"model": f"{res.provider}/{res.model}"}}
        if not repair:
            raise LLMError("model did not return JSON", detail=res.text[:400])
        fixed = await self.chat(
            [
                Message(role="system", content="Convert the user's text into valid minified JSON only."),
                Message(role="user", content=res.text[:4000]),
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            json_mode=True,
        )
        parsed = extract_json(fixed.text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"items": parsed}
        raise LLMError("model did not return JSON after repair", detail=fixed.text[:400])


router = LLMRouter()
