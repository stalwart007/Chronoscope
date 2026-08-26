"""Message, response and provider types for language model access."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass(slots=True)
class Message:
    role: Role
    content: str
    #: Base64 data URLs or local paths; only vision-capable providers use them.
    images: list[str] = field(default_factory=list)

    def to_openai(self) -> dict[str, Any]:
        if not self.images:
            return {"role": self.role, "content": self.content}
        parts: list[dict[str, Any]] = [{"type": "text", "text": self.content}]
        parts.extend({"type": "image_url", "image_url": {"url": img}} for img in self.images)
        return {"role": self.role, "content": parts}


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMProvider(ABC):
    name: str = "base"
    supports_vision: bool = False

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.15,
        max_tokens: int = 1024,
        json_mode: bool = False,
        model: str | None = None,
    ) -> LLMResponse: ...

    @abstractmethod
    async def available(self) -> bool: ...

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "vision": self.supports_vision}
