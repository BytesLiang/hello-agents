"""Simple unified LLM client using the OpenAI Python SDK."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, cast

from hello_agents.llm.config import LLMConfig
from hello_agents.llm.types import LLMMessage, LLMResponse

if TYPE_CHECKING:
    from openai import OpenAI


class LLMClient:
    """Wrap the OpenAI SDK behind a small project-level interface."""

    def __init__(self, config: LLMConfig, client: Any | None = None) -> None:
        """Store config and lazily create the underlying SDK client."""

        self._config = config
        self._client = client or self._build_client()

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: Sequence[dict[str, object]] | None = None,
    ) -> LLMResponse:
        """Create a non-streaming chat completion."""

        request_kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": cast(
                Any,
                [self._message_to_dict(message) for message in messages],
            ),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools is not None:
            request_kwargs["tools"] = cast(Any, list(tools))

        response = self._client.chat.completions.create(
            **request_kwargs,
        )

        choice = response.choices[0]
        content = choice.message.content or ""
        usage = response.usage

        return LLMResponse(
            model=response.model,
            content=content,
            finish_reason=choice.finish_reason,
            prompt_tokens=0 if usage is None else usage.prompt_tokens,
            completion_tokens=0 if usage is None else usage.completion_tokens,
            total_tokens=0 if usage is None else usage.total_tokens,
        )

    def stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Stream text deltas from the model."""

        stream = cast(
            Any,
            self._client.chat.completions.create(
                model=self._config.model,
                messages=cast(
                    Any,
                    [self._message_to_dict(message) for message in messages],
                ),
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            ),
        )

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if isinstance(content, str) and content:
                yield content

    def _build_client(self) -> OpenAI:
        """Create the SDK client."""

        from openai import OpenAI

        return OpenAI(
            api_key=self._config.resolved_api_key(),
            base_url=self._config.base_url,
            timeout=self._config.timeout,
        )

    @staticmethod
    def _message_to_dict(message: LLMMessage) -> dict[str, str]:
        """Convert a message dataclass to the OpenAI SDK payload shape."""

        return {"role": message.role, "content": message.content}
