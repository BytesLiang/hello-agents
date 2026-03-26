"""Tests for the simplified LLM client."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

from _pytest.monkeypatch import MonkeyPatch

from hello_agents.llm import LLMClient, LLMConfig, LLMMessage, LLMToolCall


class FakeChatCompletions:
    """Mimic the OpenAI chat completions interface."""

    def __init__(self) -> None:
        """Initialize fake responses and captured payloads."""

        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        """Return a fake response or a fake stream."""

        self.calls.append(kwargs)

        if kwargs.get("stream") is True:
            return _build_stream()

        return SimpleNamespace(
            model="qwen-local",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="hello"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=5,
                completion_tokens=7,
                total_tokens=12,
            ),
        )


class FakeOpenAIClient:
    """Provide a fake OpenAI SDK client."""

    def __init__(self) -> None:
        """Expose the nested chat completion namespace."""

        self.chat = SimpleNamespace(completions=FakeChatCompletions())


def _build_stream() -> Iterator[object]:
    """Yield fake streaming chunks."""

    yield SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content="hel"))]
    )
    yield SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content="lo"))]
    )


def test_llm_client_chat_uses_openai_sdk_shape() -> None:
    """Normalize a standard chat completion response."""

    fake_client = FakeOpenAIClient()
    llm = LLMClient(
        config=LLMConfig(
            model="qwen-local",
            base_url="http://localhost:11434/v1",
            provider="ollama",
        ),
        client=fake_client,
    )

    response = llm.chat(
        [LLMMessage(role="user", content="ping")],
        max_tokens=64,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo input.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert response.content == "hello"
    assert response.model == "qwen-local"
    assert response.total_tokens == 12
    call = fake_client.chat.completions.calls[0]
    assert call["model"] == "qwen-local"
    assert call["max_tokens"] == 64
    assert call["messages"] == [{"role": "user", "content": "ping"}]
    assert call["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo input.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_llm_client_normalizes_tool_calls() -> None:
    """Normalize OpenAI tool calls into framework-level tool call objects."""

    fake_client = FakeOpenAIClient()
    fake_client.chat.completions.create = lambda **kwargs: SimpleNamespace(
        model="qwen-local",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(
                                name="tavily_search",
                                arguments='{"query":"python"}',
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=5,
            completion_tokens=7,
            total_tokens=12,
        ),
    )
    llm = LLMClient(
        config=LLMConfig(model="gpt-4o-mini"),
        client=fake_client,
    )

    response = llm.chat([LLMMessage(role="user", content="search")])

    assert response.tool_calls == (
        LLMToolCall(
            id="call_1",
            name="tavily_search",
            arguments={"query": "python"},
        ),
    )


def test_llm_client_stream_returns_text_deltas() -> None:
    """Yield streaming text chunks from the SDK stream."""

    fake_client = FakeOpenAIClient()
    llm = LLMClient(
        config=LLMConfig(model="gpt-4o-mini"),
        client=fake_client,
    )

    chunks = list(llm.stream([LLMMessage(role="user", content="ping")]))

    assert chunks == ["hel", "lo"]


def test_llm_config_from_env_reads_common_fields(monkeypatch: MonkeyPatch) -> None:
    """Load model settings from environment variables."""

    monkeypatch.setenv("LLM_MODEL", "local-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("LLM_API_KEY", "token")
    monkeypatch.setenv("LLM_TIMEOUT", "45")
    monkeypatch.setenv("LLM_PROVIDER", "local")

    config = LLMConfig.from_env()

    assert config.model == "local-model"
    assert config.base_url == "http://localhost:1234/v1"
    assert config.api_key == "token"
    assert config.timeout == 45.0
    assert config.provider == "local"


def test_llm_config_defaults_api_key_for_local_compatible_endpoints() -> None:
    """Return a placeholder API key when none is configured."""

    config = LLMConfig(model="qwen-local", base_url="http://localhost:11434/v1")

    assert config.resolved_api_key() == "EMPTY"
