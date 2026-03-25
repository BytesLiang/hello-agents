"""Tests for the chat-only agent implementation."""

from __future__ import annotations

from typing import cast

from hello_agents.chat_agent import ChatAgent
from hello_agents.llm.client import LLMClient
from hello_agents.llm.types import LLMMessage, LLMResponse


class FakeLLMClient:
    """Provide a deterministic LLM client test double."""

    def __init__(self) -> None:
        """Initialize captured request state."""

        self.messages: list[LLMMessage] | None = None

    def chat(self, messages: list[LLMMessage]) -> LLMResponse:
        """Capture the chat payload and return a fixed response."""

        self.messages = messages
        return LLMResponse(model="fake-model", content="hello from llm")


def test_chat_agent_sends_system_and_user_messages() -> None:
    """Verify the chat agent only delegates a single turn to the LLM."""

    llm = FakeLLMClient()
    agent = ChatAgent(
        name="chat-demo",
        llm=cast(LLMClient, llm),
        system_prompt="You are concise.",
    )

    result = agent.run("Say hello.")

    assert result == "hello from llm"
    assert llm.messages == [
        LLMMessage(role="system", content="You are concise."),
        LLMMessage(role="user", content="Say hello."),
    ]


def test_chat_agent_uses_default_prompts() -> None:
    """Verify the chat agent uses the runtime message with default system text."""

    llm = FakeLLMClient()
    agent = ChatAgent(name="chat-demo", llm=cast(LLMClient, llm))

    agent.run("Hello.")

    assert llm.messages == [
        LLMMessage(role="system", content="You are a helpful assistant."),
        LLMMessage(role="user", content="Hello."),
    ]
