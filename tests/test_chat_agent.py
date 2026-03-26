"""Tests for the chat-only agent implementation."""

from __future__ import annotations

from typing import cast

from hello_agents.chat_agent import ChatAgent
from hello_agents.llm.client import LLMClient
from hello_agents.llm.types import LLMMessage, LLMResponse
from hello_agents.tools import (
    TavilySearchTool,
    ToolRegistry,
    build_default_tool_registry,
)


class FakeLLMClient:
    """Provide a deterministic LLM client test double."""

    def __init__(self) -> None:
        """Initialize captured request state."""

        self.messages: list[LLMMessage] | None = None
        self.tools: list[dict[str, object]] | None = None

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[dict[str, object]] | None = None,
    ) -> LLMResponse:
        """Capture the chat payload and return a fixed response."""

        self.messages = messages
        self.tools = tools
        return LLMResponse(model="fake-model", content="hello from llm")


def test_chat_agent_sends_system_and_user_messages() -> None:
    """Verify the chat agent only delegates a single turn to the LLM."""

    llm = FakeLLMClient()
    tools = ToolRegistry()
    agent = ChatAgent(
        name="chat-demo",
        llm=cast(LLMClient, llm),
        tools=tools,
        system_prompt="You are concise.",
    )

    result = agent.run("Say hello.")

    assert result == "hello from llm"
    assert agent.tools is tools
    assert llm.tools == []
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


def test_chat_agent_passes_tools_when_enabled() -> None:
    """Verify the agent exposes tool definitions to the LLM when enabled."""

    llm = FakeLLMClient()
    tools = ToolRegistry()
    tools.register(TavilySearchTool(client=object()))
    agent = ChatAgent(
        name="chat-demo",
        llm=cast(LLMClient, llm),
        tools=tools,
        use_tools=True,
    )

    agent.run("Search the web.")

    assert llm.tools == tools.describe_tools()


def test_build_default_tool_registry_registers_tavily_tool() -> None:
    """Verify the default registry includes the Tavily search tool."""

    registry = build_default_tool_registry()

    assert isinstance(registry.get("tavily_search"), TavilySearchTool)
