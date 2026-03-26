"""Tests for the chat-only agent implementation."""

from __future__ import annotations

from typing import cast

from hello_agents.chat_agent import ChatAgent
from hello_agents.llm.client import LLMClient
from hello_agents.llm.types import LLMMessage, LLMResponse, LLMToolCall
from hello_agents.tools import (
    TavilySearchTool,
    Tool,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    ToolSchema,
    build_default_tool_registry,
)


class FakeLLMClient:
    """Provide a deterministic LLM client test double."""

    def __init__(self) -> None:
        """Initialize captured request state."""

        self.calls: list[dict[str, object]] = []
        self.tools: list[dict[str, object]] | None = None
        self.responses: list[LLMResponse] = []

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[dict[str, object]] | None = None,
    ) -> LLMResponse:
        """Capture the chat payload and return a fixed response."""

        self.calls.append({"messages": list(messages), "tools": tools})
        self.tools = tools
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(model="fake-model", content="hello from llm")


class EchoTool(Tool):
    """Provide a deterministic tool for chat-agent tool-call tests."""

    def __init__(self) -> None:
        """Initialize the tool metadata."""

        super().__init__(
            name="echo",
            description="Echo the provided text.",
            schema=ToolSchema(
                parameters=(
                    ToolParameter(
                        name="text",
                        description="Text to echo.",
                    ),
                )
            ),
        )

    def execute(self, payload: dict[str, object]) -> ToolResult:
        """Return the payload text as the tool result."""

        return ToolResult(tool_name=self.name, content=str(payload["text"]))


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
    assert llm.calls[0]["messages"] == [
        LLMMessage(role="system", content="You are concise."),
        LLMMessage(role="user", content="Say hello."),
    ]


def test_chat_agent_uses_default_prompts() -> None:
    """Verify the chat agent uses the runtime message with default system text."""

    llm = FakeLLMClient()
    agent = ChatAgent(name="chat-demo", llm=cast(LLMClient, llm))

    agent.run("Hello.")

    assert llm.calls[0]["messages"] == [
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


def test_chat_agent_executes_tool_calls_and_continues_conversation() -> None:
    """Verify ChatAgent executes requested tools and sends tool results back."""

    llm = FakeLLMClient()
    llm.responses = [
        LLMResponse(
            model="fake-model",
            content="",
            tool_calls=(
                LLMToolCall(
                    id="call_1",
                    name="echo",
                    arguments={"text": "tool output"},
                ),
            ),
        ),
        LLMResponse(model="fake-model", content="final answer"),
    ]
    tools = ToolRegistry()
    tools.register(EchoTool())
    agent = ChatAgent(
        name="chat-demo",
        llm=cast(LLMClient, llm),
        tools=tools,
        use_tools=True,
    )

    result = agent.run("Use the tool.")

    assert result == "final answer"
    assert len(llm.calls) == 2
    second_messages = llm.calls[1]["messages"]
    assert second_messages == [
        LLMMessage(role="system", content="You are a helpful assistant."),
        LLMMessage(role="user", content="Use the tool."),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls=(
                LLMToolCall(
                    id="call_1",
                    name="echo",
                    arguments={"text": "tool output"},
                ),
            ),
        ),
        LLMMessage(
            role="tool",
            content="tool output",
            tool_call_id="call_1",
        ),
    ]


def test_chat_agent_limits_tool_rounds() -> None:
    """Verify the agent stops repeated tool calling loops."""

    llm = FakeLLMClient()
    llm.responses = [
        LLMResponse(
            model="fake-model",
            content="",
            tool_calls=(
                LLMToolCall(
                    id="call_1",
                    name="echo",
                    arguments={"text": "loop"},
                ),
            ),
        ),
        LLMResponse(
            model="fake-model",
            content="",
            tool_calls=(
                LLMToolCall(
                    id="call_2",
                    name="echo",
                    arguments={"text": "loop"},
                ),
            ),
        ),
    ]
    tools = ToolRegistry()
    tools.register(EchoTool())
    agent = ChatAgent(
        name="chat-demo",
        llm=cast(LLMClient, llm),
        tools=tools,
        use_tools=True,
        max_tool_rounds=1,
    )

    import pytest

    with pytest.raises(RuntimeError, match="maximum number of rounds"):
        agent.run("Loop.")
