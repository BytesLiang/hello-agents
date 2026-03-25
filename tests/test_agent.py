"""Tests for the agent module."""

from typing import cast

import pytest

from hello_agents.agent import Agent
from hello_agents.llm.client import LLMClient
from hello_agents.tools import Tool, ToolParameter, ToolRegistry, ToolResult, ToolSchema


class FakeLLMClient:
    """Provide a minimal stand-in for the shared LLM dependency."""


class EchoTool(Tool):
    """Provide a minimal concrete tool for agent tests."""

    def __init__(self) -> None:
        """Initialize the tool metadata."""

        super().__init__(
            name="echo",
            description="Echo the input payload.",
            schema=ToolSchema(
                parameters=(
                    ToolParameter(
                        name="text",
                        description="Text returned by the tool.",
                    ),
                )
            ),
        )

    def execute(self, payload: dict[str, object]) -> ToolResult:
        """Return the provided text field."""

        return ToolResult(tool_name=self.name, content=str(payload["text"]))


class DemoAgent(Agent):
    """Provide a concrete agent implementation for tests."""

    def __init__(
        self,
        name: str,
        llm: LLMClient,
        tools: ToolRegistry | None = None,
    ) -> None:
        """Initialize the concrete agent with the shared dependencies."""

        super().__init__(name=name, llm=llm, tools=tools)

    def run(self, message: str) -> str:
        """Return a predictable status message."""

        return f"Agent {self.name} received: {message}"


def test_agent_is_abstract() -> None:
    """Verify the abstract base class cannot be instantiated directly."""

    with pytest.raises(TypeError):
        Agent(
            name="demo",
            llm=cast(LLMClient, FakeLLMClient()),
        )  # type: ignore[abstract]


def test_concrete_agent_implements_shared_interface() -> None:
    """Verify concrete agents inherit the common attributes and behavior."""

    llm = cast(LLMClient, FakeLLMClient())
    tools = ToolRegistry()
    tools.register(EchoTool())
    agent = DemoAgent(name="demo", llm=llm, tools=tools)

    assert agent.name == "demo"
    assert agent.llm is llm
    assert agent.tools is tools
    assert agent.describe_tools() == [tools.get("echo").to_openai_tool()]
    assert agent.execute_tool("echo", {"text": "tool-call"}) == ToolResult(
        tool_name="echo",
        content="tool-call",
    )
    assert agent.run("hello") == "Agent demo received: hello"
