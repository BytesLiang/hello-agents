"""Tests for the ReAct agent implementation."""

from __future__ import annotations

import logging
from typing import cast

import pytest

from hello_agents.llm.client import LLMClient
from hello_agents.llm.types import LLMMessage, LLMResponse
from hello_agents.react_agent import ReActAgent
from hello_agents.tools import Tool, ToolParameter, ToolRegistry, ToolResult, ToolSchema


class FakeLLMClient:
    """Provide deterministic ReAct responses for tests."""

    def __init__(self, responses: list[str]) -> None:
        """Store queued model outputs."""

        self.responses = list(responses)
        self.calls: list[list[LLMMessage]] = []

    def chat(self, messages: list[LLMMessage]) -> LLMResponse:
        """Return queued responses in order."""

        self.calls.append(list(messages))
        return LLMResponse(model="fake-model", content=self.responses.pop(0))


class EchoTool(Tool):
    """Provide a deterministic tool for ReAct tests."""

    def __init__(self) -> None:
        """Initialize tool metadata."""

        super().__init__(
            name="echo",
            description="Echo the provided text.",
            schema=ToolSchema(
                parameters=(
                    ToolParameter(name="text", description="Text to echo back."),
                )
            ),
        )

    def execute(self, payload: dict[str, object]) -> ToolResult:
        """Return the provided text as the observation."""

        return ToolResult(tool_name=self.name, content=str(payload["text"]))


def test_react_agent_returns_final_answer_without_tools() -> None:
    """Verify the agent can finish directly when the model returns a final answer."""

    llm = FakeLLMClient(
        responses=[
            '{"thought":"I already know the answer.","final_answer":"done"}',
        ]
    )
    agent = ReActAgent(
        name="react-demo",
        llm=cast(LLMClient, llm),
        tools=ToolRegistry(),
        use_tools=False,
    )

    result = agent.run("Answer directly.")

    assert result == "done"
    assert len(llm.calls) == 1


def test_react_agent_executes_tool_and_finishes() -> None:
    """Verify the agent follows the ReAct loop with a tool observation."""

    llm = FakeLLMClient(
        responses=[
            (
                '{"thought":"I should use echo.","action":"echo",'
                '"action_input":{"text":"observed value"}}'
            ),
            '{"thought":"Now I can answer.","final_answer":"final answer"}',
        ]
    )
    tools = ToolRegistry()
    tools.register(EchoTool())
    agent = ReActAgent(
        name="react-demo",
        llm=cast(LLMClient, llm),
        tools=tools,
        use_tools=True,
    )

    result = agent.run("Use the tool.")

    assert result == "final answer"
    assert len(llm.calls) == 2
    second_prompt = llm.calls[1][1].content
    assert "Observation: observed value" in second_prompt


def test_react_agent_rejects_tool_use_when_disabled() -> None:
    """Verify tool actions fail if tools are disabled."""

    llm = FakeLLMClient(
        responses=[
            (
                '{"thought":"I should use echo.","action":"echo",'
                '"action_input":{"text":"value"}}'
            ),
        ]
    )
    agent = ReActAgent(
        name="react-demo",
        llm=cast(LLMClient, llm),
        tools=ToolRegistry(),
        use_tools=False,
    )

    with pytest.raises(RuntimeError, match="disabled"):
        agent.run("Use the tool.")


def test_react_agent_limits_steps() -> None:
    """Verify the agent stops after the configured maximum number of steps."""

    llm = FakeLLMClient(
        responses=[
            ('{"thought":"keep going","action":"echo","action_input":{"text":"one"}}'),
            ('{"thought":"still going","action":"echo","action_input":{"text":"two"}}'),
        ]
    )
    tools = ToolRegistry()
    tools.register(EchoTool())
    agent = ReActAgent(
        name="react-demo",
        llm=cast(LLMClient, llm),
        tools=tools,
        use_tools=True,
        max_steps=1,
    )

    with pytest.raises(RuntimeError, match="maximum number of steps"):
        agent.run("Loop.")


def test_react_agent_emits_logs(caplog: pytest.LogCaptureFixture) -> None:
    """Verify the ReAct loop logs critical runtime events."""

    caplog.set_level(logging.INFO)
    llm = FakeLLMClient(
        responses=[
            (
                '{"thought":"I should use echo.","action":"echo",'
                '"action_input":{"text":"logged"}}'
            ),
            '{"thought":"Now done.","final_answer":"done"}',
        ]
    )
    tools = ToolRegistry()
    tools.register(EchoTool())
    agent = ReActAgent(
        name="react-demo",
        llm=cast(LLMClient, llm),
        tools=tools,
        use_tools=True,
    )

    agent.run("Log this.")

    log_text = caplog.text
    assert "Starting ReAct run" in log_text
    assert "Sending ReAct step=1" in log_text
    assert "Executing ReAct action step=1 action=echo" in log_text
    assert "Completing ReAct run step=2" in log_text


def test_react_agent_accepts_final_answer_without_thought() -> None:
    """Verify parser tolerates final answers that omit thought."""

    llm = FakeLLMClient(
        responses=[
            '{"final_answer":"done"}',
        ]
    )
    agent = ReActAgent(
        name="react-demo",
        llm=cast(LLMClient, llm),
        tools=ToolRegistry(),
        use_tools=False,
    )

    result = agent.run("Answer directly.")

    assert result == "done"


def test_react_agent_accepts_embedded_json_response() -> None:
    """Verify parser can extract JSON wrapped in extra text."""

    llm = FakeLLMClient(
        responses=[
            'Here is the result:\n```json\n{"final_answer":"done"}\n```',
        ]
    )
    agent = ReActAgent(
        name="react-demo",
        llm=cast(LLMClient, llm),
        tools=ToolRegistry(),
        use_tools=False,
    )

    result = agent.run("Answer directly.")

    assert result == "done"
