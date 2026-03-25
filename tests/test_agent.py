"""Tests for the agent module."""

from typing import cast

import pytest

from hello_agents.agent import Agent
from hello_agents.llm.client import LLMClient


class FakeLLMClient:
    """Provide a minimal stand-in for the shared LLM dependency."""


class DemoAgent(Agent):
    """Provide a concrete agent implementation for tests."""

    def __init__(self, name: str, llm: LLMClient) -> None:
        """Initialize the concrete agent with the shared dependencies."""

        super().__init__(name=name, llm=llm)

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
    agent = DemoAgent(name="demo", llm=llm)

    assert agent.name == "demo"
    assert agent.llm is llm
    assert agent.run("hello") == "Agent demo received: hello"
