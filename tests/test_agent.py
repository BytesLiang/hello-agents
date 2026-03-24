"""Tests for the agent module."""

from hello_agents.agent import Agent


def test_agent_run_returns_ready_message() -> None:
    """Verify the agent returns a predictable ready message."""

    agent = Agent(name="demo")

    assert agent.run() == "Agent demo is ready."
