"""Define a minimal chat-only agent implementation."""

from __future__ import annotations

from hello_agents.agent import Agent
from hello_agents.llm.client import LLMClient
from hello_agents.llm.types import LLMMessage
from hello_agents.tools.registry import ToolRegistry


class ChatAgent(Agent):
    """Implement an agent whose only capability is chatting with an LLM."""

    def __init__(
        self,
        name: str,
        llm: LLMClient,
        tools: ToolRegistry | None = None,
        *,
        system_prompt: str = "You are a helpful assistant.",
    ) -> None:
        """Store the system prompt required for a simple single-turn run."""

        super().__init__(name=name, llm=llm, tools=tools)
        self.system_prompt = system_prompt

    def run(self, message: str) -> str:
        """Send a single runtime message to the shared LLM client."""

        response = self.llm.chat(
            [
                LLMMessage(role="system", content=self.system_prompt),
                LLMMessage(role="user", content=message),
            ]
        )
        return response.content
