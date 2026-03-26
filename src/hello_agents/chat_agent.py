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
        use_tools: bool = False,
        *,
        system_prompt: str = "You are a helpful assistant.",
        max_tool_rounds: int = 3,
    ) -> None:
        """Store the system prompt required for a simple single-turn run."""

        super().__init__(name=name, llm=llm, tools=tools, use_tools=use_tools)
        self.system_prompt = system_prompt
        self.max_tool_rounds = max_tool_rounds

    def run(self, message: str) -> str:
        """Run a chat turn and execute tool calls when the model requests them."""

        messages = [
            LLMMessage(role="system", content=self.system_prompt),
            LLMMessage(role="user", content=message),
        ]

        for _ in range(self.max_tool_rounds + 1):
            response = self.llm.chat(
                messages,
                tools=self.describe_tools(),
            )
            if not self.use_tools or not response.tool_calls:
                return response.content

            messages.append(
                LLMMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            for tool_call in response.tool_calls:
                tool_result = self.execute_tool(tool_call.name, tool_call.arguments)
                messages.append(
                    LLMMessage(
                        role="tool",
                        content=tool_result.content,
                        tool_call_id=tool_call.id,
                    )
                )

        raise RuntimeError("Tool calling exceeded the maximum number of rounds.")
