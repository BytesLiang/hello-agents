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

        self.logger.info(
            "Starting chat run use_tools=%s message_length=%s",
            self.use_tools,
            len(message),
        )
        messages = [
            LLMMessage(role="system", content=self.system_prompt),
            LLMMessage(role="user", content=message),
        ]

        for round_index in range(self.max_tool_rounds + 1):
            self.logger.info(
                "Sending LLM request round=%s tool_count=%s",
                round_index,
                len(self.describe_tools()),
            )
            response = self.llm.chat(
                messages,
                tools=self.describe_tools(),
            )
            self.logger.info(
                "Received LLM response round=%s finish_reason=%s tool_calls=%s",
                round_index,
                response.finish_reason,
                len(response.tool_calls),
            )
            if not self.use_tools or not response.tool_calls:
                self.logger.info(
                    "Completing chat run round=%s response_length=%s",
                    round_index,
                    len(response.content),
                )
                return response.content

            messages.append(
                LLMMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            for tool_call in response.tool_calls:
                self.logger.info(
                    "Handling tool call id=%s name=%s",
                    tool_call.id,
                    tool_call.name,
                )
                tool_result = self.execute_tool(tool_call.name, tool_call.arguments)
                self.logger.info(
                    "Tool call completed id=%s name=%s success=%s content_length=%s",
                    tool_call.id,
                    tool_call.name,
                    tool_result.success,
                    len(tool_result.content),
                )
                messages.append(
                    LLMMessage(
                        role="tool",
                        content=tool_result.content,
                        tool_call_id=tool_call.id,
                    )
                )

        self.logger.warning(
            "Tool calling exceeded max rounds=%s",
            self.max_tool_rounds,
        )
        raise RuntimeError("Tool calling exceeded the maximum number of rounds.")
