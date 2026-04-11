"""Define a minimal chat-only agent implementation."""

from __future__ import annotations

from hello_agents.agent import Agent
from hello_agents.llm.client import LLMClient
from hello_agents.llm.types import LLMMessage
from hello_agents.memory import MemoryScope
from hello_agents.memory.base import Memory
from hello_agents.tools.base import ToolResult
from hello_agents.tools.registry import ToolRegistry


class ChatAgent(Agent):
    """Implement an agent whose only capability is chatting with an LLM."""

    def __init__(
        self,
        name: str,
        llm: LLMClient,
        tools: ToolRegistry | None = None,
        use_tools: bool = False,
        memory: Memory | None = None,
        *,
        system_prompt: str = "You are a helpful assistant.",
        max_tool_rounds: int = 3,
    ) -> None:
        """Store the system prompt required for a simple single-turn run."""

        super().__init__(
            name=name,
            llm=llm,
            tools=tools,
            use_tools=use_tools,
            memory=memory,
        )
        self.system_prompt = system_prompt
        self.max_tool_rounds = max_tool_rounds

    def run(
        self,
        message: str,
        *,
        memory_scope: MemoryScope | None = None,
    ) -> str:
        """Run a chat turn and execute tool calls when the model requests them."""

        effective_message = self.build_effective_message(
            message,
            memory_scope=memory_scope,
        )
        self.logger.info(
            "Starting chat run use_tools=%s message_length=%s memory_enabled=%s",
            self.use_tools,
            len(message),
            self.memory is not None and memory_scope is not None,
        )
        messages = [
            LLMMessage(role="system", content=self.system_prompt),
            LLMMessage(role="user", content=effective_message),
        ]
        tool_results: list[ToolResult] = []

        try:
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
                    self.persist_memory_turn(
                        memory_scope=memory_scope,
                        message=message,
                        response=response.content,
                        tool_results=tuple(tool_results),
                        success=True,
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
                    tool_results.append(tool_result)
                    self.logger.info(
                        (
                            "Tool call completed id=%s name=%s "
                            "success=%s content_length=%s"
                        ),
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
        except Exception as exc:
            self.persist_memory_turn(
                memory_scope=memory_scope,
                message=message,
                response=str(exc),
                tool_results=tuple(tool_results),
                success=False,
            )
            raise

        self.logger.warning(
            "Tool calling exceeded max rounds=%s",
            self.max_tool_rounds,
        )
        self.persist_memory_turn(
            memory_scope=memory_scope,
            message=message,
            response="Tool calling exceeded the maximum number of rounds.",
            tool_results=tuple(tool_results),
            success=False,
        )
        raise RuntimeError("Tool calling exceeded the maximum number of rounds.")
