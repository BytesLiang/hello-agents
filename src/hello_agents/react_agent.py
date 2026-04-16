"""Define a ReAct-style agent implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError

from hello_agents.agent import Agent
from hello_agents.context import ContextEngine
from hello_agents.llm.client import LLMClient
from hello_agents.llm.types import LLMMessage
from hello_agents.memory import MemoryScope
from hello_agents.memory.base import Memory
from hello_agents.rag.retriever import RagRetriever
from hello_agents.tools.base import ToolResult
from hello_agents.tools.registry import ToolRegistry

REACT_OUTPUT_CONTRACT = """
Follow a strict ReAct loop.

At each step:
1. Think about the next best step.
2. If a tool is needed, choose exactly one tool action.
3. If the answer is ready, return the final answer.

You must respond with valid JSON only.
Do not return Markdown.
Do not wrap the JSON in code fences.
Do not add any text before or after the JSON.

If a tool is needed, respond with:
{
  "thought": "brief reasoning for the next step",
  "action": "tool_name",
  "action_input": {
    "param_name": "value"
  }
}

If the answer is ready, respond with:
{
  "thought": "brief reasoning for the conclusion",
  "final_answer": "the final answer for the user"
}

Rules:
- action_input must be a JSON object.
- final_answer must only be used when you are done.
- Never output both action and final_answer in the same response.
- Base each next step on the latest observation in the scratchpad.
""".strip()


@dataclass(slots=True, frozen=True)
class ReActStep:
    """Represent a parsed ReAct step emitted by the model."""

    thought: str
    action: str | None = None
    action_input: dict[str, object] | None = None
    final_answer: str | None = None


class ReActAgent(Agent):
    """Implement a classic ReAct loop over the framework tool registry."""

    def __init__(
        self,
        name: str,
        llm: LLMClient,
        tools: ToolRegistry | None = None,
        use_tools: bool = True,
        memory: Memory | None = None,
        rag: RagRetriever | None = None,
        context_engine: ContextEngine | None = None,
        *,
        system_prompt: str | None = None,
        max_steps: int = 5,
    ) -> None:
        """Store prompt and loop limits for the ReAct workflow."""

        super().__init__(
            name=name,
            llm=llm,
            tools=tools,
            use_tools=use_tools,
            memory=memory,
            rag=rag,
            context_engine=context_engine,
        )
        self.system_prompt = _build_react_system_prompt(system_prompt)
        self.max_steps = max_steps

    def run(
        self,
        message: str,
        *,
        memory_scope: MemoryScope | None = None,
    ) -> str:
        """Run the ReAct reasoning loop until a final answer is produced."""

        self.logger.info(
            "Starting ReAct run use_tools=%s message_length=%s memory_enabled=%s",
            self.use_tools,
            len(message),
            self.memory is not None and memory_scope is not None,
        )
        scratchpad: list[str] = []
        tool_results: list[ToolResult] = []

        try:
            for step_index in range(1, self.max_steps + 1):
                effective_message = self.build_effective_message(
                    message,
                    memory_scope=memory_scope,
                    tool_results=tuple(tool_results),
                )
                prompt = self._build_prompt(
                    message=effective_message,
                    scratchpad=scratchpad,
                )
                self.logger.info("Sending ReAct step=%s", step_index)
                response = self.llm.chat(
                    [
                        LLMMessage(role="system", content=self.system_prompt),
                        LLMMessage(role="user", content=prompt),
                    ]
                )
                self.logger.info(
                    "Received ReAct response step=%s response_length=%s",
                    step_index,
                    len(response.content),
                )
                self.logger.debug(
                    "Raw ReAct response step=%s content=%r",
                    step_index,
                    response.content,
                )

                parsed_step = _parse_react_step(response.content)
                self.logger.info(
                    "Parsed ReAct step=%s action=%s has_final=%s",
                    step_index,
                    parsed_step.action,
                    parsed_step.final_answer is not None,
                )

                scratchpad.append(f"Thought: {parsed_step.thought}")

                if parsed_step.final_answer is not None:
                    self.logger.info(
                        "Completing ReAct run step=%s response_length=%s",
                        step_index,
                        len(parsed_step.final_answer),
                    )
                    self.persist_memory_turn(
                        memory_scope=memory_scope,
                        message=message,
                        response=parsed_step.final_answer,
                        tool_results=tuple(tool_results),
                        success=True,
                    )
                    return parsed_step.final_answer

                if parsed_step.action is None or parsed_step.action_input is None:
                    raise ValueError(
                        "ReAct response must include either action or final_answer."
                    )

                if not self.use_tools:
                    raise RuntimeError("Tools are disabled for this agent.")

                self.logger.info(
                    "Executing ReAct action step=%s action=%s",
                    step_index,
                    parsed_step.action,
                )
                tool_result = self.execute_tool(
                    parsed_step.action,
                    parsed_step.action_input,
                )
                tool_results.append(tool_result)
                scratchpad.append(f"Action: {parsed_step.action}")
        except Exception as exc:
            self.persist_memory_turn(
                memory_scope=memory_scope,
                message=message,
                response=str(exc),
                tool_results=tuple(tool_results),
                success=False,
            )
            raise

        self.logger.warning("ReAct agent exceeded max steps=%s", self.max_steps)
        self.persist_memory_turn(
            memory_scope=memory_scope,
            message=message,
            response="ReAct agent exceeded the maximum number of steps.",
            tool_results=tuple(tool_results),
            success=False,
        )
        raise RuntimeError("ReAct agent exceeded the maximum number of steps.")

    def _build_prompt(self, *, message: str, scratchpad: list[str]) -> str:
        """Build the current ReAct prompt with tools and prior observations."""

        tool_descriptions = self.describe_tools()
        scratchpad_text = "\n".join(scratchpad) if scratchpad else "None yet."
        return (
            f"Question: {message}\n\n"
            "Available tools:\n"
            f"{json.dumps(tool_descriptions, ensure_ascii=False, indent=2)}\n\n"
            f"Scratchpad:\n{scratchpad_text}\n"
        )


def _parse_react_step(content: str) -> ReActStep:
    """Parse a ReAct model response from JSON or fenced JSON."""

    parsed = _load_react_json(content)
    if not isinstance(parsed, dict):
        raise ValueError("ReAct response must be a JSON object.")

    thought = parsed.get("thought", "")
    action = parsed.get("action") or parsed.get("tool")
    action_input = parsed.get("action_input")
    if action_input is None:
        action_input = parsed.get("actionInput") or parsed.get("tool_input")
    final_answer = (
        parsed.get("final_answer")
        or parsed.get("answer")
        or parsed.get("final")
        or parsed.get("response")
    )

    if not isinstance(thought, str):
        raise ValueError("ReAct thought must be a string.")
    if action is not None and not isinstance(action, str):
        raise ValueError("ReAct action must be a string.")
    if isinstance(action_input, str) and action_input:
        parsed_action_input = json.loads(action_input)
        if isinstance(parsed_action_input, dict):
            action_input = parsed_action_input
    if action_input is not None and not isinstance(action_input, dict):
        raise ValueError("ReAct action_input must be an object.")
    if final_answer is not None and not isinstance(final_answer, str):
        raise ValueError("ReAct final_answer must be a string.")

    if not thought:
        if final_answer is not None:
            thought = "Model returned a final answer without an explicit thought."
        elif action is not None:
            thought = "Model requested an action without an explicit thought."
        else:
            raise ValueError(
                "ReAct response requires thought, action, or final_answer content."
            )

    return ReActStep(
        thought=thought,
        action=action,
        action_input=action_input,
        final_answer=final_answer,
    )


def _load_react_json(content: str) -> dict[str, object]:
    """Extract and parse a JSON object from a model response."""

    normalized = content.strip()
    if normalized.startswith("```"):
        normalized = normalized.removeprefix("```json").removeprefix("```")
        normalized = normalized.removesuffix("```").strip()

    try:
        parsed = json.loads(normalized)
    except JSONDecodeError:
        start = normalized.find("{")
        end = normalized.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise ValueError("ReAct response must contain a JSON object.") from None
        parsed = json.loads(normalized[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("ReAct response must be a JSON object.")
    return parsed


def _build_react_system_prompt(system_prompt: str | None) -> str:
    """Build the effective system prompt with the fixed ReAct contract."""

    if system_prompt is None or not system_prompt.strip():
        return REACT_OUTPUT_CONTRACT
    return f"{system_prompt.strip()}\n\n{REACT_OUTPUT_CONTRACT}"
