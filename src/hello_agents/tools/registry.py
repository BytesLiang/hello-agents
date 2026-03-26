"""Provide a registry for tool registration, discovery, and execution."""

from __future__ import annotations

from hello_agents.tools.base import Tool, ToolResult
from hello_agents.tools.tavily import TavilySearchTool


class ToolRegistry:
    """Manage tool instances behind a consistent registration interface."""

    def __init__(self) -> None:
        """Initialize an empty in-memory tool registry."""

        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool instance by its unique name."""

        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return a registered tool by name."""

        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Tool '{name}' is not registered.") from exc

    def list_tools(self) -> list[Tool]:
        """Return all registered tools in registration order."""

        return list(self._tools.values())

    def describe_tools(self) -> list[dict[str, object]]:
        """Return the registered tools as OpenAI-compatible definitions."""

        return [tool.to_openai_tool() for tool in self._tools.values()]

    def execute(self, name: str, payload: dict[str, object]) -> ToolResult:
        """Execute a registered tool by name."""

        tool = self.get(name)
        tool.schema.validate(payload)
        return tool.execute(payload)


def build_default_tool_registry() -> ToolRegistry:
    """Build the default registry used by agents in simple setups."""

    registry = ToolRegistry()
    registry.register(TavilySearchTool())
    return registry
