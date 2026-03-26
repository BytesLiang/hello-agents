"""Top-level package for the hello_agents project."""

from hello_agents.chat_agent import ChatAgent
from hello_agents.tools import (
    TavilySearchTool,
    Tool,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    ToolSchema,
    build_default_tool_registry,
)

__all__ = [
    "ChatAgent",
    "TavilySearchTool",
    "Tool",
    "ToolParameter",
    "ToolRegistry",
    "ToolResult",
    "ToolSchema",
    "build_default_tool_registry",
    "__version__",
]

__version__ = "0.1.0"
