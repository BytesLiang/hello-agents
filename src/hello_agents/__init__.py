"""Top-level package for the hello_agents project."""

from hello_agents.chat_agent import ChatAgent
from hello_agents.tools import Tool, ToolParameter, ToolRegistry, ToolResult, ToolSchema

__all__ = [
    "ChatAgent",
    "Tool",
    "ToolParameter",
    "ToolRegistry",
    "ToolResult",
    "ToolSchema",
    "__version__",
]

__version__ = "0.1.0"
