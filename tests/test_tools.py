"""Tests for the tool abstraction and registry."""

from __future__ import annotations

import pytest

from hello_agents.tools import (
    TavilySearchTool,
    Tool,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    ToolSchema,
)


class EchoTool(Tool):
    """Provide a simple concrete tool for registry tests."""

    def __init__(self) -> None:
        """Initialize the tool metadata."""

        super().__init__(
            name="echo",
            description="Return the input unchanged.",
            schema=ToolSchema(
                parameters=(
                    ToolParameter(
                        name="text",
                        description="Text echoed back to the caller.",
                        value_type="string",
                    ),
                )
            ),
        )

    def execute(self, payload: dict[str, object]) -> ToolResult:
        """Return the input as the tool result content."""

        return ToolResult(
            tool_name=self.name,
            content=str(payload["text"]),
        )


def test_tool_is_abstract() -> None:
    """Verify the tool base class cannot be instantiated directly."""

    with pytest.raises(TypeError):
        Tool(name="base", description="abstract")  # type: ignore[abstract]


def test_registry_registers_and_discovers_tools() -> None:
    """Verify registered tools can be listed and retrieved."""

    registry = ToolRegistry()
    tool = EchoTool()

    registry.register(tool)

    assert registry.get("echo") is tool
    assert registry.list_tools() == [tool]
    assert registry.describe_tools() == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Return the input unchanged.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text echoed back to the caller.",
                        }
                    },
                    "additionalProperties": False,
                    "required": ["text"],
                },
            },
        }
    ]


def test_registry_rejects_duplicate_tool_names() -> None:
    """Verify tool names stay unique within the registry."""

    registry = ToolRegistry()
    registry.register(EchoTool())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(EchoTool())


def test_registry_executes_registered_tool() -> None:
    """Verify the registry delegates execution to the target tool."""

    registry = ToolRegistry()
    registry.register(EchoTool())

    result = registry.execute("echo", {"text": "hello tool"})

    assert result == ToolResult(tool_name="echo", content="hello tool")


def test_registry_raises_for_unknown_tools() -> None:
    """Verify missing tools fail with a clear lookup error."""

    registry = ToolRegistry()

    with pytest.raises(KeyError, match="not registered"):
        registry.get("missing")


def test_registry_validates_required_parameters() -> None:
    """Verify schema validation runs before tool execution."""

    registry = ToolRegistry()
    registry.register(EchoTool())

    with pytest.raises(ValueError, match="Missing required parameter 'text'"):
        registry.execute("echo", {})


def test_registry_validates_parameter_types() -> None:
    """Verify schema type mismatches fail with a clear error."""

    registry = ToolRegistry()
    registry.register(EchoTool())

    with pytest.raises(ValueError, match="must be of type 'string'"):
        registry.execute("echo", {"text": 123})


def test_registry_rejects_unexpected_parameters() -> None:
    """Verify schema validation rejects undeclared input keys."""

    registry = ToolRegistry()
    registry.register(EchoTool())

    with pytest.raises(ValueError, match="Unexpected parameter"):
        registry.execute("echo", {"text": "hello", "extra": "value"})


def test_tool_schema_exports_openai_compatible_json_schema() -> None:
    """Verify tool schemas serialize into OpenAI-compatible JSON Schema."""

    schema = ToolSchema(
        parameters=(
            ToolParameter(
                name="city",
                description="City name to query.",
                value_type="string",
            ),
            ToolParameter(
                name="unit",
                description="Temperature unit.",
                value_type="string",
                required=False,
                enum=("celsius", "fahrenheit"),
            ),
        )
    )

    assert schema.to_json_schema() == {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name to query.",
            },
            "unit": {
                "type": "string",
                "description": "Temperature unit.",
                "enum": ["celsius", "fahrenheit"],
            },
        },
        "additionalProperties": False,
        "required": ["city"],
    }


def test_tool_exports_openai_function_definition() -> None:
    """Verify tools can be exported as OpenAI function definitions."""

    tool = EchoTool()

    assert tool.to_openai_tool() == {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Return the input unchanged.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text echoed back to the caller.",
                    }
                },
                "additionalProperties": False,
                "required": ["text"],
            },
        },
    }


def test_tavily_tool_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Tavily search fails clearly when no API key is configured."""

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    tool = TavilySearchTool()

    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        tool.execute({"query": "hello"})


def test_tavily_tool_formats_search_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Tavily search results normalize into a ToolResult."""

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    class FakeTavilyClient:
        def search(self, *, query: str, max_results: object) -> dict[str, object]:
            assert query == "python agent framework"
            assert max_results == 2
            return {
                "answer": "A framework helps orchestrate LLMs and tools.",
                "results": [
                    {
                        "title": "Example Result",
                        "url": "https://example.com/result",
                        "content": "Framework overview",
                    }
                ],
            }

    tool = TavilySearchTool(client=FakeTavilyClient())
    result = tool.execute(
        {
            "query": "python agent framework",
            "max_results": 2,
        }
    )

    assert result.tool_name == "tavily_search"
    assert "Search query: python agent framework" in result.content
    assert "Answer: A framework helps orchestrate LLMs and tools." in result.content
    assert "1. Example Result" in result.content
    assert result.metadata["query"] == "python agent framework"


def test_tavily_tool_exports_openai_definition() -> None:
    """Verify Tavily search tool exports an OpenAI-compatible schema."""

    tool = TavilySearchTool()

    assert tool.to_openai_tool() == {
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": "Search the web with Tavily and return summarized results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of search results to return.",
                    },
                },
                "additionalProperties": False,
                "required": ["query"],
            },
        },
    }
