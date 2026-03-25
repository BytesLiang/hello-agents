"""Define the standard base abstraction for framework tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

ToolValueType = Literal["string", "integer", "number", "boolean"]


@dataclass(slots=True, frozen=True)
class ToolParameter:
    """Describe a single structured tool input parameter."""

    name: str
    description: str
    value_type: ToolValueType = "string"
    required: bool = True
    enum: tuple[str, ...] | None = None

    def to_json_schema(self) -> dict[str, object]:
        """Convert the parameter into a JSON Schema property definition."""

        schema: dict[str, object] = {
            "type": self.value_type,
            "description": self.description,
        }
        if self.enum is not None:
            schema["enum"] = list(self.enum)
        return schema


@dataclass(slots=True, frozen=True)
class ToolSchema:
    """Describe tool input as a JSON Schema object."""

    parameters: tuple[ToolParameter, ...] = ()
    additional_properties: bool = False

    def validate(self, payload: dict[str, object]) -> None:
        """Validate a payload against the schema definition."""

        allowed_names = {parameter.name for parameter in self.parameters}

        for parameter in self.parameters:
            if parameter.required and parameter.name not in payload:
                raise ValueError(
                    f"Missing required parameter '{parameter.name}'."
                )

            if parameter.name in payload and not _matches_type(
                payload[parameter.name],
                parameter.value_type,
            ):
                raise ValueError(
                    f"Parameter '{parameter.name}' must be of type "
                    f"'{parameter.value_type}'."
                )

        unexpected_names = set(payload) - allowed_names
        if unexpected_names and not self.additional_properties:
            unexpected_list = ", ".join(sorted(unexpected_names))
            raise ValueError(f"Unexpected parameter(s): {unexpected_list}.")

    def to_json_schema(self) -> dict[str, object]:
        """Convert the tool schema into an OpenAI-compatible JSON Schema."""

        properties = {
            parameter.name: parameter.to_json_schema()
            for parameter in self.parameters
        }
        required = [
            parameter.name for parameter in self.parameters if parameter.required
        ]

        schema: dict[str, object] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": self.additional_properties,
        }
        if required:
            schema["required"] = required
        return schema


@dataclass(slots=True, frozen=True)
class ToolResult:
    """Represent a normalized tool execution result."""

    tool_name: str
    content: str
    success: bool = True
    metadata: dict[str, object] = field(default_factory=dict)


class Tool(ABC):
    """Define the standard interface every concrete tool must implement."""

    name: str
    description: str

    def __init__(
        self,
        name: str,
        description: str,
        schema: ToolSchema | None = None,
    ) -> None:
        """Store the common metadata required for every tool."""

        self.name = name
        self.description = description
        self.schema = schema or ToolSchema()

    def to_openai_tool(self) -> dict[str, object]:
        """Convert the tool into an OpenAI function-calling definition."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema.to_json_schema(),
            },
        }

    @abstractmethod
    def execute(self, payload: dict[str, object]) -> ToolResult:
        """Execute the tool against a structured payload."""


def _matches_type(value: object, value_type: ToolValueType) -> bool:
    """Return whether a value matches the declared schema type."""

    if value_type == "string":
        return isinstance(value, str)
    if value_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if value_type == "number":
        return (
            isinstance(value, int) or isinstance(value, float)
        ) and not isinstance(value, bool)
    if value_type == "boolean":
        return isinstance(value, bool)
    return False
