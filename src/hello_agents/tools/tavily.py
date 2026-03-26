"""Provide a Tavily-backed web search tool."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from hello_agents.tools.base import Tool, ToolParameter, ToolResult, ToolSchema

if TYPE_CHECKING:
    from tavily import TavilyClient  # type: ignore[import-untyped]


class TavilySearchTool(Tool):
    """Search the web through the Tavily search API."""

    def __init__(
        self,
        *,
        api_key_env: str = "TAVILY_API_KEY",
        client: Any | None = None,
    ) -> None:
        """Store Tavily settings and declare the tool schema."""

        super().__init__(
            name="tavily_search",
            description="Search the web with Tavily and return summarized results.",
            schema=ToolSchema(
                parameters=(
                    ToolParameter(
                        name="query",
                        description="Search query text.",
                        value_type="string",
                    ),
                    ToolParameter(
                        name="max_results",
                        description="Maximum number of search results to return.",
                        value_type="integer",
                        required=False,
                    ),
                )
            ),
        )
        self.api_key_env = api_key_env
        self._client = client

    def execute(self, payload: dict[str, object]) -> ToolResult:
        """Execute a Tavily search and normalize the result."""

        query = str(payload["query"])
        max_results = payload.get("max_results", 5)
        response = self._client_or_default().search(
            query=query,
            max_results=max_results,
        )
        if not isinstance(response, dict):
            raise ValueError("Tavily client returned a non-object response.")
        return _normalize_tavily_response(self.name, query, response)

    def _client_or_default(self) -> TavilyClient | Any:
        """Return an injected client or create the default Tavily client."""

        if self._client is not None:
            return self._client

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(
                f"Environment variable '{self.api_key_env}' is required."
            )

        from tavily import TavilyClient  # type: ignore[import-not-found]

        self._client = TavilyClient(api_key=api_key)
        return self._client

def _normalize_tavily_response(
    tool_name: str,
    query: str,
    response: dict[str, object],
) -> ToolResult:
    """Normalize Tavily API output into a tool result."""

    results = response.get("results", [])
    if not isinstance(results, list):
        raise ValueError("Tavily response field 'results' must be a list.")

    lines = [f"Search query: {query}"]
    normalized_results: list[dict[str, str]] = []

    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        url = item.get("url")
        content = item.get("content")

        if not isinstance(title, str) or not isinstance(url, str):
            continue

        snippet = content if isinstance(content, str) else ""
        lines.append(f"{index}. {title}")
        lines.append(f"   URL: {url}")
        if snippet:
            lines.append(f"   Snippet: {snippet}")

        normalized_results.append(
            {
                "title": title,
                "url": url,
                "content": snippet,
            }
        )

    answer = response.get("answer")
    if isinstance(answer, str) and answer:
        lines.insert(1, f"Answer: {answer}")

    return ToolResult(
        tool_name=tool_name,
        content="\n".join(lines),
        metadata={
            "query": query,
            "answer": answer if isinstance(answer, str) else "",
            "results": normalized_results,
        },
    )
