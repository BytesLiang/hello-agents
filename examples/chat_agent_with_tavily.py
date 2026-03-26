"""Run a ChatAgent with Tavily search enabled."""

from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv  # type: ignore[import-not-found]

from hello_agents import ChatAgent, TavilySearchTool, ToolRegistry
from hello_agents.llm import LLMClient, LLMConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ChatAgent demo."""

    parser = argparse.ArgumentParser(
        description="Run ChatAgent with Tavily tool calling enabled."
    )
    parser.add_argument(
        "--prompt",
        default="Search the latest information about OpenAI and summarize it briefly.",
        help="User message passed to the agent.",
    )
    parser.add_argument(
        "--system",
        default=(
            "You are a concise assistant. Use the tavily_search tool when "
            "fresh web information is needed."
        ),
        help="System prompt used by the agent.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level for agent runtime events.",
    )
    return parser.parse_args()


def build_agent(system_prompt: str) -> ChatAgent:
    """Build a ChatAgent with Tavily search registered."""

    llm = LLMClient(LLMConfig.from_env())
    tools = ToolRegistry()
    tools.register(TavilySearchTool())
    return ChatAgent(
        name="search-assistant",
        llm=llm,
        tools=tools,
        use_tools=True,
        system_prompt=system_prompt,
    )


def main() -> None:
    """Load environment variables, run the agent, and print the result."""

    load_dotenv()
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    agent = build_agent(system_prompt=args.system)
    print(agent.run(args.prompt))


if __name__ == "__main__":
    main()
