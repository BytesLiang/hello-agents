"""Run a ReActAgent with Tavily search enabled."""

from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv  # type: ignore[import-not-found]

from hello_agents import ReActAgent, TavilySearchTool, ToolRegistry
from hello_agents.llm import LLMClient, LLMConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ReActAgent demo."""

    parser = argparse.ArgumentParser(
        description="Run ReActAgent with Tavily search enabled."
    )
    parser.add_argument(
        "--prompt",
        default="Search the latest Python agent frameworks and summarize them.",
        help="User message passed to the agent.",
    )
    parser.add_argument(
        "--system",
        default=(
            "You are a ReAct agent. Think step by step. "
            "Use the tavily_search tool when web information is needed. "
            "Return valid JSON only."
        ),
        help="System prompt used by the agent.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum number of ReAct reasoning steps.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level for agent runtime events.",
    )
    return parser.parse_args()


def build_agent(system_prompt: str, max_steps: int) -> ReActAgent:
    """Build a ReActAgent with Tavily search registered."""

    llm = LLMClient(LLMConfig.from_env())
    tools = ToolRegistry()
    tools.register(TavilySearchTool())
    return ReActAgent(
        name="react-search-assistant",
        llm=llm,
        tools=tools,
        use_tools=True,
        system_prompt=system_prompt,
        max_steps=max_steps,
    )


def main() -> None:
    """Load environment variables, run the agent, and print the result."""

    load_dotenv()
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    agent = build_agent(system_prompt=args.system, max_steps=args.max_steps)
    print(agent.run(args.prompt))


if __name__ == "__main__":
    main()
