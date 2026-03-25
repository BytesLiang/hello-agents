"""Run a chat prompt through the unified LLM client."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from hello_agents.llm import LLMClient, LLMConfig, LLMMessage


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run a prompt with the LLM client.")
    parser.add_argument(
        "--prompt",
        default="Say hello in Chinese.",
        help="Prompt text sent as a user message.",
    )
    parser.add_argument(
        "--system",
        default="You are a concise assistant.",
        help="Optional system prompt.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response instead of waiting for a full completion.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the example client interaction."""

    load_dotenv()
    args = parse_args()
    client = LLMClient(LLMConfig.from_env())
    messages = [
        LLMMessage(role="system", content=args.system),
        LLMMessage(role="user", content=args.prompt),
    ]

    if args.stream:
        for chunk in client.stream(messages):
            print(chunk, end="", flush=True)
        print()
        return

    response = client.chat(messages)
    print(response.content)


if __name__ == "__main__":
    main()
