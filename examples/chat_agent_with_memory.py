"""Run a ChatAgent with layered memory enabled."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv  # type: ignore[import-not-found]

from hello_agents import (
    ChatAgent,
    LayeredMemory,
    MemoryConfig,
    MemoryScope,
    RagConfig,
    RagIndexer,
    RagRetriever,
)
from hello_agents.llm import LLMClient, LLMConfig
from hello_agents.memory import SQLiteStoreConfig
from hello_agents.memory.extractors import (
    LLMMemoryAnalyzer,
    RuleBasedMemoryAnalyzer,
)
from hello_agents.rag.qdrant_store import RagQdrantStore


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the memory-enabled ChatAgent demo."""

    parser = argparse.ArgumentParser(
        description="Run ChatAgent with layered memory enabled."
    )
    parser.add_argument(
        "--user-id",
        default="demo-user",
        help="Stable user identifier used for long-term memory.",
    )
    parser.add_argument(
        "--session-id",
        default="demo-session",
        help="Session identifier used for working memory.",
    )
    parser.add_argument(
        "--agent-id",
        default="memory-chat-agent",
        help="Agent identifier used for namespacing memory.",
    )
    parser.add_argument(
        "--memory-db",
        default=".hello_agents/example-memory.sqlite3",
        help="SQLite path used for long-term memory storage.",
    )
    parser.add_argument(
        "--system",
        default=(
            "You are a concise assistant. Use the provided memory block to "
            "preserve user preferences, confirmed facts, recent plans, and "
            "successful prior context across turns."
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


def build_agent(*, system_prompt: str, memory_db: Path, agent_id: str) -> ChatAgent:
    """Build a ChatAgent backed by the command-style memory component."""

    llm = LLMClient(LLMConfig.from_env())
    base_config = MemoryConfig.from_env()
    memory = LayeredMemory(
        config=MemoryConfig(
            working=base_config.working,
            sqlite=SQLiteStoreConfig(path=memory_db),
            redis=base_config.redis,
            qdrant=base_config.qdrant,
            neo4j=base_config.neo4j,
            embed=base_config.embed,
        ),
        analyzer=LLMMemoryAnalyzer(
            llm,
            fallback=RuleBasedMemoryAnalyzer(),
        ),
    )
    rag = _build_rag()
    return ChatAgent(
        name=agent_id,
        llm=llm,
        memory=memory,
        rag=rag,
        system_prompt=system_prompt,
    )


def _build_rag() -> RagRetriever | None:
    """Build and optionally index RAG sources when enabled."""

    rag_config = RagConfig.from_env()
    if not rag_config.enabled or not rag_config.paths:
        return None
    if rag_config.embed is None or rag_config.qdrant_url is None:
        logging.warning("RAG enabled but missing embeddings or Qdrant URL.")
        return None

    store = RagQdrantStore(rag_config)
    indexer = RagIndexer(config=rag_config, store=store)
    for path in rag_config.paths:
        indexer.index_folder(path)
    return RagRetriever(config=rag_config, store=store)


def main() -> None:
    """Run a simple REPL that demonstrates cross-turn memory."""

    load_dotenv()
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    memory_db = Path(args.memory_db)
    agent = build_agent(
        system_prompt=args.system,
        memory_db=memory_db,
        agent_id=args.agent_id,
    )
    memory_scope = MemoryScope(
        user_id=args.user_id,
        session_id=args.session_id,
        agent_id=args.agent_id,
    )

    print(f"Memory DB: {memory_db}")
    print("Type 'exit' or 'quit' to stop.")
    print("Try these two turns:")
    print("1. I prefer concise answers. remember that my project is atlas.")
    print("2. Summarize my project status in one sentence.")

    while True:
        try:
            prompt = input("\nYou> ").strip()
        except EOFError:
            print()
            break

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break

        response = agent.run(prompt, memory_scope=memory_scope)
        print(f"Agent> {response}")


if __name__ == "__main__":
    main()
