"""Run a ChatAgent with lightweight RAG enabled."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv  # type: ignore[import-not-found]

from hello_agents import ChatAgent
from hello_agents.llm import LLMClient, LLMConfig
from hello_agents.rag import RagConfig, RagIndexer, RagRetriever
from hello_agents.rag.qdrant_store import RagQdrantStore


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the RAG demo."""

    parser = argparse.ArgumentParser(
        description="Run ChatAgent with RAG-enabled retrieval."
    )
    parser.add_argument(
        "--paths",
        default="",
        help="Comma-separated list of folders/files to index.",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant. Use retrieved context when relevant.",
        help="System prompt used by the agent.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level for agent runtime events.",
    )
    return parser.parse_args()


def _build_rag(paths: tuple[Path, ...]) -> RagRetriever:
    """Build the RAG retriever and index the supplied paths."""

    config = RagConfig.from_env()
    if not config.qdrant_url:
        raise RuntimeError("RAG requires QDRANT_URL to be configured.")
    if config.embed is None:
        raise RuntimeError("RAG requires embedding configuration (EMBED_*).")

    store = RagQdrantStore(config)
    indexer = RagIndexer(config=config, store=store)
    for path in paths:
        indexed = indexer.index_folder(path)
        logging.info("Indexed %s chunks from %s", indexed, path)
    return RagRetriever(config=config, store=store)


def main() -> None:
    """Run a simple REPL that demonstrates RAG retrieval."""

    load_dotenv()
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    rag_paths = tuple(
        Path(item.strip()) for item in args.paths.split(",") if item.strip()
    )
    if not rag_paths:
        raise RuntimeError("Please pass --paths to index (folder or file).")

    rag = _build_rag(rag_paths)
    llm = LLMClient(LLMConfig.from_env())
    agent = ChatAgent(
        name="rag-chat-agent",
        llm=llm,
        rag=rag,
        system_prompt=args.system,
    )

    print("Type 'exit' or 'quit' to stop.")
    print("Ask a question about the indexed content.")

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

        response = agent.run(prompt)
        print(f"Agent> {response}")


if __name__ == "__main__":
    main()
