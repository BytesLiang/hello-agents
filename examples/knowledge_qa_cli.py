"""Run the knowledge QA application from the command line."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv  # type: ignore[import-not-found]

from hello_agents.apps.knowledge_qa import (
    JsonKnowledgeBaseStore,
    JsonlRunTraceStore,
    KnowledgeQAConfig,
    KnowledgeQAService,
)
from hello_agents.llm import LLMClient, LLMConfig
from hello_agents.rag import RagConfig, RagIndexer, RagRetriever
from hello_agents.rag.qdrant_store import RagQdrantStore


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace | None:
    """Parse command-line arguments for the knowledge QA application."""

    parser = argparse.ArgumentParser(description="Run the knowledge QA application.")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="Index one knowledge base.")
    ingest_parser.add_argument("--name", required=True, help="Knowledge base name.")
    ingest_parser.add_argument(
        "--paths",
        required=True,
        help="Comma-separated list of folders or files to index.",
    )
    ingest_parser.add_argument(
        "--description",
        default="",
        help="Optional knowledge base description.",
    )

    ask_parser = subparsers.add_parser("ask", help="Ask one question.")
    ask_parser.add_argument("--question", required=True, help="Question to answer.")
    ask_parser.add_argument(
        "--kb-id",
        default=None,
        help="Optional knowledge base identifier for source filtering.",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="List persisted knowledge bases.",
    )
    inspect_parser.add_argument(
        "--traces",
        action="store_true",
        help="Show recent traces instead of knowledge bases.",
    )

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv:
        parser.print_help()
        return None

    args = parser.parse_args(raw_argv)
    if args.command is None:
        parser.print_help()
        return None
    return args


def main() -> None:
    """Run the selected knowledge QA command."""

    load_dotenv()
    args = parse_args()
    if args is None:
        return
    config = KnowledgeQAConfig.from_env()
    kb_store = JsonKnowledgeBaseStore(config.knowledge_base_store_path)
    trace_store = JsonlRunTraceStore(config.trace_store_path)

    if args.command == "inspect":
        service = KnowledgeQAService(
            config=config,
            knowledge_base_store=kb_store,
            trace_store=trace_store,
        )
        if args.traces:
            for trace in service.list_recent_traces(limit=10):
                print(f"{trace.trace_id} {trace.created_at} answered={trace.answered}")
                print(f"Q: {trace.question}")
                print(f"A: {trace.answer}\n")
            return

        for knowledge_base in service.list_knowledge_bases():
            print(
                f"{knowledge_base.kb_id} {knowledge_base.name} "
                f"status={knowledge_base.status.value} "
                f"docs={knowledge_base.document_count} "
                f"chunks={knowledge_base.chunk_count}"
            )
        return

    rag_config = RagConfig.from_env()
    if not rag_config.qdrant_url:
        raise RuntimeError("Knowledge QA requires QDRANT_URL to be configured.")
    if rag_config.embed is None:
        raise RuntimeError("Knowledge QA requires embedding configuration (EMBED_*).")

    store = RagQdrantStore(rag_config)

    if args.command == "ingest":
        service = KnowledgeQAService(
            config=config,
            rag_indexer=RagIndexer(config=rag_config, store=store),
            knowledge_base_store=kb_store,
            trace_store=trace_store,
        )
        knowledge_base = service.ingest(
            args.name,
            paths=_parse_paths(args.paths),
            description=args.description,
        )
        print(
            f"Indexed knowledge base {knowledge_base.name} ({knowledge_base.kb_id}) "
            f"docs={knowledge_base.document_count} chunks={knowledge_base.chunk_count}"
        )
        return

    if args.command == "ask":
        service = KnowledgeQAService(
            config=config,
            llm=LLMClient(LLMConfig.from_env()),
            rag_retriever=RagRetriever(config=rag_config, store=store),
            knowledge_base_store=kb_store,
            trace_store=trace_store,
        )
        result = service.ask(args.question, kb_id=args.kb_id)
        print(result.answer)
        if result.citations:
            print("\nCitations:")
            for citation in result.citations:
                print(f"[{citation.index}] {citation.source} - {citation.snippet}")
        if result.trace_id:
            print(f"\nTrace: {result.trace_id}")
        return

    raise RuntimeError(f"Unsupported command: {args.command}")


def _parse_paths(raw: str) -> tuple[Path, ...]:
    """Parse a comma-separated path list."""

    return tuple(Path(part.strip()) for part in raw.split(",") if part.strip())


if __name__ == "__main__":
    main()
