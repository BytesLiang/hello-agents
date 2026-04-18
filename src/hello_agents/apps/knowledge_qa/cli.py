"""Command-line interface for the knowledge QA application."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv  # type: ignore[import-not-found]

from hello_agents.apps.knowledge_qa.runtime import KnowledgeQARuntime


def create_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the knowledge QA application."""

    parser = argparse.ArgumentParser(
        description="Run the hello-agents knowledge QA application."
    )
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
        help="Knowledge base identifier for source filtering.",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="List persisted knowledge bases or recent traces.",
    )
    inspect_parser.add_argument(
        "--traces",
        action="store_true",
        help="Show recent traces instead of knowledge bases.",
    )
    inspect_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of trace rows to show with --traces.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace | None:
    """Parse command-line arguments for the knowledge QA application."""

    parser = create_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv:
        parser.print_help()
        return None

    args = parser.parse_args(raw_argv)
    if args.command is None:
        parser.print_help()
        return None
    return args


def build_runtime() -> KnowledgeQARuntime:
    """Build the default CLI runtime from environment variables."""

    return KnowledgeQARuntime()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected knowledge QA command."""

    load_dotenv()
    args = parse_args(argv)
    if args is None:
        return 0

    runtime = build_runtime()

    if args.command == "inspect":
        service = runtime.build_read_service()
        if args.traces:
            for trace in service.list_recent_traces(limit=args.limit):
                print(f"{trace.trace_id} {trace.created_at} answered={trace.answered}")
                print(f"Q: {trace.question}")
                print(f"A: {trace.answer}\n")
            return 0

        for knowledge_base in service.list_knowledge_bases():
            print(
                f"{knowledge_base.kb_id} {knowledge_base.name} "
                f"status={knowledge_base.status.value} "
                f"docs={knowledge_base.document_count} "
                f"chunks={knowledge_base.chunk_count}"
            )
        return 0

    if args.command == "ingest":
        service = runtime.build_ingest_service()
        knowledge_base = service.ingest(
            args.name,
            paths=_parse_paths(args.paths),
            description=args.description,
        )
        print(
            f"Indexed knowledge base {knowledge_base.name} ({knowledge_base.kb_id}) "
            f"docs={knowledge_base.document_count} chunks={knowledge_base.chunk_count}"
        )
        return 0

    if args.command == "ask":
        service = runtime.build_answer_service()
        result = service.ask(args.question, kb_id=args.kb_id)
        print(result.answer)
        if result.citations:
            print("\nCitations:")
            for citation in result.citations:
                print(f"[{citation.index}] {citation.source} - {citation.snippet}")
        if result.trace_id:
            print(f"\nTrace: {result.trace_id}")
        return 0

    raise RuntimeError(f"Unsupported command: {args.command}")


def _parse_paths(raw: str) -> tuple[Path, ...]:
    """Parse a comma-separated path list."""

    return tuple(Path(part.strip()) for part in raw.split(",") if part.strip())


if __name__ == "__main__":
    raise SystemExit(main())
