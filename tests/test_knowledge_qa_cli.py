"""Tests for the formal knowledge QA CLI."""

from __future__ import annotations

import tomllib
from pathlib import Path

from hello_agents.apps.knowledge_qa import (
    JsonKnowledgeBaseStore,
    JsonlRunTraceStore,
    KnowledgeQAConfig,
)
from hello_agents.apps.knowledge_qa.cli import main, parse_args
from hello_agents.apps.knowledge_qa.runtime import KnowledgeQARuntime
from hello_agents.llm.types import LLMMessage, LLMResponse
from hello_agents.rag.models import RagChunk


class FakeLLM:
    """Return deterministic completions for CLI tests."""

    def __init__(self, response: str) -> None:
        """Store the response payload."""

        self._response = response
        self.calls: list[list[LLMMessage]] = []

    def chat(
        self,
        messages,
        *,
        temperature=None,
        max_tokens=None,
        tools=None,
    ) -> LLMResponse:
        """Capture the request and return one fixed response."""

        del temperature, max_tokens, tools
        self.calls.append(list(messages))
        return LLMResponse(
            model="fake-model",
            content=self._response,
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
        )


class StubRagRetriever:
    """Return deterministic retrieval hits for CLI tests."""

    def __init__(self, chunks: list[RagChunk]) -> None:
        """Store the chunk list returned by query()."""

        self._chunks = chunks

    def query(
        self,
        text: str,
        *,
        top_k: int | None = None,
        kb_id: str | None = None,
    ) -> list[RagChunk]:
        """Return the configured retrieval hits."""

        del text, kb_id
        if top_k is None:
            return list(self._chunks)
        return list(self._chunks[:top_k])


class StubRagIndexer:
    """Return deterministic indexing counts for CLI tests."""

    def __init__(self) -> None:
        """Track deletion requests for assertions."""

        self.deleted_kb_ids: list[str] = []

    def index_file(self, path: Path, *, kb_id: str, document_id: str) -> int:
        """Return a fixed chunk count for indexed inputs."""

        del kb_id, document_id
        return 3 if path.is_file() else 5

    def delete_document(self, *, kb_id: str, document_id: str) -> None:
        """Accept document deletion requests during tests."""

        del kb_id, document_id

    def delete_knowledge_base(self, *, kb_id: str) -> None:
        """Accept knowledge-base deletion requests during tests."""

        self.deleted_kb_ids.append(kb_id)


def build_runtime(
    tmp_path: Path,
    *,
    llm: FakeLLM | None = None,
    retriever: StubRagRetriever | None = None,
    indexer: StubRagIndexer | None = None,
) -> KnowledgeQARuntime:
    """Build a temporary runtime for CLI tests."""

    config = KnowledgeQAConfig(
        knowledge_base_store_path=tmp_path / "knowledge_bases.json",
        trace_store_path=tmp_path / "traces.jsonl",
    )
    return KnowledgeQARuntime(
        config=config,
        knowledge_base_store=JsonKnowledgeBaseStore(config.knowledge_base_store_path),
        trace_store=JsonlRunTraceStore(config.trace_store_path),
        llm=llm,
        rag_retriever=retriever,
        rag_indexer=indexer,
    )


def test_parse_args_reads_inspect_options() -> None:
    """Verify inspect options are parsed for the formal CLI."""

    args = parse_args(["inspect", "--traces", "--limit", "3"])

    assert args is not None
    assert args.command == "inspect"
    assert args.traces is True
    assert args.limit == 3


def test_cli_ingest_and_inspect_commands(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Verify ingest and inspect both use the shared runtime."""

    document = tmp_path / "guide.md"
    document.write_text("# Guide\n\nAlpha", encoding="utf-8")
    runtime = build_runtime(tmp_path, indexer=StubRagIndexer())
    monkeypatch.setattr(
        "hello_agents.apps.knowledge_qa.cli.build_runtime",
        lambda: runtime,
    )

    ingest_exit_code = main(
        [
            "ingest",
            "--name",
            "Guide KB",
            "--paths",
            str(document),
            "--description",
            "Docs",
        ]
    )
    inspect_exit_code = main(["inspect"])
    output = capsys.readouterr().out

    assert ingest_exit_code == 0
    assert inspect_exit_code == 0
    assert "Indexed knowledge base Guide KB" in output
    assert "Guide KB status=ready docs=1 chunks=3" in output


def test_cli_ask_command_prints_answer_and_trace(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Verify ask prints the answer, citations, and trace id."""

    document = tmp_path / "atlas.md"
    document.write_text("# Atlas\n\nUses Qdrant.", encoding="utf-8")
    runtime = build_runtime(
        tmp_path,
        llm=FakeLLM(
            '{"answer":"Atlas uses Qdrant for retrieval.",'
            '"answered":true,'
            '"reason":null,'
            '"citation_indices":[1]}'
        ),
        retriever=StubRagRetriever(
            [
                RagChunk(
                    id="chunk-1",
                    source=str(document),
                    content="Atlas uses Qdrant for retrieval.",
                    score=0.9,
                )
            ]
        ),
        indexer=StubRagIndexer(),
    )
    knowledge_base = runtime.build_ingest_service().ingest("Atlas Docs", [document])
    monkeypatch.setattr(
        "hello_agents.apps.knowledge_qa.cli.build_runtime",
        lambda: runtime,
    )

    exit_code = main(
        [
            "ask",
            "--question",
            "What does Atlas use for retrieval?",
            "--kb-id",
            knowledge_base.kb_id,
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Atlas uses Qdrant for retrieval." in output
    assert "Citations:" in output
    assert "Trace:" in output


def test_console_script_points_to_formal_cli() -> None:
    """Verify the packaging metadata exposes the formal CLI entry point."""

    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert (
        payload["project"]["scripts"]["hello-agents-knowledge-qa"]
        == "hello_agents.apps.knowledge_qa.cli:main"
    )
