"""Tests for the command-style layered memory subsystem."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import cast

from hello_agents.chat_agent import ChatAgent
from hello_agents.llm.client import LLMClient
from hello_agents.llm.types import LLMMessage, LLMResponse
from hello_agents.memory import (
    LayeredMemory,
    MemoryConfig,
    MemoryKind,
    MemoryPatch,
    MemoryRecord,
    MemoryScope,
)
from hello_agents.memory.base import Embedder, VectorStore
from hello_agents.memory.config import (
    QdrantStoreConfig,
    SQLiteStoreConfig,
    WorkingMemoryConfig,
)
from hello_agents.memory.extractors import (
    LLMMemoryAnalyzer,
    RuleBasedMemoryAnalyzer,
)
from hello_agents.memory.models import (
    MemoryCandidate,
    MemoryProposal,
    VectorDocument,
    VectorSearchHit,
    WorkingMemoryKind,
    WorkingMemoryRecord,
)
from hello_agents.memory.stores.in_memory import InMemoryWorkingMemoryStore
from hello_agents.memory.stores.sqlite_store import SQLiteMemoryStore
from hello_agents.tools import ToolResult


class FakeLLMClient:
    """Provide deterministic chat responses for memory tests."""

    def __init__(self, responses: list[str]) -> None:
        """Store queued responses and captured requests."""

        self.responses = list(responses)
        self.calls: list[list[LLMMessage]] = []

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[dict[str, object]] | None = None,
    ) -> LLMResponse:
        """Capture the request and return the next queued response."""

        del tools
        self.calls.append(list(messages))
        return LLMResponse(model="fake-model", content=self.responses.pop(0))


class StubEmbedder(Embedder):
    """Return deterministic embeddings for vector-store tests."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return simple fixed-size vectors based on text length."""

        return [[float(len(text)), 1.0, 0.5] for text in texts]


class FailingEmbedder(Embedder):
    """Raise an exception for every embedding request."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Always fail to verify graceful degradation."""

        del texts
        raise RuntimeError("embedding unavailable")


class StubVectorStore(VectorStore):
    """Provide a deterministic in-memory vector store."""

    def __init__(self) -> None:
        """Initialize recorded upserts and stubbed search hits."""

        self.documents: dict[str, VectorDocument] = {}

    def upsert(self, document: VectorDocument, embedding: Sequence[float]) -> None:
        """Record the indexed document for later assertions."""

        del embedding
        self.documents[document.memory_id] = document

    def search(
        self,
        scope: MemoryScope,
        *,
        embedding: Sequence[float],
        memory_kinds: Sequence[MemoryKind],
        limit: int,
    ) -> list[VectorSearchHit]:
        """Return all matching hits inside the requested namespace."""

        del scope, embedding
        hits: list[VectorSearchHit] = []
        for document in self.documents.values():
            if document.memory_kind not in memory_kinds:
                continue
            hits.append(
                VectorSearchHit(
                    memory_id=document.memory_id,
                    memory_kind=document.memory_kind,
                    score=1.0,
                )
            )
        return hits[:limit]


def test_memory_query_augments_prompts_across_turns(tmp_path: Path) -> None:
    """Verify agent prompt construction uses the query command."""

    sqlite_path = tmp_path / "memory.sqlite3"
    llm = FakeLLMClient(
        responses=[
            "Plan: summarize atlas status in three bullets.",
            (
                '{"working_records":[{"kind":"working_plan","content":"summarize '
                'atlas status in three bullets.","summary":"summarize atlas status '
                'in three bullets.","pinned":true,"metadata":{}}],'
                '"candidates":[{"kind":"semantic_preference","key":"response_style",'
                '"value":"concise answers","content":"concise answers",'
                '"summary":"User preference: concise answers","confidence":0.9,'
                '"confirmed":true,"requires_confirmation":false,"metadata":{}},'
                '{"kind":"semantic_fact","key":"project_name","value":"my project is '
                'atlas","content":"my project is atlas","summary":"Confirmed fact: '
                'my project is atlas","confidence":0.9,"confirmed":true,'
                '"requires_confirmation":false,"metadata":{}},'
                '{"kind":"episodic","content":"Task: I prefer concise answers. '
                "remember that my project is atlas.\\nAssistant: Plan: summarize "
                'atlas status in three bullets.","summary":"Stored concise response '
                'preference and project fact.","confidence":0.8,"confirmed":false,'
                '"requires_confirmation":false,"metadata":{"task":"I prefer concise '
                'answers. remember that my project is atlas.","success":true,'
                '"tool_names":[]}}]}'
            ),
            "done",
            '{"working_records":[],"candidates":[]}',
        ]
    )
    memory = LayeredMemory(
        config=MemoryConfig(
            working=WorkingMemoryConfig(),
            sqlite=SQLiteStoreConfig(path=sqlite_path),
        ),
        analyzer=LLMMemoryAnalyzer(
            cast(LLMClient, llm),
            fallback=RuleBasedMemoryAnalyzer(),
        ),
    )
    agent = ChatAgent(
        name="memory-chat",
        llm=cast(LLMClient, llm),
        memory=memory,
    )
    memory_scope = MemoryScope(
        user_id="user-1",
        session_id="session-1",
        agent_id="agent-1",
    )

    agent.run(
        "I prefer concise answers. remember that my project is atlas.",
        memory_scope=memory_scope,
    )
    agent.run("Summarize atlas status.", memory_scope=memory_scope)

    second_user_message = llm.calls[2][1].content
    assert "[MEMORY]" in second_user_message
    assert "Current plan:" in second_user_message
    assert "User preferences:" in second_user_message
    assert "Confirmed facts:" in second_user_message
    assert "Relevant task history:" in second_user_message
    assert "concise answers" in second_user_message
    assert "my project is atlas" in second_user_message


def test_llm_analyzer_can_store_natural_language_semantic_memory(
    tmp_path: Path,
) -> None:
    """Verify non-regex semantic signals can flow through propose and commit."""

    sqlite_path = tmp_path / "memory.sqlite3"
    llm = FakeLLMClient(
        responses=[
            "Understood.",
            (
                "{"
                '"working_records": [],'
                '"candidates": ['
                '{"kind":"semantic_preference","key":"response_style",'
                '"value":"terse responses","content":"terse responses",'
                '"summary":"User prefers terse responses.",'
                '"confidence":0.92,"confirmed":true,'
                '"requires_confirmation":false,"metadata":{}},'
                '{"kind":"semantic_fact","key":"project_codename",'
                '"value":"Atlas is the project codename",'
                '"content":"Atlas is the project codename",'
                '"summary":"Atlas is the current project codename.",'
                '"confidence":0.9,"confirmed":true,'
                '"requires_confirmation":false,"metadata":{}}'
                "]"
                "}"
            ),
            "done",
            '{"working_records":[],"candidates":[]}',
        ]
    )
    memory = LayeredMemory(
        config=MemoryConfig(
            working=WorkingMemoryConfig(),
            sqlite=SQLiteStoreConfig(path=sqlite_path),
        ),
        analyzer=LLMMemoryAnalyzer(
            cast(LLMClient, llm),
            fallback=RuleBasedMemoryAnalyzer(),
        ),
    )
    agent = ChatAgent(
        name="memory-chat",
        llm=cast(LLMClient, llm),
        memory=memory,
    )
    memory_scope = MemoryScope(
        user_id="user-1",
        session_id="session-1",
        agent_id="agent-1",
    )

    agent.run(
        "Please keep responses short. Atlas is the codename for this project.",
        memory_scope=memory_scope,
    )
    query_result = memory.query(
        "What do you know about my style and project?",
        scope=memory_scope,
    )

    assert len(query_result.preferences) == 1
    assert len(query_result.facts) == 1
    assert query_result.preferences[0].value == "terse responses"
    assert query_result.facts[0].summary == "Atlas is the current project codename."


def test_propose_has_no_persistence_side_effect(tmp_path: Path) -> None:
    """Verify propose alone does not write long-term memory."""

    sqlite_path = tmp_path / "memory.sqlite3"
    memory = LayeredMemory(
        config=MemoryConfig(sqlite=SQLiteStoreConfig(path=sqlite_path)),
    )
    memory_scope = MemoryScope(
        user_id="user-1",
        session_id="session-1",
        agent_id="agent-1",
    )

    proposal = memory.propose(
        "remember that the repo name is hello-agents",
        "Stored.",
        scope=memory_scope,
    )
    assert proposal.candidates

    query_result = memory.query("hello-agents", scope=memory_scope)
    assert query_result.facts == ()


def test_commit_filters_and_persists_candidates(tmp_path: Path) -> None:
    """Verify commit persists accepted candidates and rejects generic procedures."""

    sqlite_path = tmp_path / "memory.sqlite3"
    memory = LayeredMemory(
        config=MemoryConfig(sqlite=SQLiteStoreConfig(path=sqlite_path)),
    )
    memory_scope = MemoryScope(
        user_id="user-1",
        session_id="session-1",
        agent_id="agent-1",
    )

    proposal = memory.propose(
        "Summarize this document.",
        "Here is a summary.",
        scope=memory_scope,
        tool_results=(ToolResult(tool_name="echo", content="tool output"),),
        success=True,
    )
    result = memory.commit(proposal, scope=memory_scope)
    query_result = memory.query("summarize", scope=memory_scope)

    assert result.accepted_records
    assert len(query_result.procedures) == 1
    assert query_result.procedures[0].metadata["task_type"] == "summarize"


def test_generic_success_is_rejected_as_procedural_memory(tmp_path: Path) -> None:
    """Verify policy blocks generic successful answers from procedural memory."""

    sqlite_path = tmp_path / "memory.sqlite3"
    memory = LayeredMemory(
        config=MemoryConfig(sqlite=SQLiteStoreConfig(path=sqlite_path)),
    )
    memory_scope = MemoryScope(
        user_id="user-1",
        session_id="session-1",
        agent_id="agent-1",
    )

    proposal = memory.propose(
        "agent分几类",
        "这里是一个通用回答。",
        scope=memory_scope,
        success=True,
    )
    proposal = MemoryProposal(
        working_records=proposal.working_records,
        candidates=proposal.candidates
        + (
            MemoryCandidate(
                kind=MemoryKind.PROCEDURAL,
                content="Generic successful answer.",
                summary="Successful approach for general_task",
                confidence=0.9,
                metadata={"task_type": "general_task", "tool_count": 0},
            ),
        ),
    )
    result = memory.commit(proposal, scope=memory_scope)
    query_result = memory.query("agent", scope=memory_scope)

    assert any(item.reason == "generic_procedure" for item in result.rejected)
    assert query_result.procedures == ()


def test_add_and_update_semantic_record(tmp_path: Path) -> None:
    """Verify direct add and update commands operate on semantic memory."""

    sqlite_path = tmp_path / "memory.sqlite3"
    memory = LayeredMemory(
        config=MemoryConfig(sqlite=SQLiteStoreConfig(path=sqlite_path)),
    )
    memory_scope = MemoryScope(
        user_id="user-1",
        session_id="session-1",
        agent_id="agent-1",
    )
    record = MemoryRecord(
        kind=MemoryKind.SEMANTIC_FACT,
        user_id=memory_scope.user_id,
        session_id=memory_scope.session_id,
        agent_id=memory_scope.agent_id,
        key="repo_name",
        value="hello-agents",
        content="hello-agents",
        summary="Confirmed fact: repo name is hello-agents",
        confidence=0.95,
        confirmed=True,
    )

    written = memory.add(record, scope=memory_scope)
    updated = memory.update(
        written.id,
        MemoryPatch(summary="Confirmed fact: repository name is hello-agents"),
        scope=memory_scope,
    )
    query_result = memory.query("repository name", scope=memory_scope)

    assert updated.summary == "Confirmed fact: repository name is hello-agents"
    assert query_result.facts[0].summary == updated.summary


def test_working_memory_ttl_expires_entries() -> None:
    """Verify expired working-memory entries are pruned."""

    store = InMemoryWorkingMemoryStore(
        WorkingMemoryConfig(ttl_seconds=1, max_entries=4)
    )
    scope = MemoryScope(
        user_id="user-1",
        session_id="session-1",
        agent_id="agent-1",
    )
    baseline = MemoryRecord(
        kind=MemoryKind.WORKING_CONTEXT,
        user_id=scope.user_id,
        session_id=scope.session_id,
        agent_id=scope.agent_id,
        content="short lived",
        summary="short lived",
    )
    internal_record = store.list_entries(scope)
    assert internal_record == []
    store.append_entries(
        scope,
        [
            WorkingMemoryRecord(
                user_id=scope.user_id,
                session_id=scope.session_id,
                agent_id=scope.agent_id,
                kind=WorkingMemoryKind.CONTEXT,
                content=baseline.content,
            )
        ],
        now=baseline.created_at,
    )

    later = baseline.created_at + timedelta(seconds=2)
    assert store.list_entries(scope, now=later) == []


def test_vector_indexing_gracefully_degrades_when_embedding_fails(
    tmp_path: Path,
) -> None:
    """Verify long-term persistence survives embedding failure."""

    sqlite_path = tmp_path / "memory.sqlite3"
    store = SQLiteMemoryStore(SQLiteStoreConfig(path=sqlite_path))
    memory = LayeredMemory(
        config=MemoryConfig(
            sqlite=SQLiteStoreConfig(path=sqlite_path),
            qdrant=QdrantStoreConfig(enabled=True, url="http://example.com"),
        ),
        long_term_store=store,
        vector_store=StubVectorStore(),
        embedder=FailingEmbedder(),
    )
    scope = MemoryScope(
        user_id="user-1",
        session_id="session-1",
        agent_id="agent-1",
    )

    proposal = memory.propose(
        "remember that the repo name is hello-agents",
        "Stored.",
        scope=scope,
    )
    result = memory.commit(proposal, scope=scope)
    query_result = memory.query("hello-agents", scope=scope)

    assert result.accepted_records
    assert len(query_result.facts) == 1


def test_query_uses_vector_scores_when_available(tmp_path: Path) -> None:
    """Verify vector-backed retrieval contributes records to query results."""

    sqlite_path = tmp_path / "memory.sqlite3"
    store = SQLiteMemoryStore(SQLiteStoreConfig(path=sqlite_path))
    vector_store = StubVectorStore()
    memory = LayeredMemory(
        config=MemoryConfig(
            sqlite=SQLiteStoreConfig(path=sqlite_path),
            qdrant=QdrantStoreConfig(enabled=True, url="http://example.com"),
        ),
        long_term_store=store,
        vector_store=vector_store,
        embedder=StubEmbedder(),
    )
    scope = MemoryScope(
        user_id="user-1",
        session_id="session-1",
        agent_id="agent-1",
    )

    record = MemoryRecord(
        kind=MemoryKind.SEMANTIC_FACT,
        user_id=scope.user_id,
        session_id=scope.session_id,
        agent_id=scope.agent_id,
        key="repo_name",
        value="hello-agents",
        content="hello-agents",
        summary="Confirmed fact: repo name is hello-agents",
        confidence=0.95,
        confirmed=True,
    )
    memory.add(record, scope=scope)

    query_result = memory.query("What is the repo name?", scope=scope)

    assert vector_store.documents
    assert len(query_result.facts) == 1
