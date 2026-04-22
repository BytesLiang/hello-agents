"""Runtime assembly helpers for the knowledge QA interfaces."""

from __future__ import annotations

from dataclasses import dataclass

from hello_agents.apps.knowledge_qa.config import KnowledgeQAConfig
from hello_agents.apps.knowledge_qa.ingest import SupportsDocumentIndex
from hello_agents.apps.knowledge_qa.retrieve import SupportsRagQuery
from hello_agents.apps.knowledge_qa.service import KnowledgeQAService, SupportsChat
from hello_agents.apps.knowledge_qa.store import JsonKnowledgeBaseStore
from hello_agents.apps.knowledge_qa.trace import JsonlRunTraceStore
from hello_agents.apps.knowledge_qa.uploads import UploadedDocumentStore
from hello_agents.llm import LLMClient, LLMConfig
from hello_agents.rag import RagConfig, RagIndexer, RagRetriever
from hello_agents.rag.qdrant_store import RagQdrantStore


@dataclass(slots=True, frozen=True)
class KnowledgeQAHealthStatus:
    """Describe the current runtime configuration health."""

    status: str
    llm_model: str
    llm_provider: str
    qdrant_configured: bool
    embedding_configured: bool
    knowledge_base_store_path: str
    trace_store_path: str
    upload_root_path: str


class KnowledgeQARuntime:
    """Assemble reusable dependencies for CLI and HTTP interfaces."""

    def __init__(
        self,
        *,
        config: KnowledgeQAConfig | None = None,
        rag_config: RagConfig | None = None,
        llm_config: LLMConfig | None = None,
        knowledge_base_store: JsonKnowledgeBaseStore | None = None,
        trace_store: JsonlRunTraceStore | None = None,
        llm: SupportsChat | None = None,
        rag_indexer: SupportsDocumentIndex | None = None,
        rag_retriever: SupportsRagQuery | None = None,
    ) -> None:
        """Store shared runtime dependencies and optional test overrides."""

        self.config = config or KnowledgeQAConfig.from_env()
        self.rag_config = rag_config or RagConfig.from_env()
        self.llm_config = llm_config or LLMConfig.from_env()
        self._knowledge_base_store = knowledge_base_store or JsonKnowledgeBaseStore(
            self.config.knowledge_base_store_path
        )
        self._trace_store = trace_store or JsonlRunTraceStore(
            self.config.trace_store_path
        )
        self._llm_override = llm
        self._rag_indexer_override = rag_indexer
        self._rag_retriever_override = rag_retriever
        self._llm_client: LLMClient | None = None
        self._rag_store: RagQdrantStore | None = None
        self._rag_indexer: RagIndexer | None = None
        self._rag_retriever: RagRetriever | None = None
        self._upload_store: UploadedDocumentStore | None = None

    def build_read_service(self) -> KnowledgeQAService:
        """Build a service for read-only metadata and trace access."""

        return KnowledgeQAService(
            config=self.config,
            knowledge_base_store=self._knowledge_base_store,
            trace_store=self._trace_store,
        )

    def build_ingest_service(self) -> KnowledgeQAService:
        """Build a service with ingestion capabilities."""

        return KnowledgeQAService(
            config=self.config,
            rag_indexer=self._rag_indexer_override or self._build_rag_indexer(),
            knowledge_base_store=self._knowledge_base_store,
            trace_store=self._trace_store,
        )

    def build_answer_service(self) -> KnowledgeQAService:
        """Build a service with question-answering capabilities."""

        return KnowledgeQAService(
            config=self.config,
            llm=self._llm_override or self._build_llm_client(),
            rag_retriever=self._rag_retriever_override or self._build_rag_retriever(),
            knowledge_base_store=self._knowledge_base_store,
            trace_store=self._trace_store,
        )

    def build_full_service(self) -> KnowledgeQAService:
        """Build a service with both ingestion and question-answering enabled."""

        return KnowledgeQAService(
            config=self.config,
            llm=self._llm_override or self._build_llm_client(),
            rag_retriever=self._rag_retriever_override or self._build_rag_retriever(),
            rag_indexer=self._rag_indexer_override or self._build_rag_indexer(),
            knowledge_base_store=self._knowledge_base_store,
            trace_store=self._trace_store,
        )

    def health_status(self) -> KnowledgeQAHealthStatus:
        """Return a lightweight health snapshot without touching external services."""

        return KnowledgeQAHealthStatus(
            status="ok",
            llm_model=self.llm_config.model,
            llm_provider=self.llm_config.provider,
            qdrant_configured=bool(self.rag_config.qdrant_url),
            embedding_configured=self.rag_config.embed is not None,
            knowledge_base_store_path=str(self.config.knowledge_base_store_path),
            trace_store_path=str(self.config.trace_store_path),
            upload_root_path=str(self.config.upload_root_path),
        )

    def build_upload_store(self) -> UploadedDocumentStore:
        """Build and cache the uploaded document store."""

        if self._upload_store is None:
            self._upload_store = UploadedDocumentStore(self.config.upload_root_path)
        return self._upload_store

    def _build_llm_client(self) -> LLMClient:
        """Build and cache the default LLM client."""

        if self._llm_client is None:
            self._llm_client = LLMClient(self.llm_config)
        return self._llm_client

    def _build_rag_store(self) -> RagQdrantStore:
        """Build and cache the shared RAG store."""

        if not self.rag_config.qdrant_url:
            raise RuntimeError("Knowledge QA requires QDRANT_URL to be configured.")
        if self.rag_config.embed is None:
            raise RuntimeError(
                "Knowledge QA requires embedding configuration (EMBED_*)."
            )
        if self._rag_store is None:
            self._rag_store = RagQdrantStore(self.rag_config)
        return self._rag_store

    def _build_rag_indexer(self) -> RagIndexer:
        """Build and cache the shared RAG indexer."""

        if self._rag_indexer is None:
            self._rag_indexer = RagIndexer(
                config=self.rag_config,
                store=self._build_rag_store(),
            )
        return self._rag_indexer

    def _build_rag_retriever(self) -> RagRetriever:
        """Build and cache the shared RAG retriever."""

        if self._rag_retriever is None:
            self._rag_retriever = RagRetriever(
                config=self.rag_config,
                store=self._build_rag_store(),
            )
        return self._rag_retriever
