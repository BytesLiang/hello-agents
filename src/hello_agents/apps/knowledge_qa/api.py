"""FastAPI application for the knowledge QA product surfaces."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, TypeVar

from dotenv import load_dotenv  # type: ignore[import-not-found]
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status

from hello_agents.apps.knowledge_qa.api_schemas import (
    AnswerResultResponse,
    AskKnowledgeBaseRequest,
    CreateKnowledgeBaseRequest,
    HealthResponse,
    KnowledgeBaseResponse,
    RunTraceResponse,
)
from hello_agents.apps.knowledge_qa.runtime import KnowledgeQARuntime
from hello_agents.apps.knowledge_qa.service import KnowledgeQAService

ServiceT = TypeVar("ServiceT", bound=KnowledgeQAService)


def create_app(runtime: KnowledgeQARuntime | None = None) -> FastAPI:
    """Create the FastAPI app for knowledge QA management and querying."""

    load_dotenv()
    app_runtime = runtime or KnowledgeQARuntime()
    app = FastAPI(
        title="hello-agents Knowledge QA API",
        version="0.1.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    @app.get("/api/health", response_model=HealthResponse)
    def get_health() -> HealthResponse:
        """Return a lightweight health response."""

        return HealthResponse.from_runtime(app_runtime.health_status())

    @app.get("/api/knowledge-bases", response_model=list[KnowledgeBaseResponse])
    def list_knowledge_bases() -> list[KnowledgeBaseResponse]:
        """List all persisted knowledge bases."""

        service = _build_service(app_runtime.build_read_service)
        return [
            KnowledgeBaseResponse.from_domain(knowledge_base)
            for knowledge_base in service.list_knowledge_bases()
        ]

    @app.get(
        "/api/knowledge-bases/{kb_id}",
        response_model=KnowledgeBaseResponse,
    )
    def get_knowledge_base(kb_id: str) -> KnowledgeBaseResponse:
        """Return one knowledge base by identifier."""

        service = _build_service(app_runtime.build_read_service)
        knowledge_base = service.get_knowledge_base(kb_id)
        if knowledge_base is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown knowledge base: {kb_id}",
            )
        return KnowledgeBaseResponse.from_domain(knowledge_base)

    @app.post(
        "/api/knowledge-bases",
        response_model=KnowledgeBaseResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_knowledge_base(
        request: CreateKnowledgeBaseRequest,
    ) -> KnowledgeBaseResponse:
        """Create and ingest one knowledge base."""

        service = _build_service(app_runtime.build_ingest_service)
        normalized_paths = _normalize_paths(request.paths)
        try:
            knowledge_base = service.ingest(
                request.name,
                [Path(path) for path in normalized_paths],
                description=request.description,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
        return KnowledgeBaseResponse.from_domain(knowledge_base)

    @app.post(
        "/api/knowledge-bases/upload",
        response_model=KnowledgeBaseResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def upload_knowledge_base(
        name: Annotated[str, Form(...)],
        files: Annotated[list[UploadFile], File(...)],
        description: Annotated[str, Form()] = "",
    ) -> KnowledgeBaseResponse:
        """Create a knowledge base by uploading local documents from the browser."""

        normalized_name = _normalize_name(name)
        upload_files = [upload for upload in files if upload.filename]
        if not upload_files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please upload at least one document.",
            )

        upload_store = app_runtime.build_upload_store()
        ingest_service = _build_service(app_runtime.build_ingest_service)
        try:
            saved_files = await upload_store.save_files(upload_files)
            knowledge_base = ingest_service.ingest(
                normalized_name,
                [saved.stored_path for saved in saved_files],
                description=description.strip(),
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
        return KnowledgeBaseResponse.from_domain(knowledge_base)

    @app.post(
        "/api/knowledge-bases/{kb_id}/ask",
        response_model=AnswerResultResponse,
    )
    def ask_knowledge_base(
        kb_id: str,
        request: AskKnowledgeBaseRequest,
    ) -> AnswerResultResponse:
        """Answer a question against one knowledge base."""

        service = _build_service(app_runtime.build_answer_service)
        if service.get_knowledge_base(kb_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown knowledge base: {kb_id}",
            )
        question = request.question.strip()
        if not question:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question must not be empty.",
            )
        try:
            result = service.ask(question, kb_id=kb_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
        return AnswerResultResponse.from_domain(result)

    @app.get("/api/traces", response_model=list[RunTraceResponse])
    def list_recent_traces(
        limit: int = Query(default=10, ge=1, le=100),
    ) -> list[RunTraceResponse]:
        """Return recent traces for UI activity views."""

        service = _build_service(app_runtime.build_read_service)
        return [
            RunTraceResponse.from_domain(trace)
            for trace in service.list_recent_traces(limit=limit)
        ]

    return app


def main() -> None:
    """Run the knowledge QA API with a local uvicorn server."""

    import uvicorn

    load_dotenv()
    uvicorn.run(
        "hello_agents.apps.knowledge_qa.api:create_app",
        factory=True,
        host="127.0.0.1",
        port=8000,
    )


def _build_service(factory: Callable[[], ServiceT]) -> ServiceT:
    """Build one runtime-backed service and map setup failures cleanly."""

    try:
        return factory()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


def _normalize_paths(paths: list[str]) -> list[str]:
    """Normalize incoming path strings and reject empty values."""

    normalized = [path.strip() for path in paths if path.strip()]
    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide at least one source path to ingest.",
        )
    return normalized


def _normalize_name(raw_name: str) -> str:
    """Normalize a knowledge base name and reject empty values."""

    normalized = raw_name.strip()
    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Knowledge base name must not be empty.",
        )
    return normalized


if __name__ == "__main__":
    main()
