"""FastAPI application for the knowledge QA product surfaces."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, TypeVar

from dotenv import load_dotenv  # type: ignore[import-not-found]
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from hello_agents.apps.knowledge_qa.api_schemas import (
    AddKnowledgeBaseDocumentsRequest,
    AnswerResultResponse,
    AskKnowledgeBaseRequest,
    CreateKnowledgeBaseRequest,
    HealthResponse,
    KnowledgeBaseResponse,
    KnowledgeDocumentResponse,
    RunTraceResponse,
)
from hello_agents.apps.knowledge_qa.runtime import KnowledgeQARuntime
from hello_agents.apps.knowledge_qa.service import KnowledgeQAService

ServiceT = TypeVar("ServiceT", bound=KnowledgeQAService)
logger = logging.getLogger(__name__)
_APP_LOGGER_NAME = "hello_agents"


def create_app(runtime: KnowledgeQARuntime | None = None) -> FastAPI:
    """Create the FastAPI app for knowledge QA management and querying."""

    _configure_application_logging()
    load_dotenv()
    app_runtime = runtime or KnowledgeQARuntime()
    app = FastAPI(
        title="hello-agents Knowledge QA API",
        version="0.1.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
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
        logger.info(
            "Create knowledge base requested. name=%r path_count=%d",
            request.name,
            len(normalized_paths),
        )
        try:
            knowledge_base = service.ingest(
                request.name,
                [Path(path) for path in normalized_paths],
                description=request.description,
            )
        except ValueError as exc:
            logger.info(
                "Create knowledge base rejected. name=%r reason=%s",
                request.name,
                exc,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception(
                "Create knowledge base failed. name=%r paths=%s",
                request.name,
                normalized_paths,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
        logger.info(
            "Create knowledge base completed. kb_id=%s documents=%d chunks=%d",
            knowledge_base.kb_id,
            knowledge_base.document_count,
            knowledge_base.chunk_count,
        )
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
        logger.info(
            "Upload knowledge base requested. name=%r file_count=%d",
            normalized_name,
            len(upload_files),
        )
        try:
            saved_files = await upload_store.save_files(upload_files)
            knowledge_base = ingest_service.ingest(
                normalized_name,
                [saved.stored_path for saved in saved_files],
                description=description.strip(),
            )
        except ValueError as exc:
            logger.info(
                "Upload knowledge base rejected. name=%r reason=%s",
                normalized_name,
                exc,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception(
                "Upload knowledge base failed. name=%r filenames=%s",
                normalized_name,
                [upload.filename for upload in upload_files],
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
        logger.info(
            "Upload knowledge base completed. kb_id=%s documents=%d chunks=%d",
            knowledge_base.kb_id,
            knowledge_base.document_count,
            knowledge_base.chunk_count,
        )
        return KnowledgeBaseResponse.from_domain(knowledge_base)

    @app.get(
        "/api/knowledge-bases/{kb_id}/documents",
        response_model=list[KnowledgeDocumentResponse],
    )
    def list_knowledge_base_documents(kb_id: str) -> list[KnowledgeDocumentResponse]:
        """List the managed documents for one knowledge base."""

        service = _build_service(app_runtime.build_read_service)
        knowledge_base = service.get_knowledge_base(kb_id)
        if knowledge_base is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown knowledge base: {kb_id}",
            )
        return [
            KnowledgeDocumentResponse.from_domain(document)
            for document in knowledge_base.documents
        ]

    @app.post(
        "/api/knowledge-bases/{kb_id}/documents",
        response_model=KnowledgeBaseResponse,
    )
    def add_knowledge_base_documents(
        kb_id: str,
        request: AddKnowledgeBaseDocumentsRequest,
    ) -> KnowledgeBaseResponse:
        """Add local documents to an existing knowledge base."""

        service = _build_service(app_runtime.build_ingest_service)
        normalized_paths = _normalize_paths(request.paths)
        logger.info(
            "Add knowledge base documents requested. kb_id=%s path_count=%d",
            kb_id,
            len(normalized_paths),
        )
        try:
            knowledge_base = service.add_documents(
                kb_id,
                [Path(path) for path in normalized_paths],
            )
        except ValueError as exc:
            logger.info(
                "Add knowledge base documents rejected. kb_id=%s reason=%s",
                kb_id,
                exc,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception(
                "Add knowledge base documents failed. kb_id=%s paths=%s",
                kb_id,
                normalized_paths,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
        logger.info(
            "Add knowledge base documents completed. kb_id=%s documents=%d chunks=%d",
            knowledge_base.kb_id,
            knowledge_base.document_count,
            knowledge_base.chunk_count,
        )
        return KnowledgeBaseResponse.from_domain(knowledge_base)

    @app.post(
        "/api/knowledge-bases/{kb_id}/documents/upload",
        response_model=KnowledgeBaseResponse,
    )
    async def upload_knowledge_base_documents(
        kb_id: str,
        files: Annotated[list[UploadFile], File(...)],
    ) -> KnowledgeBaseResponse:
        """Upload and add browser-supplied documents to one knowledge base."""

        upload_files = [upload for upload in files if upload.filename]
        if not upload_files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please upload at least one document.",
            )

        upload_store = app_runtime.build_upload_store()
        ingest_service = _build_service(app_runtime.build_ingest_service)
        logger.info(
            "Upload knowledge base documents requested. kb_id=%s file_count=%d",
            kb_id,
            len(upload_files),
        )
        try:
            saved_files = await upload_store.save_files(upload_files)
            knowledge_base = ingest_service.add_documents(
                kb_id,
                [saved.stored_path for saved in saved_files],
            )
        except ValueError as exc:
            logger.info(
                "Upload knowledge base documents rejected. kb_id=%s reason=%s",
                kb_id,
                exc,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception(
                "Upload knowledge base documents failed. kb_id=%s filenames=%s",
                kb_id,
                [upload.filename for upload in upload_files],
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
        logger.info(
            "Upload knowledge base documents completed. "
            "kb_id=%s documents=%d chunks=%d",
            knowledge_base.kb_id,
            knowledge_base.document_count,
            knowledge_base.chunk_count,
        )
        return KnowledgeBaseResponse.from_domain(knowledge_base)

    @app.delete(
        "/api/knowledge-bases/{kb_id}/documents/{document_id}",
        response_model=KnowledgeBaseResponse,
    )
    def delete_knowledge_base_document(
        kb_id: str,
        document_id: str,
    ) -> KnowledgeBaseResponse:
        """Delete one managed document from a knowledge base."""

        service = _build_service(app_runtime.build_ingest_service)
        logger.info(
            "Delete knowledge base document requested. kb_id=%s document_id=%s",
            kb_id,
            document_id,
        )
        try:
            knowledge_base = service.remove_document(kb_id, document_id)
        except ValueError as exc:
            logger.info(
                "Delete knowledge base document rejected. kb_id=%s reason=%s",
                kb_id,
                exc,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception(
                "Delete knowledge base document failed. kb_id=%s document_id=%s",
                kb_id,
                document_id,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
        logger.info(
            "Delete knowledge base document completed. kb_id=%s documents=%d chunks=%d",
            knowledge_base.kb_id,
            knowledge_base.document_count,
            knowledge_base.chunk_count,
        )
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
        logger.info(
            "Ask knowledge base requested. kb_id=%s question_length=%d",
            kb_id,
            len(question),
        )
        try:
            result = service.ask(question, kb_id=kb_id)
        except ValueError as exc:
            logger.info(
                "Ask knowledge base rejected. kb_id=%s reason=%s",
                kb_id,
                exc,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception("Ask knowledge base failed. kb_id=%s", kb_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
        logger.info(
            "Ask knowledge base completed. kb_id=%s answered=%s trace_id=%s",
            kb_id,
            result.answered,
            result.trace_id,
        )
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


def main(argv: list[str] | None = None) -> None:
    """Run the knowledge QA API with a local uvicorn server."""

    import uvicorn

    _configure_application_logging()
    load_dotenv()
    args = _parse_args(argv)
    uvicorn.run(
        "hello_agents.apps.knowledge_qa.api:create_app",
        factory=True,
        host=args.host,
        port=args.port,
    )


def _build_service(factory: Callable[[], ServiceT]) -> ServiceT:
    """Build one runtime-backed service and map setup failures cleanly."""

    try:
        return factory()
    except Exception as exc:
        logger.exception("Build knowledge QA service failed.")
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


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the local server startup arguments."""

    parser = argparse.ArgumentParser(
        prog="python -m hello_agents.apps.knowledge_qa.api",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def _cors_origins() -> list[str]:
    """Return the allowed frontend origins for local web and desktop clients."""

    configured = [
        origin.strip()
        for origin in os.getenv("KNOWLEDGE_QA_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    if configured:
        return configured

    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://tauri.localhost",
        "https://tauri.localhost",
        "tauri://localhost",
    ]


def _configure_application_logging() -> None:
    """Ensure hello_agents logs are emitted to stderr for desktop capture."""

    app_logger = logging.getLogger(_APP_LOGGER_NAME)
    app_logger.setLevel(logging.INFO)

    if any(
        isinstance(handler, logging.StreamHandler)
        and getattr(handler, "_hello_agents_handler", False)
        for handler in app_logger.handlers
    ):
        return

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    )
    handler._hello_agents_handler = True  # type: ignore[attr-defined]
    app_logger.addHandler(handler)
    app_logger.propagate = False


if __name__ == "__main__":
    main()
