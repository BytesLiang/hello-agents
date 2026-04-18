import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { KnowledgeBaseDetailPage } from "../routes/KnowledgeBaseDetailPage";
import { KnowledgeBaseListPage } from "../routes/KnowledgeBaseListPage";

function jsonResponse(payload, status = 200) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    headers: {
      get(name) {
        return name === "content-type" ? "application/json" : null;
      },
    },
    json: async () => payload,
    text: async () => JSON.stringify(payload),
  });
}

describe("Knowledge QA frontend", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("loads knowledge bases and submits the create form", async () => {
    const fetchMock = vi
      .spyOn(global, "fetch")
      .mockImplementationOnce(() =>
        jsonResponse([
          {
            kb_id: "kb-1",
            name: "Atlas Docs",
            description: "Project docs",
            source_paths: ["/srv/atlas"],
            status: "ready",
            document_count: 3,
            chunk_count: 12,
            created_at: "2026-04-18T12:00:00+00:00",
            updated_at: "2026-04-18T12:00:00+00:00",
          },
        ])
      )
      .mockImplementationOnce(() =>
        jsonResponse({
          kb_id: "kb-2",
          name: "Guide KB",
          description: "Guide docs",
          source_paths: ["/srv/guide"],
          status: "ready",
          document_count: 2,
          chunk_count: 8,
          created_at: "2026-04-18T13:00:00+00:00",
          updated_at: "2026-04-18T13:00:00+00:00",
        }, 201)
      )
      .mockImplementationOnce(() =>
        jsonResponse([
          {
            kb_id: "kb-2",
            name: "Guide KB",
            description: "Guide docs",
            source_paths: ["/srv/guide"],
            status: "ready",
            document_count: 2,
            chunk_count: 8,
            created_at: "2026-04-18T13:00:00+00:00",
            updated_at: "2026-04-18T13:00:00+00:00",
          },
        ])
      );

    render(
      <MemoryRouter initialEntries={["/"]}>
        <KnowledgeBaseListPage />
      </MemoryRouter>
    );

    expect(
      await screen.findByRole("link", { name: /Atlas Docs/i })
    ).toBeInTheDocument();

    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Name"), "Guide KB");
    await user.type(screen.getByLabelText("Description"), "Guide docs");
    const file = new File(["# Guide"], "guide.md", { type: "text/markdown" });
    await user.upload(screen.getByLabelText("Local Documents"), file);
    await user.click(screen.getByRole("button", { name: "Create Knowledge Base" }));

    expect(await screen.findByText("Imported Guide KB successfully.")).toBeInTheDocument();
    expect(await screen.findByText("Guide KB")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(3);
    const uploadCall = fetchMock.mock.calls[1];
    expect(uploadCall[0]).toBe("/api/knowledge-bases/upload");
    expect(uploadCall[1].method).toBe("POST");
    expect(uploadCall[1].body).toBeInstanceOf(FormData);
  });

  it("loads the detail workspace and submits a question", async () => {
    vi.spyOn(global, "fetch")
      .mockImplementationOnce(() =>
        jsonResponse({
          kb_id: "kb-1",
          name: "Atlas Docs",
          description: "Project docs",
          source_paths: ["/srv/atlas"],
          status: "ready",
          document_count: 3,
          chunk_count: 12,
          created_at: "2026-04-18T12:00:00+00:00",
          updated_at: "2026-04-18T12:00:00+00:00",
        })
      )
      .mockImplementationOnce(() =>
        jsonResponse([
          {
            trace_id: "trace-0",
            question: "What is Atlas?",
            rewritten_query: "What is Atlas?",
            retrieved_chunks: [],
            selected_chunks: [],
            rendered_prompt: "",
            answer: "Atlas is a demo workspace.",
            citations: [],
            answered: true,
            reason: null,
            latency_ms: 12,
            token_usage: {
              prompt_tokens: 3,
              completion_tokens: 4,
              total_tokens: 7,
            },
            created_at: "2026-04-18T12:10:00+00:00",
          },
        ])
      )
      .mockImplementationOnce(() =>
        jsonResponse({
          answer: "Atlas uses Qdrant for retrieval.",
          citations: [
            {
              index: 1,
              source: "/srv/atlas/architecture.md",
              snippet: "Atlas uses Qdrant for retrieval.",
              chunk_id: "chunk-1",
            },
          ],
          confidence: 0.9,
          answered: true,
          reason: null,
          trace_id: "trace-1",
        })
      )
      .mockImplementationOnce(() =>
        jsonResponse([
          {
            trace_id: "trace-1",
            question: "What vector store does Atlas use?",
            rewritten_query: "What vector store does Atlas use?",
            retrieved_chunks: [],
            selected_chunks: [],
            rendered_prompt: "",
            answer: "Atlas uses Qdrant for retrieval.",
            citations: [],
            answered: true,
            reason: null,
            latency_ms: 15,
            token_usage: {
              prompt_tokens: 5,
              completion_tokens: 6,
              total_tokens: 11,
            },
            created_at: "2026-04-18T12:11:00+00:00",
          },
        ])
      );

    render(
      <MemoryRouter initialEntries={["/knowledge-bases/kb-1"]}>
        <Routes>
          <Route
            path="/knowledge-bases/:kbId"
            element={<KnowledgeBaseDetailPage />}
          />
        </Routes>
      </MemoryRouter>
    );

    expect(
      await screen.findByRole("heading", { level: 1, name: "Atlas Docs" })
    ).toBeInTheDocument();

    const user = userEvent.setup();
    await user.type(
      screen.getByLabelText("Question"),
      "What vector store does Atlas use?"
    );
    await user.click(screen.getByRole("button", { name: "Ask Knowledge Base" }));

    expect(
      await screen.findByText("Trace ID: trace-1")
    ).toBeInTheDocument();
    expect(
      screen.getAllByText("Atlas uses Qdrant for retrieval.").length
    ).toBeGreaterThan(0);
    expect(await screen.findByText("/srv/atlas/architecture.md")).toBeInTheDocument();
  });

  it("renders API errors instead of failing silently", async () => {
    vi.spyOn(global, "fetch").mockImplementationOnce(() =>
      jsonResponse({ detail: "Backend unavailable" }, 500)
    );

    render(
      <MemoryRouter initialEntries={["/"]}>
        <KnowledgeBaseListPage />
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText("Backend unavailable")).toBeInTheDocument();
    });
  });
});
