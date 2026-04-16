# Memory And RAG Design

This document describes the current memory and RAG design in `hello-agents`.
It focuses on the implementation that exists today rather than an aspirational
architecture.

## Overview

The framework currently has two retrieval layers:

- `Memory`: conversation-derived state and long-term knowledge for an agent.
- `RAG`: external document retrieval from indexed files.

They are intentionally separate:

- memory is derived from turns inside the framework
- RAG is derived from external documents

At prompt-build time, the agent can prepend both a `[RAG]` block and a
`[MEMORY]` block to the user message.

The framework now places a dedicated context-engineering layer above these
retrieval subsystems:

- memory and RAG remain responsible for retrieval
- `ContextEngine` is responsible for context assembly, budgeting, and rendering
- the final output still targets the same chat-message protocol

## Memory Design

The public entrypoint is `LayeredMemory` in
`src/hello_agents/memory/manager.py`.

### Memory API

The command-style memory interface is:

- `query(message, *, scope, kinds=None, limit=10)`
- `add(record, *, scope)`
- `update(record_id, patch, *, scope)`
- `propose(message, response, *, scope, tool_results=(), success=True)`
- `commit(proposal, *, scope)`

The agent only depends on the `Memory` protocol in
`src/hello_agents/memory/base.py`.

### Memory Types

Current memory kinds are defined in
`src/hello_agents/memory/models.py`:

- `working_plan`
- `working_context`
- `working_message`
- `semantic_preference`
- `semantic_fact`
- `episodic`
- `procedural`

Operationally they map to four groups:

- working memory: plan, context, recent messages
- semantic memory: preferences and confirmed facts
- episodic memory: task history and outcomes
- procedural memory: reusable successful patterns

### Storage Strategy

Working memory:

- stored in process by default via `InMemoryWorkingMemoryStore`
- optional Redis implementation exists
- scoped by `user_id + session_id + agent_id`
- governed by TTL and max-entry count

Long-term memory:

- stored in SQLite
- tables:
  - `semantic_preferences`
  - `semantic_facts`
  - `episodic_memories`
  - `procedural_memories`

Optional vector index:

- Qdrant is used only as an auxiliary vector index
- SQLite remains the source of truth

### Read Path

At prompt time:

1. the agent calls `memory.query(...)`
2. working memory is loaded directly from the working store
3. long-term memory is queried from SQLite by kind
4. if Qdrant is enabled, vector scores are merged into ranking
5. the context-engineering layer renders the result into a `[MEMORY]` block

The current rendering lives in `src/hello_agents/agent.py`.

### Write Path

At turn completion:

1. the agent calls `memory.propose(...)`
2. the analyzer returns:
   - working records
   - long-term candidates
3. `memory.commit(...)` applies fixed policy checks
4. accepted records are written to SQLite
5. if enabled, accepted long-term records are also indexed in Qdrant

### Extraction Strategy

The default analyzer is `RuleBasedMemoryAnalyzer` in
`src/hello_agents/memory/extractors/rule_based.py`.

Current extraction rules:

- preferences:
  - detected from explicit preference phrases
- facts:
  - detected only from explicit "remember that / confirmed / 请记住 / 已确认"
    style phrases
- episodic:
  - stores compressed `Task / Result / Tools`
- procedural:
  - only produced for successful turns
  - intended for reusable successful patterns

There is also an optional `LLMMemoryAnalyzer`, but commit policy still decides
what is allowed into long-term storage.

### Retrieval Strategy

Working memory:

- no semantic ranking
- only active entries are returned

Long-term memory:

- each kind is queried independently
- SQLite ranking is hybrid:
  - lexical overlap
  - vector score from Qdrant if enabled
  - recency boost
  - success-based boosts for episodic/procedural memory

### Retention Strategy

Working memory:

- TTL-based expiration
- max-entry limit

Episodic memory:

- query-time retention by `episodic_retention_days`

Semantic and procedural memory:

- no automatic expiry
- semantic memory uses superseding/version-like behavior

### Current Constraints

- memory prompt rendering is still done in the agent layer
- semantic extraction is intentionally conservative
- procedural memory is aggressively filtered to avoid storing generic answers

## RAG Design

The RAG implementation lives in `src/hello_agents/rag/`.

### Main Components

- `RagConfig`: runtime configuration
- `RagIndexer`: document parsing, chunking, embedding, indexing
- `RagRetriever`: query embedding and retrieval
- `RagQdrantStore`: Qdrant persistence and search
- `RagSearchTool`: tool wrapper for model-driven retrieval

### Document Ingestion

The ingestion flow is:

1. load files from a folder or a specific file path
2. convert each file to Markdown using `MarkItDown`
3. split Markdown into chunks
4. embed chunks
5. upsert chunks into Qdrant

`MarkItDown` is mandatory in the current design. The indexer does not maintain
separate raw-text and structured-parser code paths.

### Chunking Strategy

Chunking is structure-aware:

1. parse Markdown headings (`#` to `######`)
2. split by paragraph boundaries
3. preserve heading hierarchy in `heading_path`
4. if one paragraph is still too long, fall back to sliding-window chunking

Chunk metadata currently includes:

- `chunk_index`
- `path`
- `heading_path`

Default chunk parameters come from `RagConfig`:

- `chunk_size=800`
- `chunk_overlap=120`

### Vector Strategy

RAG uses hybrid retrieval with both dense and sparse vectors.

- embedding backend is shared with memory via `build_embedder()`
- Qdrant stores:
  - one dense vector named `dense`
  - one sparse vector named `sparse`
- the dense side uses cosine distance
- the sparse side uses token-based sparse vectors with Qdrant IDF modifier
- each chunk is stored as one Qdrant point with:
  - `id`
  - named vectors:
    - dense embedding
    - sparse lexical vector
  - payload:
    - `source`
    - `content`
    - `metadata`

### Query Path

There are two access paths:

- automatic augmentation:
  - `Agent.build_effective_message()` asks `ContextEngine` to retrieve top chunks
    and prepend a `[RAG]` block
- tool-driven retrieval:
  - `RagSearchTool` exposes retrieval as a callable tool

The retriever works as follows:

1. embed the query text into a dense vector
2. convert the query text into a sparse lexical vector
3. run dense and sparse prefetch queries in Qdrant
4. fuse the two candidate lists with Qdrant `Fusion.RRF`
3. return `RagChunk` objects

### Current Constraints

- only local files are indexed
- sparse retrieval is not standard BM25; it is a hashed token-frequency sparse vector
- no reranker is applied after Qdrant retrieval
- no deduplication or chunk-merging happens at retrieval time

## How Memory And RAG Work Together

Prompt construction currently follows this shape:

- `[RAG]` block first, if RAG is enabled and returns chunks
- `[MEMORY]` block second, if memory is enabled and returns content
- `[TOOLS]` block third, if recent tool observations are available
- user request last

This means:

- RAG supplies external source context
- memory supplies conversation state and accumulated agent knowledge
- tool results supply recent runtime observations

The context-engineering layer owns:

- collecting data from memory, RAG, and tool observations
- applying lightweight section and character budgets
- rendering the final prompt payload while preserving source order

They are independent subsystems and can be enabled separately.
