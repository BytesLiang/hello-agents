# Context Engineering Design

This document describes the context-engineering layer in `hello-agents`.

## Overview

`ContextEngine` sits above memory and RAG retrieval. Its job is to:

- collect context from framework-owned sources
- apply lightweight budgeting rules
- render the final user message sent to the LLM

The first implementation supports three sources:

- `RAG`: retrieved external document chunks
- `Memory`: retrieved working and long-term memory
- `ToolResult`: recent runtime observations from executed tools

The rendered prompt remains compatible with the existing LLM client:

- `[RAG]`
- `[MEMORY]`
- `[TOOLS]`
- `User request`

## Public Interfaces

The context package exports:

- `ApproximateTokenEstimator`
- `ContextConfig`
- `ContextDebugInfo`
- `ContextRequest`
- `ContextSection`
- `ContextSectionTrace`
- `ContextEngine`
- `TokenEstimator`

Operationally, `ContextEngine.compose(request)` returns a `ContextEnvelope`
containing:

- the selected structured sections
- the final rendered user message
- debug metadata describing budgets, selected sections, and drop reasons

## Source Assembly

### RAG

- calls `RagRetriever.query(message)`
- preserves retriever ordering
- formats each chunk as `source + snippet`

### Memory

- calls `Memory.query(message, scope=...)`
- preserves memory subsystem ordering
- renders the same semantic groups as before:
  - current plan
  - session context
  - user preferences
  - confirmed facts
  - relevant task history
  - successful experience

### Tool Results

- uses the most recent tool observations only
- orders them from newest to oldest
- formats each item as `tool_name + success/failure + snippet`

## Budgeting

The first version uses lightweight budgeting rather than exact token counting.

`ContextConfig` controls:

- enabled sources
- source render order
- max total context characters
- max total context tokens
- max section characters
- max section tokens
- max items per section
- max item characters
- max item tokens
- max remembered tool results

The engine also supports a replaceable token estimator:

- `ApproximateTokenEstimator` is the default
- callers may inject a custom `TokenEstimator` into `ContextEngine`

Budgeting now happens in two stages:

1. build sections independently per source
2. trim or omit sections to satisfy token and character budgets

Empty sections are never rendered.

`ContextEnvelope.debug` exposes:

- whether token and character budgets were applied
- estimated tokens and characters for the selected context
- estimated tokens and characters for the rendered message
- per-section traces with selection status and drop reasons

## Agent Integration

`Agent` owns a `ContextEngine` and keeps `build_effective_message()` as the
compatibility entrypoint.

### ChatAgent

- composes context once before the first LLM call
- continues the tool loop with native `assistant` and `tool` messages
- does not re-inject same-turn tool results into a `[TOOLS]` block

### ReActAgent

- recomposes context before each reasoning step
- includes accumulated tool observations via `[TOOLS]`
- keeps the scratchpad limited to `Thought` and `Action`

## Non-Goals In This Version

- no exact token accounting
- no reranker across context sources
- no few-shot asset management
- no model-specific rendering protocols
