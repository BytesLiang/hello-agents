# Atlas Assistant Architecture

## Retrieval Pipeline

Atlas Assistant uses a hybrid retrieval pipeline. Dense embeddings are combined
with sparse lexical features before the final candidate list is fused and sent
to the answer stage.

The vector store used for retrieval is Qdrant. It stores document chunks and
supports hybrid dense plus sparse search for the demo system.

## Storage

Long-term conversational memory is stored in SQLite. SQLite is treated as the
source of truth for durable memory records.

Qdrant is used as an auxiliary vector index for both retrieval and optional
memory ranking, but it is not the primary truth store for user facts.

## Context Assembly

The answer layer composes retrieved chunks into a numbered context block.
Answers are expected to cite supporting chunks by those numbers.

If the model output cannot be parsed as structured JSON, the service falls back
to plain text and still returns the answer when possible.

## Failure Handling

If Qdrant is unavailable during query time, Atlas Assistant degrades by
returning no retrieved context and the answer layer should refuse the question
instead of fabricating an answer.
