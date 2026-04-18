# Knowledge QA Demo Data

This folder contains a small demo knowledge base for the knowledge QA CLI.

The demo is intentionally small, but it is structured enough to exercise:

- retrieval across multiple files
- section-aware chunking
- citation rendering
- no-answer handling

## Files

- `01_overview.md`: product scope and goals
- `02_architecture.md`: service architecture and storage choices
- `03_operations.md`: deployment and runbook details
- `04_release_notes.md`: version and rollout information

## Suggested Ingest Command

```bash
python examples/knowledge_qa_cli.py ingest \
  --name "Atlas Demo KB" \
  --paths "examples/knowledge_qa_demo_data" \
  --description "Demo knowledge base for Atlas assistant"
```

## Suggested Questions

- What problem does Atlas Assistant solve?
- Which vector store does Atlas use for retrieval?
- Where is long-term memory stored?
- What happens when Qdrant is unavailable?
- How often are embeddings refreshed?
- Which environment variable controls the public API base URL?
- What release introduced the citation response format?

## Questions That Should Likely Refuse

- Who is the CEO of Atlas?
- What is the exact revenue target for next year?
- Which GPU model is required in production?
