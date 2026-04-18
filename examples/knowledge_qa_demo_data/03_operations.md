# Atlas Assistant Operations

## Deployment

Atlas Assistant runs as a small service behind an internal gateway. The API
container listens on port `8080` in the demo deployment.

The main runtime environment variables are:

- `LLM_MODEL`
- `OPENAI_API_KEY`
- `QDRANT_URL`
- `EMBED_MODEL_NAME`
- `ATLAS_PUBLIC_API_BASE_URL`

The `ATLAS_PUBLIC_API_BASE_URL` variable should be set to
`https://atlas-demo.internal/api` for the demo environment.

## Index Refresh

Embeddings are refreshed every 6 hours by a scheduled indexing job. Operators
can also trigger a manual re-index after major document updates.

## Incident Guidance

If retrieval latency spikes, operators should first check Qdrant health and
collection status. If Qdrant is healthy, they should verify the embedding
provider and network connectivity to the model endpoint.

If the answer service starts returning too many refusals, operators should
confirm that the latest documents were indexed successfully.
