# hello-agents

A Python-based agent framework project scaffold.

## Quick Start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

## Knowledge QA App

The repository now includes a first-party knowledge QA product surface with:

- formal CLI: `hello-agents-knowledge-qa`
- HTTP API: FastAPI under `src/hello_agents/apps/knowledge_qa/api.py`
- Web console: React + Vite app under `frontend/knowledge-qa/`

Backend startup:

```bash
uvicorn hello_agents.apps.knowledge_qa.api:create_app --factory --reload
```

CLI examples:

```bash
hello-agents-knowledge-qa ingest \
  --name "Atlas Demo KB" \
  --paths "examples/knowledge_qa_demo_data"

hello-agents-knowledge-qa inspect
```

Frontend startup:

```bash
cd frontend/knowledge-qa
npm install
npm run dev
```

See [`docs/knowledge_qa.md`](/Users/liang/code/hello-agents/docs/knowledge_qa.md)
for API routes, frontend workflow, and end-to-end setup.

## Unified LLM Client

The project includes a lightweight unified LLM client built on top of the
official `openai` Python SDK. It supports:

- OpenAI-hosted models
- OpenAI-compatible local services such as Ollama, vLLM, and LM Studio

The example script loads a local `.env` file with `python-dotenv`. The core
library still reads from `os.environ` only.

### Environment Variables

```bash
export LLM_MODEL="gpt-4o-mini"
export OPENAI_API_KEY="your-api-key"
```

You can also create a local `.env` file:

```bash
cp .env.example .env
```

For a local OpenAI-compatible endpoint:

```bash
export LLM_MODEL="qwen2.5:14b"
export LLM_BASE_URL="http://localhost:11434/v1"
export LLM_PROVIDER="ollama"
```

### Example

```python
from hello_agents.llm import LLMClient, LLMConfig, LLMMessage

client = LLMClient(LLMConfig.from_env())
response = client.chat([LLMMessage(role="user", content="Say hello in Chinese.")])

print(response.content)
```

See [`docs/llm.md`](/Users/liang/code/hello-agents/docs/llm.md) and
[`examples/llm_chat.py`](/Users/liang/code/hello-agents/examples/llm_chat.py)
for complete usage examples.

See [`docs/memory_and_rag.md`](/Users/liang/code/hello-agents/docs/memory_and_rag.md)
for the current memory and RAG design.

## ChatAgent With Tavily

The project also includes a complete `ChatAgent + TavilySearchTool` example.

Required environment variables:

```bash
cp .env.example .env
```

Then set at least:

```bash
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your-openai-api-key
TAVILY_API_KEY=your-tavily-api-key
```

Run the example:

```bash
python examples/chat_agent_with_tavily.py \
  --prompt "Search the latest Python agent frameworks and summarize them."
```

## ChatAgent With Memory

The project also includes a `ChatAgent + LayeredMemory` example that uses the
command-style memory protocol, keeps working memory in-process, and persists
long-term memory in SQLite.

Use the same `.env` file, then run:

```bash
python examples/chat_agent_with_memory.py
```

Inside the REPL, try:

```text
I prefer concise answers. remember that my project is atlas.
Summarize my project status in one sentence.
```

## ReActAgent With Tavily

There is also a complete `ReActAgent + TavilySearchTool` example for explicit
reasoning-and-acting loops.

Use the same `.env` file:

```bash
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your-openai-api-key
TAVILY_API_KEY=your-tavily-api-key
```

Run the example:

```bash
python examples/react_agent_with_tavily.py \
  --prompt "Search the latest Python agent frameworks and summarize them."
```

## Project Structure

```text
.
├── AGENT.md
├── config/
├── docs/
├── examples/
├── frontend/
├── pyproject.toml
├── src/
│   └── hello_agents/
└── tests/
```
