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

## Unified LLM Client

The project includes a lightweight unified LLM client built on top of the
official `openai` Python SDK. It supports:

- OpenAI-hosted models
- OpenAI-compatible local services such as Ollama, vLLM, and LM Studio

### Environment Variables

```bash
export LLM_MODEL="gpt-4o-mini"
export OPENAI_API_KEY="your-api-key"
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

## Project Structure

```text
.
├── AGENT.md
├── config/
├── docs/
├── examples/
├── pyproject.toml
├── src/
│   └── hello_agents/
└── tests/
```
