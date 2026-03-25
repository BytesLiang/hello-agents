# Unified LLM Client

This project exposes a small LLM abstraction built on top of
`from openai import OpenAI`.

The intent is to keep the project-level API stable while reusing the existing
OpenAI SDK for both hosted and OpenAI-compatible local model endpoints.

## Supported Usage

- OpenAI-hosted models
- Local OpenAI-compatible services
- Simple non-streaming chat completion
- Simple streaming text output

## Configuration

Use [`LLMConfig`](/Users/liang/code/hello-agents/src/hello_agents/llm/config.py)
to configure the client directly, or load it from environment variables.
The library itself does not auto-read `.env`; the example script does.

### Environment Variables

- `LLM_MODEL`: model name, defaults to `gpt-4o-mini`
- `LLM_API_KEY`: explicit API key override
- `OPENAI_API_KEY`: fallback API key for hosted OpenAI usage
- `LLM_BASE_URL`: optional custom endpoint URL
- `LLM_TIMEOUT`: request timeout in seconds, defaults to `30`
- `LLM_PROVIDER`: optional label for the endpoint, used for bookkeeping only

If no API key is provided, the client passes `EMPTY` to the SDK. This is useful
for local OpenAI-compatible services that ignore API keys.

## .env Support

For local development, the example script calls `load_dotenv()` before reading
configuration, so values from a project `.env` file become visible through
`os.environ`.

Example `.env`:

```bash
LLM_MODEL=qwen2.5:14b
LLM_BASE_URL=http://localhost:11434/v1
LLM_PROVIDER=ollama
LLM_API_KEY=EMPTY
```

## Direct Usage

```python
from hello_agents.llm import LLMClient, LLMConfig, LLMMessage

client = LLMClient(
    LLMConfig(
        model="gpt-4o-mini",
        api_key="your-api-key",
    )
)

response = client.chat(
    [
        LLMMessage(role="system", content="You are a concise assistant."),
        LLMMessage(role="user", content="Summarize the purpose of this project."),
    ],
    temperature=0.2,
    max_tokens=128,
)

print(response.content)
```

## Local Model Usage

Ollama, vLLM, and LM Studio can be used through the same client as long as they
expose an OpenAI-compatible API.

```python
from hello_agents.llm import LLMClient, LLMConfig, LLMMessage

client = LLMClient(
    LLMConfig(
        model="qwen2.5:14b",
        base_url="http://localhost:11434/v1",
        provider="ollama",
    )
)

response = client.chat([LLMMessage(role="user", content="Explain RAG briefly.")])
print(response.content)
```

## Streaming

```python
from hello_agents.llm import LLMClient, LLMConfig, LLMMessage

client = LLMClient(LLMConfig.from_env())

for chunk in client.stream(
    [LLMMessage(role="user", content="Count from 1 to 5.")],
    temperature=0,
):
    print(chunk, end="", flush=True)
print()
```

## Example Script

Run the packaged example:

```bash
python examples/llm_chat.py --prompt "Say hello"
```
