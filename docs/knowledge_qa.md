# Knowledge QA

本文档说明 `hello-agents` 中知识库问答应用层的当前用法与运行方式。

相关实现位于：

- `src/hello_agents/apps/knowledge_qa/`
- `examples/knowledge_qa_cli.py`
- `examples/knowledge_qa_demo_data/`

设计方案见 [`knowledge_qa_design_zh.md`](/Users/liang/code/hello-agents/docs/knowledge_qa_design_zh.md)。

## 当前能力

当前版本已经具备以下最小能力：

- 本地目录或文件导入
- 基于现有 RAG 子系统建立索引
- 通过 `ask` 入口完成知识库问答
- 回答返回引用信息
- 运行 trace 落盘到本地 JSONL
- 知识库元数据落盘到本地 JSON

当前版本仍是 MVP，尚未实现：

- query rewrite
- rerank
- 多知识库权限隔离
- API / Web UI
- 离线评测 runner

## 目录结构

知识库问答相关目录如下：

```text
src/hello_agents/apps/knowledge_qa/
├── __init__.py
├── answer.py
├── config.py
├── ingest.py
├── models.py
├── retrieve.py
├── service.py
├── store.py
└── trace.py

examples/knowledge_qa_cli.py
config/knowledge_qa.example.env
tests/test_knowledge_qa.py
```

## 配置

复制配置模板：

```bash
cp config/knowledge_qa.example.env .env
```

至少需要配置：

```bash
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your-openai-api-key
QDRANT_URL=http://localhost:6333
EMBED_MODEL_TYPE=dashscope
EMBED_MODEL_NAME=text-embedding-v4
EMBED_API_KEY=your-embedding-api-key
EMBED_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

知识库问答应用层自身支持这些配置：

- `KNOWLEDGE_QA_RETRIEVAL_TOP_K`
- `KNOWLEDGE_QA_ANSWER_CONTEXT_TOP_K`
- `KNOWLEDGE_QA_MAX_CITATIONS`
- `KNOWLEDGE_QA_MIN_RETRIEVED_CHUNKS`
- `KNOWLEDGE_QA_ANSWER_MAX_TOKENS`
- `KNOWLEDGE_QA_STORE_PATH`
- `KNOWLEDGE_QA_TRACE_PATH`

默认情况下：

- 知识库元数据写入 `.hello_agents/knowledge_bases.json`
- 运行 trace 写入 `.hello_agents/knowledge_qa_traces.jsonl`

## CLI 用法

### 1. 导入知识库

```bash
python examples/knowledge_qa_cli.py ingest \
  --name "Project Docs" \
  --paths "docs,README.md" \
  --description "Project knowledge base"
```

执行后会：

1. 调用现有 `RagIndexer` 建立索引
2. 统计文档数与 chunk 数
3. 把知识库元数据写入本地 JSON

如果你想快速演示，可以直接导入仓库内的 demo 数据：

```bash
python examples/knowledge_qa_cli.py ingest \
  --name "Atlas Demo KB" \
  --paths "examples/knowledge_qa_demo_data" \
  --description "Demo knowledge base for Atlas assistant"
```

### 2. 查看知识库

```bash
python examples/knowledge_qa_cli.py inspect
```

输出示例：

```text
<kb_id> Project Docs status=ready docs=8 chunks=42
```

### 3. 提问

```bash
python examples/knowledge_qa_cli.py ask \
  --question "What does the context engine do?" \
  --kb-id "<kb_id>"
```

使用 demo 数据时，推荐先试这些问题：

- `What problem does Atlas Assistant solve?`
- `Which vector store does Atlas use for retrieval?`
- `Where is long-term memory stored?`
- `What release introduced the citation response format?`
- `Which environment variable controls the public API base URL?`

输出包括：

- 最终回答
- 引用来源
- trace id

### 4. 查看最近 trace

```bash
python examples/knowledge_qa_cli.py inspect --traces
```

## `KnowledgeQAService` 接口

应用层主入口是 `KnowledgeQAService`。

### `ingest()`

```python
knowledge_base = service.ingest(
    "Project Docs",
    paths=[Path("docs"), Path("README.md")],
    description="Project knowledge base",
)
```

返回 `KnowledgeBase`，包括：

- `kb_id`
- `name`
- `status`
- `document_count`
- `chunk_count`

### `ask()`

```python
result = service.ask(
    "What does the context engine do?",
    kb_id=knowledge_base.kb_id,
)
```

返回 `AnswerResult`，包括：

- `answer`
- `citations`
- `confidence`
- `answered`
- `reason`
- `trace_id`

## 结构化输出解析

当前 `ask()` 的生成阶段使用受控 JSON contract。

模型被要求输出：

```json
{
  "answer": "string",
  "answered": true,
  "reason": null,
  "citation_indices": [1]
}
```

服务端会做以下处理：

1. 优先把模型输出解析为 JSON object
2. 从 `citation_indices` 里解析引用编号
3. 把编号映射为对应的 chunk 引用
4. 若 JSON 解析失败，则退回到纯文本回答

当前行为约定：

- `answered=false` 时，通常返回空引用
- `citation_indices` 必须引用上下文中的编号
- 非法编号会被过滤
- 若模型返回空字符串，则结果会被标记为 `empty_answer`

## Trace

每次 `ask()` 都会记录一条 `RunTrace`。

当前 trace 包含：

- `trace_id`
- `question`
- `rewritten_query`
- `retrieved_chunks`
- `selected_chunks`
- `rendered_prompt`
- `answer`
- `citations`
- `answered`
- `reason`
- `latency_ms`
- `token_usage`

这部分数据当前以 JSONL 形式顺序追加，便于后续接入：

- 调试页面
- 离线评测
- 可观测平台

## 当前限制

当前版本有一些明确限制：

- `ask()` 仍是单轮受控问答，不是通用 agent loop
- 没有实现 query rewrite 和 rerank
- 没有做精确引用命中校验
- 没有对模型输出做严格 JSON schema 验证
- CLI 只适合开发与演示，不适合作为正式 API

## 建议的下一步

在当前基础上，优先继续补这几项：

1. 生成结果的更严格 schema 校验
2. `knowledge_qa` 使用示例数据集
3. 基础离线评测脚本
4. query rewrite
5. rerank
