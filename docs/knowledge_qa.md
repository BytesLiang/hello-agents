# Knowledge QA

本文档说明 `hello-agents` 中知识库问答产品的当前用法与运行方式。

相关实现位于：

- `src/hello_agents/apps/knowledge_qa/`
- `frontend/knowledge-qa/`
- `examples/knowledge_qa_cli.py`
- `examples/knowledge_qa_demo_data/`

设计方案见 [`knowledge_qa_design_zh.md`](/Users/liang/code/hello-agents/docs/knowledge_qa_design_zh.md)。

## 当前能力

当前版本已经具备以下最小能力：

- 本地目录或文件导入
- 基于现有 RAG 子系统建立索引
- 通过 `ask` 入口完成知识库问答
- 回答返回引用信息
- 正式 CLI 命令入口
- FastAPI HTTP API
- React + Vite Web 管理台
- Tauri 本地桌面壳
- 运行 trace 落盘到本地 JSONL
- 知识库元数据落盘到本地 JSON

当前版本仍是 MVP，尚未实现：

- query rewrite
- rerank
- 多知识库权限隔离
- 浏览器目录上传
- 异步导入任务与进度轮询
- 离线评测 runner
- Tauri 自动拉起本地 Python API

## 目录结构

知识库问答相关目录如下：

```text
src/hello_agents/apps/knowledge_qa/
├── __init__.py
├── api.py
├── api_schemas.py
├── answer.py
├── cli.py
├── config.py
├── ingest.py
├── models.py
├── retrieve.py
├── runtime.py
├── service.py
├── store.py
└── trace.py

frontend/knowledge-qa/
frontend/knowledge-qa/src-tauri/
examples/knowledge_qa_cli.py
config/knowledge_qa.example.env
tests/test_knowledge_qa.py
tests/test_knowledge_qa_api.py
tests/test_knowledge_qa_cli.py
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
- `KNOWLEDGE_QA_CORS_ORIGINS`

默认情况下：

- 知识库元数据写入 `.hello_agents/knowledge_bases.json`
- 运行 trace 写入 `.hello_agents/knowledge_qa_traces.jsonl`
- 本地 Web / Tauri 前端允许从 `localhost:5173` 和 `tauri.localhost` 访问 API

## 运行后端 API

先准备 `.env` 和 Qdrant / LLM 依赖，然后启动：

```bash
uvicorn hello_agents.apps.knowledge_qa.api:create_app --factory --reload
```

如果你希望直接从 Python 模块启动，也可以运行：

```bash
python -m hello_agents.apps.knowledge_qa.api
```

### API 路由

- `GET /api/health`
- `GET /api/knowledge-bases`
- `GET /api/knowledge-bases/{kb_id}`
- `POST /api/knowledge-bases`
- `POST /api/knowledge-bases/upload`
- `POST /api/knowledge-bases/{kb_id}/ask`
- `GET /api/traces?limit=10`

### API 示例

创建知识库：

```bash
curl -X POST http://127.0.0.1:8000/api/knowledge-bases \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Atlas Demo KB",
    "description": "Demo knowledge base",
    "paths": ["examples/knowledge_qa_demo_data"]
  }'
```

从 Web 或浏览器客户端上传本地文件创建知识库：

```bash
curl -X POST http://127.0.0.1:8000/api/knowledge-bases/upload \
  -F 'name=Atlas Upload KB' \
  -F 'description=Uploaded local docs' \
  -F 'files=@examples/knowledge_qa_demo_data/01_overview.md' \
  -F 'files=@examples/knowledge_qa_demo_data/02_architecture.md'
```

提问：

```bash
curl -X POST http://127.0.0.1:8000/api/knowledge-bases/<kb_id>/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Which vector store does Atlas use for retrieval?"
  }'
```

## CLI 用法

安装后推荐使用正式命令：

```bash
hello-agents-knowledge-qa --help
```

仓库内保留了 [examples/knowledge_qa_cli.py](/Users/liang/code/hello-agents/examples/knowledge_qa_cli.py)
作为薄兼容包装，但正式入口已经切到 console script。

### 1. 导入知识库

```bash
hello-agents-knowledge-qa ingest \
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
hello-agents-knowledge-qa ingest \
  --name "Atlas Demo KB" \
  --paths "examples/knowledge_qa_demo_data" \
  --description "Demo knowledge base for Atlas assistant"
```

### 2. 查看知识库

```bash
hello-agents-knowledge-qa inspect
```

输出示例：

```text
<kb_id> Project Docs status=ready docs=8 chunks=42
```

### 3. 提问

```bash
hello-agents-knowledge-qa ask \
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
hello-agents-knowledge-qa inspect --traces --limit 10
```

## Web 管理台

前端工程位于 `frontend/knowledge-qa/`，默认通过 Vite dev server 代理 `/api`
到本地 FastAPI 服务。

```bash
cd frontend/knowledge-qa
npm install
npm run dev
```

默认开发地址：

- 前端：`http://127.0.0.1:5173`
- 后端：`http://127.0.0.1:8000`

当前 Web 管理台包含两页：

- `/`
  - 展示知识库列表
  - 提供 `name / description / local files` 的同步导入表单
- `/knowledge-bases/:kbId`
  - 展示知识库元数据和来源路径
  - 提供问答表单
  - 展示最新回答、引用和最近 trace

注意：

- Web 端当前支持“本地文件上传”
- CLI 仍然支持“服务器本地路径”
- 上传文件会先落到 `.hello_agents/uploads/` 再复用现有 ingest 流程

## Tauri 桌面端

Tauri 桌面壳位于 `frontend/knowledge-qa/src-tauri/`，继续复用同一套 React
前端。和浏览器模式不同，桌面模式优先使用原生文件选择器返回本地路径，再调本机
FastAPI 接口完成导入。

### 启动方式

```bash
cd frontend/knowledge-qa
npm install
npm run tauri:dev
```

默认情况下，桌面壳会自动探测 `http://127.0.0.1:8000`。如果本地 API 尚未启动，
它会尝试自动拉起：

```bash
python3 -m hello_agents.apps.knowledge_qa.api --host 127.0.0.1 --port 8000
```

### 当前行为

- 浏览器模式：上传本地文件到 `/api/knowledge-bases/upload`
- Tauri 模式：先自动探测或拉起本地 Python API，再选择本地文件路径并调用 `POST /api/knowledge-bases`
- CLI 模式：继续直接传入 `--paths`

### 桌面端环境变量

如果桌面壳无法自动找到你的 Python 环境，可以显式指定：

- `KNOWLEDGE_QA_DESKTOP_PYTHON`
  - 例如：`/Users/liang/code/hello-agents/.venv/bin/python`
- `KNOWLEDGE_QA_DESKTOP_PROJECT_ROOT`
  - 指向仓库根目录，桌面端会在这里设置工作目录并补 `PYTHONPATH=src`

### 当前限制

- 需要本机安装 Rust toolchain 和 Tauri CLI 依赖
- 当前默认读取 `http://127.0.0.1:8000`
- 当前自动拉起逻辑面向本地开发和自托管场景，打包分发仍需要单独设计 Python 运行时
- 目录选择、后台进程管理、应用内升级暂未实现

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
- Web 端目前只接受服务端可访问路径
- 导入仍是同步阻塞式，没有后台任务机制
- recent traces 当前是全局视图，不按知识库过滤

1. 生成结果的更严格 schema 校验
2. `knowledge_qa` 使用示例数据集
3. 基础离线评测脚本
4. query rewrite
5. rerank
