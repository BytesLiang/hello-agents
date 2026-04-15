# Memory 与 RAG 设计说明

本文档说明 `hello-agents` 当前的 memory 与 RAG 设计，重点描述**已经落地的实现**，而不是理想化架构。

## 总览

框架目前有两层检索能力：

- `Memory`：来自对话和执行过程的状态与长期知识
- `RAG`：来自外部文档的检索增强

两者有意分开：

- memory 来自框架内部的 turn
- RAG 来自外部文件索引

在 prompt 构建阶段，agent 可以同时在用户消息前注入：

- `[RAG]` block
- `[MEMORY]` block

## Memory 设计

公开入口是 `LayeredMemory`，实现位于
`src/hello_agents/memory/manager.py`。

### Memory API

当前使用命令式接口：

- `query(message, *, scope, kinds=None, limit=10)`
- `add(record, *, scope)`
- `update(record_id, patch, *, scope)`
- `propose(message, response, *, scope, tool_results=(), success=True)`
- `commit(proposal, *, scope)`

agent 只依赖 `Memory` protocol，定义在
`src/hello_agents/memory/base.py`。

### 记忆类型

当前的 `MemoryKind` 定义在
`src/hello_agents/memory/models.py`：

- `working_plan`
- `working_context`
- `working_message`
- `semantic_preference`
- `semantic_fact`
- `episodic`
- `procedural`

从职责上可以归为四组：

- 工作记忆：计划、上下文、最近消息
- 语义记忆：偏好与已确认事实
- 情景记忆：任务历史与执行结果
- 程序性记忆：可复用成功经验

### 存储策略

工作记忆：

- 默认用进程内存 `InMemoryWorkingMemoryStore`
- 可选 Redis
- 按 `user_id + session_id + agent_id` 做作用域隔离
- 通过 TTL 和最大条目数约束

长期记忆：

- 用 SQLite 持久化
- 表包括：
  - `semantic_preferences`
  - `semantic_facts`
  - `episodic_memories`
  - `procedural_memories`

可选向量索引：

- Qdrant 只作为辅助向量索引
- SQLite 仍然是真相源

### 读取流程

在 prompt 构建阶段：

1. agent 调用 `memory.query(...)`
2. 工作记忆直接从 working store 读取
3. 长期记忆按类型从 SQLite 查询
4. 如果启用 Qdrant，则把向量分数合并进排序
5. agent 把结果渲染为 `[MEMORY]` block

当前的渲染逻辑在 `src/hello_agents/agent.py`。

### 写入流程

在 turn 结束后：

1. agent 调用 `memory.propose(...)`
2. analyzer 产出：
   - working records
   - long-term candidates
3. `memory.commit(...)` 应用固定 policy
4. 通过的记录写入 SQLite
5. 如果启用 Qdrant，则长期记忆同步写入向量索引

### 抽取策略

默认 analyzer 是 `RuleBasedMemoryAnalyzer`，位于
`src/hello_agents/memory/extractors/rule_based.py`。

当前规则：

- preference：
  - 只从明确的偏好表达中提取
- fact：
  - 只从显式“remember that / confirmed / 请记住 / 已确认”风格语句提取
- episodic：
  - 只保存压缩后的 `Task / Result / Tools`
- procedural：
  - 只在成功 turn 中生成
  - 用于表达可复用的成功模式

另外还有可选的 `LLMMemoryAnalyzer`，但是否允许进入长期记忆仍由 commit policy 决定。

### 检索策略

工作记忆：

- 不做语义排序
- 只返回当前有效条目

长期记忆：

- 每类独立查询
- SQLite 排序是混合分数：
  - 词法重叠
  - Qdrant 向量分数（如果启用）
  - 时间新近性
  - episodic/procedural 的成功权重

### 过期策略

工作记忆：

- 基于 TTL 过期
- 同时受最大条目数限制

情景记忆：

- 按 `episodic_retention_days` 做查询期截断

语义记忆与程序性记忆：

- 默认不自动过期
- semantic 使用 supersede 方式覆盖旧版本

### 当前限制

- memory prompt 渲染仍然在 agent 层
- semantic 抽取策略偏保守
- procedural 过滤比较严格，目的是避免把泛化回答写进长期记忆

## RAG 设计

RAG 相关实现位于 `src/hello_agents/rag/`。

### 主要组件

- `RagConfig`：运行时配置
- `RagIndexer`：文档解析、分块、向量化、索引写入
- `RagRetriever`：查询向量化与检索
- `RagQdrantStore`：Qdrant 存储与搜索
- `RagSearchTool`：把检索能力暴露成 tool

### 文档导入流程

当前流程是：

1. 从目录或单个文件读取输入
2. 使用 `MarkItDown` 把文件统一转换成 Markdown
3. 对 Markdown 做分块
4. 生成 embedding
5. 把 chunk 写入 Qdrant

当前设计里，`MarkItDown` 是强制路径，不再保留单独的原始文本读取分支。

### 分块策略

当前分块是结构感知的：

1. 先识别 Markdown 标题（`#` 到 `######`）
2. 再按段落边界切分
3. 保留标题层级到 `heading_path`
4. 如果单段仍然过长，再回退到滑动窗口切分

chunk metadata 当前包括：

- `chunk_index`
- `path`
- `heading_path`

默认分块参数来自 `RagConfig`：

- `chunk_size=800`
- `chunk_overlap=120`

### 向量策略

RAG 当前使用 dense + sparse 的 hybrid 检索。

- embedding 后端与 memory 共享，通过 `build_embedder()` 构建
- Qdrant 中会同时存两类向量：
  - 名为 `dense` 的 dense embedding
  - 名为 `sparse` 的稀疏词法向量
- dense 侧使用 cosine distance
- sparse 侧使用 token-based sparse vector，并启用 Qdrant 的 IDF modifier
- 每个 chunk 作为一个 Qdrant point 写入，包含：
  - `id`
  - named vectors：
    - dense embedding
    - sparse lexical vector
  - payload：
    - `source`
    - `content`
    - `metadata`

### 查询路径

RAG 有两种接入方式：

- 自动增强：
  - `Agent.build_effective_message()` 会检索 top chunks，并注入 `[RAG]` block
- 工具调用：
  - `RagSearchTool` 允许模型在 tool calling 场景下主动检索

retriever 当前的查询过程是：

1. 对查询生成 dense embedding
2. 把查询文本转成 sparse lexical vector
3. 在 Qdrant 中分别做 dense / sparse 两路预检索
4. 用 Qdrant 的 `Fusion.RRF` 做结果融合
3. 返回 `RagChunk`

### 当前限制

- 当前只支持本地文件索引
- sparse 检索不是标准 BM25，而是 hashed token frequency sparse vector
- 检索后没有 reranker
- 检索时没有做 chunk 去重或相邻 chunk 合并

## Memory 与 RAG 如何协同

当前 prompt 构建顺序是：

- 先放 `[RAG]` block（如果启用了 RAG 且有命中）
- 再放 `[MEMORY]` block（如果启用了 memory 且有内容）
- 最后是用户原始请求

两者职责不同：

- RAG 提供外部文档上下文
- memory 提供会话状态与累计知识

因此它们是两个独立子系统，可以分别启用，也可以同时启用。
