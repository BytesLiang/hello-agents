# 知识库问答产品设计说明

本文档给出 `hello-agents` 在知识库问答场景下的总体设计方案，以及面向落地的分阶段需求清单。

目标不是继续堆叠抽象能力，而是基于现有的 `LLMClient`、`RAG`、`Memory`、
`ContextEngine` 等基础能力，落成一个可演示、可评测、可迭代的知识库问答系统。

## 目标

第一阶段产品目标是打通以下链路：

1. 文档导入
2. 索引构建
3. 查询改写
4. 检索召回
5. 重排与过滤
6. 上下文组装
7. 生成回答
8. 引用出处
9. 拒答降级
10. 追踪与评测

系统第一版聚焦单知识库、高质量回答和工程可观测性，暂不把 `MCP`、
`Multi-Agent` 作为主目标。

## 设计原则

- 继续在当前仓库内开发，保持 `framework + first-party app` 的 monorepo 结构。
- 通用能力继续保留在 `src/hello_agents/` 下，产品逻辑收敛到独立应用目录。
- 先做受控的 retrieval-augmented answering pipeline，再视需要复用通用 agent 能力。
- 先保证可解释、可评测、可拒答，再考虑更复杂的连接器和多 agent 编排。
- 优先沉淀产品闭环能力，而不是继续扩张抽象层。

## 总体架构

建议把知识库问答系统拆成 6 层。

### 1. Ingestion 层

负责知识导入、文档解析、切分、索引构建和去重。

可直接复用：

- `RagIndexer`
- `MarkItDown`
- embedding 构建逻辑
- Qdrant 存储能力

本层需要补充：

- 知识库元数据管理
- 文档导入任务入口
- 索引状态记录
- 文件级去重与重建策略

### 2. Retrieval 层

负责 query rewrite、hybrid retrieval、结果过滤和可选 rerank。

可直接复用：

- `RagRetriever`
- Qdrant dense + sparse hybrid 检索

本层需要补充：

- query rewrite
- top-k、来源过滤等检索控制
- 去重与阈值过滤
- 检索调试信息输出
- 可选 reranker

### 3. Answering 层

负责把检索结果渲染为可控 prompt，并生成带引用的最终回答。

第一版不建议直接依赖通用 `ChatAgent` / `ReActAgent` 循环，而是做受控链路：

1. 构造知识问答专用 prompt
2. 强约束模型只基于证据回答
3. 输出结构化结果
4. 证据不足时拒答

### 4. Knowledge Base 管理层

负责知识库实体和索引状态管理。

建议至少支持：

- 创建知识库
- 更新知识来源
- 查看文档数和 chunk 数
- 查看最近索引时间
- 删除知识库

### 5. Observability / Eval 层

负责运行追踪、结构化日志、离线评测和回归验证。

这层是知识库问答从 demo 走向产品的关键能力，需要尽早建设。

### 6. Interface 层

第一版建议先做：

- CLI
- 极简 API

先不把复杂前端作为阻塞项。

## 建议目录

建议在当前仓库新增以下目录和文件：

- `src/hello_agents/apps/knowledge_qa/`
- `src/hello_agents/eval/`
- `examples/knowledge_qa_cli.py`
- `docs/knowledge_qa_design_zh.md`
- `config/knowledge_qa.example.env`
- `tests/test_knowledge_qa.py`
- `tests/test_eval.py`

`src/hello_agents/apps/knowledge_qa/` 下建议包含：

- `service.py`
- `models.py`
- `ingest.py`
- `retrieve.py`
- `answer.py`
- `trace.py`
- `config.py`
- `store.py`

## 核心对象

### KnowledgeBase

用于描述一个知识库实体。

建议字段：

- `kb_id`
- `name`
- `description`
- `source_paths`
- `status`
- `document_count`
- `chunk_count`
- `created_at`
- `updated_at`

### RetrievedChunk

用于表达检索出的候选证据。

建议字段：

- `chunk_id`
- `source`
- `heading_path`
- `content`
- `score`
- `rerank_score`

### Citation

用于表达最终回答中的出处引用。

建议字段：

- `index`
- `source`
- `snippet`
- `chunk_id`

### AnswerResult

用于表达一次知识问答的输出结果。

建议字段：

- `answer`
- `citations`
- `confidence`
- `answered`
- `reason`
- `trace_id`

### RunTrace

用于记录一次问答请求的完整执行轨迹。

建议字段：

- `trace_id`
- `question`
- `rewritten_query`
- `retrieved_chunks`
- `selected_chunks`
- `rendered_prompt`
- `answer`
- `latency_ms`
- `token_usage`
- `failure_reason`

## 核心流程

一次问答请求的推荐链路如下：

1. 用户输入问题
2. 对问题做可选 query rewrite
3. 调用 `RagRetriever` 召回 top-k 候选 chunk
4. 对结果做去重、阈值过滤和可选 rerank
5. 把保留 chunk 渲染成带引用编号的上下文
6. 构造知识问答专用 prompt
7. 调用 LLM 生成结构化回答
8. 若证据不足，则返回拒答结果
9. 输出 `answer + citations + trace`

第一版建议尽量保持流程可控，减少开放式 agent 循环带来的不稳定性。

## 生成与引用策略

回答生成层建议约束模型遵守以下 contract：

- 只基于提供的上下文回答
- 必须引用证据编号
- 找不到依据时明确说不知道
- 不允许编造未出现的信息

建议输出结构化结果，例如：

- `answer`
- `citation_indices`
- `answered`
- `reason`

这样更利于：

- 自动化测试
- 前端展示
- 引用渲染
- 失败分析

## 拒答策略

第一版必须支持拒答能力，避免系统在证据不足时幻觉回答。

建议至少做两层拒答：

### 检索层拒答

在以下情况直接拒答：

- top-k 分数整体过低
- 有效 chunk 数不足
- 没有满足最小证据阈值的结果

### 生成层拒答

即使检索有结果，模型仍可以根据证据充分性返回：

- `answered = false`
- `reason = "insufficient_evidence"`

## 追踪与评测

### Trace

每次问答至少记录：

- 原始问题
- rewrite 后问题
- 召回结果
- 最终选中证据
- prompt 内容或摘要
- 模型输出
- 耗时
- token 统计
- 是否拒答

### Eval

第一版评测建议覆盖：

- `answer_accuracy`
- `citation_hit_rate`
- `no_answer_precision`
- `avg_latency_ms`
- `avg_total_tokens`

这层能力直接决定后续是否能系统化优化检索、prompt 和拒答策略。

## 分阶段需求清单

### 阶段 1：MVP 跑通

目标：完成单知识库问答主链路，能够本地演示。

需求：

- 支持本地目录导入并建立索引
- 支持单知识库问答
- 返回答案和引用来源
- 提供 CLI 命令入口
- 提供最小运行文档

建议命令形态：

- `ingest`
- `ask`
- `inspect`

验收标准：

- 能对一批本地文档完成索引
- 能完成知识问答演示
- 输出中包含引用文件和证据片段

### 阶段 2：回答质量提升

目标：提升结果相关性、可信度和稳定性。

需求：

- query rewrite
- rerank 或轻量重排
- chunk 去重
- 无答案拒答
- 回答结构化输出
- citation 编号与片段展示

验收标准：

- 明显减少无依据回答
- 回答和证据之间的对应关系更清晰
- 检索命中率和回答相关性优于阶段 1

### 阶段 3：工程化与可观测

目标：让系统可调试、可分析、可回归。

需求：

- `RunTrace` 持久化
- 结构化日志
- latency / token 统计
- 常见错误分类
- 核心测试补齐
- 离线评测能力

验收标准：

- 单次问答可完整追踪
- 检索和生成问题能快速定位
- 调整检索参数或 prompt 后可回归比较

### 阶段 4：产品化增强

目标：把单 demo 升级为最小可用产品。

需求：

- 多知识库管理
- 按知识库过滤
- 知识库元数据展示
- 索引状态查看
- 简单 API 或 Web UI
- 配置模板和部署说明

验收标准：

- 可切换不同知识库
- 可查看知识库规模和状态
- 可对外做稳定演示

### 阶段 5：高级能力

目标：为复杂场景和进一步扩展做准备。

需求：

- 增量索引
- 多租户与权限隔离
- 外部数据源接入
- 用户反馈闭环
- 更强的 reranker
- answer verification

这一阶段再评估是否引入：

- `MCP`：用于标准化连接外部知识源与工具
- `Multi-Agent`：用于复杂研究型问答或多角色协作

## 当前版本非目标

为避免项目过早分散，以下能力不作为第一阶段目标：

- `MCP`
- `Multi-Agent`
- 复杂工作流编排
- 大而全前端平台
- 多租户权限系统
- 在线实时同步全量数据源

## 推荐优先级

最值得优先落地的 5 项能力：

1. `KnowledgeQAService`
2. 带引用回答
3. 拒答策略
4. `RunTrace`
5. `EvalRunner`

## 一句话路线

先把当前仓库落成一个**单知识库、带引用、可拒答、可追踪、可评测**的知识库问答系统；等产品链路稳定后，再决定是否继续扩展 `MCP`、`Multi-Agent` 等高级能力。
