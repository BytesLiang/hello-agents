# Context Engineering 设计说明

本文档说明 `hello-agents` 中的 context engineering（上下文工程）层。

## 概览

`ContextEngine` 位于 memory 和 RAG 检索能力之上，负责：

- 从框架内已有来源收集上下文
- 应用轻量级预算裁剪规则
- 渲染最终发送给 LLM 的用户消息

第一版支持三类上下文来源：

- `RAG`：来自外部文档的检索结果
- `Memory`：来自工作记忆和长期记忆的检索结果
- `ToolResult`：工具执行后的近期运行时观察

最终渲染结果仍兼容现有 LLM client 的消息协议：

- `[RAG]`
- `[MEMORY]`
- `[TOOLS]`
- `User request`

## Public Interface

`context` 包当前导出：

- `ApproximateTokenEstimator`
- `ContextConfig`
- `ContextDebugInfo`
- `ContextRequest`
- `ContextSection`
- `ContextSectionTrace`
- `ContextEngine`
- `TokenEstimator`

在运行时，`ContextEngine.compose(request)` 会返回一个
`ContextEnvelope`，其中包含：

- 经过筛选后的结构化 section
- 最终渲染好的用户消息
- 一份用于调试预算与选择结果的 metadata

## 上下文组装

### RAG

- 调用 `RagRetriever.query(message)`
- 保持 retriever 原有的排序结果
- 把每个 chunk 格式化为 `source + snippet`

### Memory

- 调用 `Memory.query(message, scope=...)`
- 保持 memory 子系统原有的排序结果
- 继续按原有语义分组渲染：
  - current plan
  - session context
  - user preferences
  - confirmed facts
  - relevant task history
  - successful experience

### Tool Result

- 只使用最近的工具观察结果
- 按从新到旧的顺序排列
- 每一项格式为 `tool_name + success/failure + snippet`

## 预算控制

第一版使用轻量预算，而不是精确的 token 计数。

`ContextConfig` 负责控制：

- 是否启用各类来源
- section 的渲染顺序
- 总上下文字符数上限
- 总上下文 token 上限
- 单个 section 的字符数上限
- 单个 section 的 token 上限
- 每个 section 的最大条目数
- 单条上下文的最大字符数
- 单条上下文的最大 token 数
- 可保留的最近工具结果数量

这层同时支持可替换的 token estimator：

- 默认实现是 `ApproximateTokenEstimator`
- 调用方也可以向 `ContextEngine` 注入自定义 `TokenEstimator`

预算裁剪现在分两步进行：

1. 每个来源先独立构建 section
2. 再按 token 和字符预算做裁剪或省略

空 section 永远不会被渲染出来。

`ContextEnvelope.debug` 会暴露：

- 是否启用了 token / char 预算
- 最终选中上下文的 token / char 估算
- 最终渲染消息的 token / char 估算
- 每个 section 是否被选中，以及被裁剪或丢弃的原因

## Agent 集成

`Agent` 持有一个 `ContextEngine`，同时保留
`build_effective_message()` 作为兼容入口。

### ChatAgent

- 只在第一次 LLM 调用前组装上下文
- 工具循环继续沿用原生 `assistant` 和 `tool` 消息
- 不会把同一轮的工具结果再次注入 `[TOOLS]`，避免重复

### ReActAgent

- 在每一步推理前重新组装上下文
- 把累计的工具观察通过 `[TOOLS]` 注入
- `scratchpad` 只保留 `Thought` 和 `Action`

## 本版本非目标

- 不做精确 token 预算
- 不做跨上下文来源 reranker
- 不做 few-shot 资产管理
- 不做模型特化的渲染协议
