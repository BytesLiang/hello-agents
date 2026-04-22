# Knowledge QA 文档管理助手 — 多客户端构建与使用文档

> 版本：0.1.0 · 最后更新：2026-04-19

---

## 目录

1. [产品概述](#1-产品概述)
2. [环境准备](#2-环境准备)
3. [构建指南](#3-构建指南)
4. [安装说明](#4-安装说明)
5. [使用指南](#5-使用指南)
6. [配置说明](#6-配置说明)
7. [故障排除](#7-故障排除)

---

## 1. 产品概述

### 1.1 产品定位

Knowledge QA 是一款基于 RAG（检索增强生成）技术的文档管理助手，面向需要将本地文档转化为可问答知识库的个人和团队用户。用户导入文档后，可通过自然语言提问，系统基于文档内容生成带引用的结构化回答。

### 1.2 核心功能

| 功能 | 说明 |
|------|------|
| **文档导入与索引** | 支持本地路径、浏览器上传、桌面端原生选择器三种方式导入文档；自动完成格式转换、智能切分、向量嵌入和混合索引构建 |
| **知识库问答** | 基于检索增强生成（RAG），对知识库进行自然语言提问，返回带引用编号的结构化回答 |
| **混合检索** | 同时使用 dense 向量（语义相似度）和 sparse 向量（BM25 关键词匹配），通过 RRF 融合排序提升召回质量 |
| **引用追踪** | 回答附带 citation_indices，映射到具体文档片段，支持来源验证 |
| **拒答机制** | 检索结果不足或证据不充分时，系统主动拒答并给出原因 |
| **运行追踪** | 每次问答自动记录完整 RunTrace（含检索过程、Token 用量、延迟等），持久化到本地 JSONL |
| **知识库管理** | 创建、查看、列举知识库，展示元数据（文档数、chunk 数、状态、来源路径等） |
| **健康检查** | 一键查看服务运行状态、LLM/嵌入模型/Qdrant 配置是否就绪 |

### 1.3 支持的文档格式

| 格式 | 扩展名 |
|------|--------|
| Markdown | `.md`, `.markdown` |
| 纯文本 | `.txt` |
| reStructuredText | `.rst` |
| JSON | `.json` |
| YAML | `.yaml`, `.yml` |
| HTML | `.html`, `.htm` |
| PDF | `.pdf` |

### 1.4 多客户端支持

| 客户端 | 技术栈 | 适用场景 |
|--------|--------|----------|
| **CLI 命令行** | Python | 自动化脚本、服务器环境、CI/CD 集成 |
| **Web 管理台** | React + Vite | 浏览器访问、团队协作、快速体验 |
| **Tauri 桌面端** | React + Rust + Tauri 2 | 本地桌面使用、原生文件选择、离线友好 |

### 1.5 系统架构概览

```
┌─────────────────────────────────────────────────────────┐
│                      客户端层                            │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ CLI 命令行 │  │ Web 管理台    │  │ Tauri 桌面端      │  │
│  │ (Python)  │  │ (React+Vite) │  │ (React+Rust)     │  │
│  └─────┬─────┘  └──────┬───────┘  └────────┬──────────┘  │
│        │               │                    │             │
│        │               │  HTTP API          │ HTTP API    │
│        │               │  (Vite Proxy)      │ (Direct)    │
├────────┼───────────────┼────────────────────┼─────────────┤
│        │               │                    │             │
│  ┌─────▼───────────────▼────────────────────▼──────────┐  │
│  │              FastAPI HTTP API 服务端                  │  │
│  │         (hello_agents.apps.knowledge_qa.api)        │  │
│  └──────────────────────┬──────────────────────────────┘  │
│                         │                                 │
│  ┌──────────────────────▼──────────────────────────────┐  │
│  │            KnowledgeQAService 服务层                  │  │
│  │   (ingest / ask / list / trace / health)            │  │
│  └───┬──────────┬──────────┬──────────┬────────────────┘  │
│      │          │          │          │                    │
│  ┌───▼───┐ ┌───▼───┐ ┌───▼───┐ ┌───▼──────┐             │
│  │ LLM   │ │ RAG   │ │ Store │ │ Trace    │             │
│  │Client │ │Indexer│ │(JSON) │ │(JSONL)   │             │
│  │       │ │Retriev│ │       │ │          │             │
│  └───┬───┘ └───┬───┘ └───────┘ └──────────┘             │
│      │         │                                          │
├──────┼─────────┼──────────────────────────────────────────┤
│  ┌───▼───┐ ┌───▼──────────┐                               │
│  │OpenAI │ │ Qdrant       │     外部依赖层                  │
│  │/兼容  │ │ (Dense+Sparse)│                               │
│  │API    │ │              │                               │
│  └───────┘ └──────────────┘                               │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 环境准备

### 2.1 通用系统要求

| 组件 | 最低版本 | 说明 |
|------|----------|------|
| Python | 3.11+ | 与 `.python-version` 保持一致 |
| Node.js | 18+ | 前端构建和开发 |
| Rust | 1.70+ | 仅桌面端构建需要 |
| Qdrant | 1.7+ | 向量数据库服务 |
| Git | 2.30+ | 代码获取 |

### 2.2 外部服务依赖

| 服务 | 用途 | 必需 |
|------|------|------|
| LLM API（OpenAI 或兼容服务） | 问答生成 | 是 |
| Qdrant | 向量存储与混合检索 | 是 |
| 嵌入模型 API（如 DashScope） | 文档向量化 | 是 |

### 2.3 各客户端额外要求

#### CLI 命令行

- Python 3.11+
- pip 或 uv 包管理器

#### Web 管理台

- Node.js 18+
- npm 9+

#### Tauri 桌面端

- Node.js 18+、npm 9+
- Rust 工具链（通过 [rustup](https://rustup.rs/) 安装）
- 平台特定依赖：
  - **macOS**：Xcode Command Line Tools（`xcode-select --install`）
  - **Windows**：Microsoft Visual Studio C++ Build Tools
  - **Linux**：`libwebkit2gtk-4.1-dev`、`build-essential`、`libssl-dev`、`libgtk-3-dev`、`libayatana-appindicator3-dev`、`librsvg2-dev`

### 2.4 前置安装步骤

#### 2.4.1 获取代码

```bash
git clone <repository-url> hello-agents
cd hello-agents
```

#### 2.4.2 创建 Python 虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```

#### 2.4.3 安装 Python 依赖

```bash
pip install -e ".[dev]"
```

该命令安装项目运行时依赖及开发工具（pytest、ruff、mypy 等）。

#### 2.4.4 启动 Qdrant

**Docker 方式（推荐）：**

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant:latest
```

**本地安装方式：**

参考 [Qdrant 官方文档](https://qdrant.tech/documentation/guides/installation/)。

#### 2.4.5 配置环境变量

```bash
cp config/knowledge_qa.example.env .env
```

编辑 `.env` 文件，填入实际的 API Key 和服务地址（详见[第 6 节](#6-配置说明)）。

---

## 3. 构建指南

### 3.1 服务端部署

#### 3.1.1 开发模式启动

```bash
source .venv/bin/activate
uvicorn hello_agents.apps.knowledge_qa.api:create_app --factory --reload
```

参数说明：

| 参数 | 说明 |
|------|------|
| `--factory` | 告知 uvicorn 使用工厂函数创建 app |
| `--reload` | 代码变更时自动重载（仅开发环境） |
| `--host` | 绑定地址，默认 `127.0.0.1` |
| `--port` | 绑定端口，默认 `8000` |

#### 3.1.2 模块方式启动

```bash
python -m hello_agents.apps.knowledge_qa.api
```

支持 `--host` 和 `--port` 参数：

```bash
python -m hello_agents.apps.knowledge_qa.api --host 0.0.0.0 --port 8080
```

#### 3.1.3 生产部署建议

使用 `uvicorn` 或 `gunicorn + uvicorn worker`：

```bash
pip install gunicorn
gunicorn hello_agents.apps.knowledge_qa.api:create_app \
  --factory \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

> **注意**：当前版本导入操作为同步阻塞式，多 worker 可提升并发问答能力，但不会加速单次导入。

#### 3.1.4 验证服务启动

```bash
curl http://127.0.0.1:8000/api/health
```

正常返回示例：

```json
{
  "status": "ok",
  "llm_model": "gpt-4o-mini",
  "llm_provider": "openai",
  "qdrant_configured": true,
  "embedding_configured": true,
  "knowledge_base_store_path": ".hello_agents/knowledge_bases.json",
  "trace_store_path": ".hello_agents/knowledge_qa_traces.jsonl",
  "upload_root_path": ".hello_agents/uploads"
}
```

### 3.2 Web 管理台构建

#### 3.2.1 开发环境搭建

```bash
cd frontend/knowledge-qa
npm install
```

#### 3.2.2 开发模式运行

```bash
npm run dev
```

Vite 开发服务器默认在 `http://127.0.0.1:5173` 启动，自动将 `/api` 请求代理到 `http://127.0.0.1:8000`。

代理配置位于 [vite.config.js](../frontend/knowledge-qa/vite.config.js)：

```javascript
server: {
  port: 5173,
  proxy: {
    "/api": {
      target: "http://127.0.0.1:8000",
      changeOrigin: true,
    },
  },
}
```

#### 3.2.3 生产构建

```bash
npm run build
```

构建产物输出到 `frontend/knowledge-qa/dist/`，可直接由任何静态文件服务器托管。

#### 3.2.4 预览生产构建

```bash
npm run preview
```

#### 3.2.5 连接远程后端

Web 管理台默认通过 Vite 代理连接本地后端。如需连接远程后端，设置环境变量：

```bash
VITE_KNOWLEDGE_QA_API_BASE_URL=https://your-server.example.com npm run dev
```

或在项目根目录创建 `.env.local`：

```
VITE_KNOWLEDGE_QA_API_BASE_URL=https://your-server.example.com
```

> **注意**：远程后端需配置 `KNOWLEDGE_QA_CORS_ORIGINS` 允许前端来源。

### 3.3 Tauri 桌面端构建

#### 3.3.1 安装 Rust 工具链

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

#### 3.3.2 开发模式运行

```bash
cd frontend/knowledge-qa
npm install
npm run tauri:dev
```

Tauri 开发模式会：
1. 启动 Vite 开发服务器（前端热更新）
2. 编译 Rust 后端
3. 打开桌面窗口

桌面端启动时会自动探测 `http://127.0.0.1:8000`，若后端不可达则自动拉起本地 Python API 进程。

#### 3.3.3 生产构建与打包

```bash
cd frontend/knowledge-qa
npm run tauri:build
```

构建产物位于 `frontend/knowledge-qa/src-tauri/target/release/bundle/`：

| 平台 | 产物格式 |
|------|----------|
| macOS | `.dmg`、`.app` |
| Windows | `.msi`、`.exe` |
| Linux | `.deb`、`.AppImage` |

#### 3.3.4 Tauri 构建配置

核心配置位于 [tauri.conf.json](../frontend/knowledge-qa/src-tauri/tauri.conf.json)：

```json
{
  "productName": "Knowledge QA",
  "version": "0.1.0",
  "identifier": "com.helloagents.knowledgeqa",
  "build": {
    "beforeDevCommand": "npm run dev",
    "beforeBuildCommand": "npm run build",
    "devUrl": "http://localhost:5173",
    "frontendDist": "../dist"
  },
  "app": {
    "windows": [
      {
        "title": "Knowledge QA",
        "label": "main",
        "width": 1400,
        "height": 960,
        "resizable": true
      }
    ]
  },
  "bundle": {
    "active": true,
    "targets": "all"
  }
}
```

Rust Release 优化配置位于 [Cargo.toml](../frontend/knowledge-qa/src-tauri/Cargo.toml)：

```toml
[profile.release]
lto = true
codegen-units = 1
panic = "abort"
```

#### 3.3.5 桌面端 Python 环境探测

桌面端启动后端时按以下优先级探测 Python：

1. 环境变量 `KNOWLEDGE_QA_DESKTOP_PYTHON` 指定的路径
2. 项目根目录下 `.venv/bin/python`
3. 系统 `python3`

项目根目录探测优先级：

1. 环境变量 `KNOWLEDGE_QA_DESKTOP_PROJECT_ROOT`
2. 从 `CARGO_MANIFEST_DIR` 向上推导两级目录

#### 3.3.6 桌面端后端日志

桌面端后端进程日志写入：

```
.hello_agents/logs/knowledge_qa_desktop_backend.log
```

### 3.4 构建参数汇总

#### Python 服务端

| 参数 / 环境变量 | 默认值 | 说明 |
|------------------|--------|------|
| `--host` | `127.0.0.1` | API 监听地址 |
| `--port` | `8000` | API 监听端口 |
| `PYTHONPATH` | `src` | Python 模块搜索路径 |

#### Web 前端

| 参数 / 环境变量 | 默认值 | 说明 |
|------------------|--------|------|
| `VITE_KNOWLEDGE_QA_API_BASE_URL` | 空（使用 Vite 代理） | 后端 API 基础 URL |
| `--port` (vite) | `5173` | 开发服务器端口 |

#### Tauri 桌面端

| 参数 / 环境变量 | 默认值 | 说明 |
|------------------|--------|------|
| `KNOWLEDGE_QA_DESKTOP_PYTHON` | 自动探测 | Python 可执行文件路径 |
| `KNOWLEDGE_QA_DESKTOP_PROJECT_ROOT` | 自动探测 | 项目根目录路径 |

### 3.5 常见构建问题

#### 问题：`pip install -e ".[dev]"` 失败

**原因**：Python 版本不满足要求或虚拟环境未激活。

**解决**：

```bash
python3 --version  # 确认 >= 3.11
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

#### 问题：`npm install` 报错依赖冲突

**解决**：

```bash
rm -rf node_modules package-lock.json
npm install
```

#### 问题：Tauri 构建失败 "Cargo not found"

**解决**：安装 Rust 工具链后重启终端：

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

#### 问题：Tauri macOS 构建报 Xcode 相关错误

**解决**：

```bash
xcode-select --install
```

#### 问题：Tauri Linux 构建缺少系统库

**解决**（Debian/Ubuntu）：

```bash
sudo apt update
sudo apt install libwebkit2gtk-4.1-dev build-essential libssl-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev
```

---

## 4. 安装说明

### 4.1 CLI 命令行安装

#### 从源码安装

```bash
cd hello-agents
source .venv/bin/activate
pip install -e .
```

安装完成后，`hello-agents-knowledge-qa` 命令可用：

```bash
hello-agents-knowledge-qa --help
```

#### 验证安装

```bash
hello-agents-knowledge-qa inspect
```

### 4.2 Web 管理台安装

Web 管理台无需独立安装，通过以下方式访问：

#### 开发模式

```bash
# 终端 1：启动后端
source .venv/bin/activate
python -m hello_agents.apps.knowledge_qa.api

# 终端 2：启动前端
cd frontend/knowledge-qa
npm install
npm run dev
```

浏览器访问 `http://127.0.0.1:5173`。

#### 生产部署

1. 构建前端静态资源：

```bash
cd frontend/knowledge-qa
npm run build
```

2. 使用 Nginx 等反向服务器托管 `dist/` 目录，并将 `/api` 请求代理到后端。

Nginx 配置示例：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    root /path/to/hello-agents/frontend/knowledge-qa/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 4.3 Tauri 桌面端安装

#### 从源码构建安装包

```bash
cd frontend/knowledge-qa
npm install
npm run tauri:build
```

安装包位置：

| 平台 | 路径 |
|------|------|
| macOS | `src-tauri/target/release/bundle/dmg/Knowledge QA_0.1.0_aarch64.dmg` |
| Windows | `src-tauri/target/release/bundle/msi/Knowledge QA_0.1.0_x64_en-US.msi` |
| Linux | `src-tauri/target/release/bundle/deb/knowledge-qa-desktop_0.1.0_amd64.deb` |

#### 安装步骤

**macOS：**

1. 双击 `.dmg` 文件
2. 将 `Knowledge QA.app` 拖入 `Applications` 文件夹
3. 首次启动可能需要在"系统设置 > 隐私与安全性"中允许运行

**Windows：**

1. 双击 `.msi` 安装包
2. 按安装向导完成安装
3. 从开始菜单启动

**Linux：**

```bash
sudo dpkg -i knowledge-qa-desktop_0.1.0_amd64.deb
```

或使用 AppImage：

```bash
chmod +x knowledge-qa-desktop_0.1.0_amd64.AppImage
./knowledge-qa-desktop_0.1.0_amd64.AppImage
```

> **重要**：桌面端运行需要本机已安装 Python 3.11+ 并配置好项目环境（含 `.env` 和 Qdrant 连接）。

---

## 5. 使用指南

### 5.1 CLI 命令行使用

#### 5.1.1 导入知识库

```bash
hello-agents-knowledge-qa ingest \
  --name "项目文档库" \
  --paths "docs,README.md" \
  --description "项目相关知识库"
```

参数说明：

| 参数 | 必需 | 说明 |
|------|------|------|
| `--name` | 是 | 知识库名称 |
| `--paths` | 是 | 逗号分隔的文件或目录路径列表 |
| `--description` | 否 | 知识库描述 |

执行流程：

1. 扫描指定路径下的文档文件
2. 使用 MarkItDown 将文档转换为 Markdown
3. 按标题层级和段落智能切分为 chunk
4. 生成向量嵌入并存入 Qdrant
5. 将知识库元数据写入本地 JSON

快速演示（使用内置 demo 数据）：

```bash
hello-agents-knowledge-qa ingest \
  --name "Atlas Demo KB" \
  --paths "examples/knowledge_qa_demo_data" \
  --description "Demo knowledge base"
```

#### 5.1.2 查看知识库列表

```bash
hello-agents-knowledge-qa inspect
```

输出示例：

```
a1b2c3d4e5f6 Project Docs status=ready docs=8 chunks=42
f6e5d4c3b2a1 Atlas Demo KB status=ready docs=4 chunks=18
```

#### 5.1.3 提问

```bash
hello-agents-knowledge-qa ask \
  --question "What does the context engine do?" \
  --kb-id <kb_id>
```

参数说明：

| 参数 | 必需 | 说明 |
|------|------|------|
| `--question` | 是 | 要提问的问题 |
| `--kb-id` | 否 | 知识库 ID，用于来源过滤 |

输出示例：

```
The context engine assembles memory, RAG, and tool observations into a
budget-constrained prompt context, supporting section-level and item-level
character/token budget control.

Citations:
[1] docs/context_engine.md - The context engine assembles memory, RAG, and tool observations into a budget-constrained prompt context...
[2] docs/architecture.md - Context Engine: Responsible for prompt assembly with budget control...

Trace: a1b2c3d4e5f67890
```

推荐试用问题（使用 demo 数据）：

- `What problem does Atlas Assistant solve?`
- `Which vector store does Atlas use for retrieval?`
- `Where is long-term memory stored?`
- `What release introduced the citation response format?`
- `Which environment variable controls the public API base URL?`

#### 5.1.4 查看运行追踪

```bash
hello-agents-knowledge-qa inspect --traces --limit 10
```

输出示例：

```
a1b2c3d4 2026-04-19T10:30:00 answered=True
Q: What does the context engine do?
A: The context engine assembles memory, RAG, and tool observations...
```

### 5.2 Web 管理台使用

#### 5.2.1 访问管理台

启动后端和前端后，浏览器访问 `http://127.0.0.1:5173`。

#### 5.2.2 创建知识库

1. 在首页左侧 **Create Knowledge Base** 面板中填写：
   - **Name**：知识库名称（必填）
   - **Description**：知识库描述（选填）
   - **Local Documents**：点击文件选择器，选择一个或多个本地文档
2. 点击 **Create Knowledge Base** 按钮
3. 等待索引完成，页面自动刷新知识库列表

> **注意**：Web 端通过浏览器文件上传方式导入文档，文件先保存到 `.hello_agents/uploads/` 再进行索引。

#### 5.2.3 查看知识库列表

首页右侧 **Knowledge Bases** 面板展示所有知识库卡片，包含：

- 知识库名称和描述
- 状态标签（`indexing` / `ready` / `failed`）
- 文档数和 chunk 数
- 最后更新时间

点击知识库卡片进入详情页。

#### 5.2.4 知识库问答

1. 进入知识库详情页
2. 在 **Ask This Knowledge Base** 面板的文本框中输入问题
3. 点击 **Ask Knowledge Base** 按钮
4. 查看回答结果：
   - **Latest Answer**：回答文本
   - **状态标签**：`answered`（已回答）或 `refused`（拒答）
   - **Citations**：引用列表，包含来源文件和片段摘要
   - **Trace ID**：运行追踪标识

#### 5.2.5 查看运行追踪

详情页底部 **Recent Traces** 面板展示最近 10 条追踪记录，包含：

- 提问时间
- 回答状态
- 问题和回答摘要

### 5.3 Tauri 桌面端使用

#### 5.3.1 启动应用

1. 启动 Knowledge QA 桌面应用
2. 应用自动检测本地 Python API 状态
3. 若后端未运行，显示引导画面并自动启动后端
4. 后端就绪后自动进入主界面

#### 5.3.2 创建知识库（桌面端）

1. 在首页左侧面板填写名称和描述
2. 点击 **Choose Documents** 按钮，使用原生文件选择器选择文档
   - 支持多选
   - 文件类型过滤：md、markdown、txt、rst、json、yaml、yml、html、htm、pdf
3. 选中的文件路径以标签形式展示
4. 点击 **Create Knowledge Base** 完成导入

> **与 Web 端的区别**：桌面端使用原生文件选择器获取本地文件路径，直接将路径传给后端 API，无需上传文件内容。

#### 5.3.3 知识库问答

操作方式与 Web 管理台一致，参见 [5.2.4](#524-知识库问答)。

### 5.4 HTTP API 直接调用

#### 5.4.1 健康检查

```bash
curl http://127.0.0.1:8000/api/health
```

#### 5.4.2 创建知识库（路径方式）

```bash
curl -X POST http://127.0.0.1:8000/api/knowledge-bases \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Atlas Demo KB",
    "description": "Demo knowledge base",
    "paths": ["examples/knowledge_qa_demo_data"]
  }'
```

#### 5.4.3 创建知识库（上传方式）

```bash
curl -X POST http://127.0.0.1:8000/api/knowledge-bases/upload \
  -F 'name=Atlas Upload KB' \
  -F 'description=Uploaded local docs' \
  -F 'files=@examples/knowledge_qa_demo_data/01_overview.md' \
  -F 'files=@examples/knowledge_qa_demo_data/02_architecture.md'
```

#### 5.4.4 列出知识库

```bash
curl http://127.0.0.1:8000/api/knowledge-bases
```

#### 5.4.5 获取单个知识库

```bash
curl http://127.0.0.1:8000/api/knowledge-bases/<kb_id>
```

#### 5.4.6 提问

```bash
curl -X POST http://127.0.0.1:8000/api/knowledge-bases/<kb_id>/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Which vector store does Atlas use for retrieval?"
  }'
```

返回示例：

```json
{
  "answer": "Atlas uses Qdrant as its vector store for retrieval...",
  "citations": [
    {
      "index": 1,
      "source": "02_architecture.md",
      "snippet": "Qdrant serves as the primary vector store supporting both dense and sparse vectors...",
      "chunk_id": "abc123"
    }
  ],
  "confidence": 0.85,
  "answered": true,
  "reason": null,
  "trace_id": "def456"
}
```

#### 5.4.7 获取运行追踪

```bash
curl "http://127.0.0.1:8000/api/traces?limit=10"
```

#### 5.4.8 API 文档

启动后端后访问交互式 API 文档：

```
http://127.0.0.1:8000/api/docs
```

### 5.5 Python SDK 调用

安装项目后，可在 Python 代码中直接使用服务层：

```python
from pathlib import Path
from hello_agents.apps.knowledge_qa.runtime import KnowledgeQARuntime

runtime = KnowledgeQARuntime()

# 导入知识库
ingest_service = runtime.build_ingest_service()
kb = ingest_service.ingest(
    "Project Docs",
    paths=[Path("docs"), Path("README.md")],
    description="Project knowledge base",
)
print(f"Created: {kb.kb_id}, docs={kb.document_count}, chunks={kb.chunk_count}")

# 提问
answer_service = runtime.build_answer_service()
result = answer_service.ask(
    "What does the context engine do?",
    kb_id=kb.kb_id,
)
print(f"Answer: {result.answer}")
print(f"Answered: {result.answered}")
print(f"Confidence: {result.confidence}")
for citation in result.citations:
    print(f"  [{citation.index}] {citation.source} - {citation.snippet}")
```

---

## 6. 配置说明

### 6.1 环境变量配置

所有配置通过环境变量或 `.env` 文件管理。复制模板开始：

```bash
cp config/knowledge_qa.example.env .env
```

### 6.2 必需配置

| 环境变量 | 说明 | 示例 |
|----------|------|------|
| `LLM_MODEL` | LLM 模型名称 | `gpt-4o-mini` |
| `OPENAI_API_KEY` | OpenAI API Key | `sk-...` |
| `QDRANT_URL` | Qdrant 服务地址 | `http://localhost:6333` |
| `EMBED_MODEL_TYPE` | 嵌入模型类型 | `dashscope` |
| `EMBED_MODEL_NAME` | 嵌入模型名称 | `text-embedding-v4` |
| `EMBED_API_KEY` | 嵌入模型 API Key | `sk-...` |
| `EMBED_BASE_URL` | 嵌入模型 API 地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |

### 6.3 LLM 配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `LLM_MODEL` | `gpt-4o-mini` | 模型名称 |
| `OPENAI_API_KEY` | — | OpenAI API Key（也作为 `LLM_API_KEY` 的回退） |
| `LLM_API_KEY` | — | LLM API Key（优先于 `OPENAI_API_KEY`） |
| `LLM_BASE_URL` | — | OpenAI 兼容 API 地址 |
| `LLM_PROVIDER` | `openai` | 提供商标识 |
| `LLM_TIMEOUT` | `30` | 请求超时（秒） |

使用本地模型示例（Ollama）：

```bash
LLM_MODEL=qwen2.5:14b
LLM_BASE_URL=http://localhost:11434/v1
LLM_PROVIDER=ollama
LLM_API_KEY=EMPTY
```

### 6.4 Qdrant 配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `QDRANT_URL` | — | Qdrant 服务地址（必需） |
| `QDRANT_API_KEY` | — | Qdrant API Key |
| `QDRANT_TIMEOUT` | `10` | 连接超时（秒） |
| `QDRANT_UPSERT_BATCH_SIZE` | `64` | 批量写入大小 |
| `QDRANT_WAIT_FOR_UPSERT` | `true` | 是否等待写入确认 |
| `RAG_RECREATE_COLLECTION_ON_SCHEMA_MISMATCH` | `false` | Schema 不兼容时是否重建集合 |

### 6.5 嵌入模型配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `EMBED_MODEL_TYPE` | `dashscope` | 嵌入模型类型 |
| `EMBED_MODEL_NAME` | — | 模型名称（必需） |
| `EMBED_API_KEY` | — | API Key（必需） |
| `EMBED_BASE_URL` | — | API 地址（必需） |
| `EMBED_TIMEOUT` | `30` | 请求超时（秒） |

### 6.6 RAG 配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `RAG_CHUNK_SIZE` | `800` | 文档切分 chunk 大小（字符） |
| `RAG_CHUNK_OVERLAP` | `120` | chunk 重叠字符数 |
| `RAG_COLLECTION` | `hello_agents_rag` | Qdrant 集合名称 |
| `RAG_TOP_K` | `5` | 检索返回的候选数量 |

### 6.7 知识库问答应用配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `KNOWLEDGE_QA_RETRIEVAL_TOP_K` | `5` | 检索阶段 top-k 候选数 |
| `KNOWLEDGE_QA_ANSWER_CONTEXT_TOP_K` | `4` | 回答上下文使用的 chunk 数 |
| `KNOWLEDGE_QA_MAX_CITATIONS` | `4` | 最大引用数量 |
| `KNOWLEDGE_QA_MIN_RETRIEVED_CHUNKS` | `1` | 最少检索 chunk 数（低于此值触发拒答） |
| `KNOWLEDGE_QA_ANSWER_MAX_TOKENS` | `512` | 回答最大 token 数 |
| `KNOWLEDGE_QA_STORE_PATH` | `.hello_agents/knowledge_bases.json` | 知识库元数据存储路径 |
| `KNOWLEDGE_QA_TRACE_PATH` | `.hello_agents/knowledge_qa_traces.jsonl` | 运行追踪存储路径 |
| `KNOWLEDGE_QA_UPLOAD_ROOT` | `.hello_agents/uploads` | 上传文件暂存目录 |
| `KNOWLEDGE_QA_CORS_ORIGINS` | 见下方默认值 | CORS 允许的前端来源 |

CORS 默认允许来源：

```
http://localhost:5173
http://127.0.0.1:5173
http://tauri.localhost
https://tauri.localhost
tauri://localhost
```

### 6.8 Reranker 配置

Reranker（重排器）用于对检索结果进行二次排序，提升相关性。系统提供两种实现：

| 实现方式 | 说明 | 依赖 |
|----------|------|------|
| **启发式重排**（默认） | 基于词法重叠、标题匹配、文件名匹配等多信号加权评分 | 无外部依赖 |
| **DashScope 模型重排** | 使用 qwen3-rerank 模型进行语义重排 | 需要 `RERANK_API_KEY` |

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `RERANK_API_KEY` | — | DashScope API Key（设置后启用模型重排，否则使用启发式） |
| `RERANK_MODEL_NAME` | `qwen3-rerank` | DashScope 重排模型名称 |
| `KNOWLEDGE_QA_RERANK_INSTRUCT` | `Given a web search query, retrieve relevant passages that answer the query.` | 重排指令模板 |

**工作流程**：

1. 检索阶段获取 `top_k * 3` 个候选 chunk
2. 按 `source_paths` 过滤
3. 去重
4. **Reranker 重排**（此步骤）
5. 引用文件名优先排列
6. 截断到 `top_k`

**降级策略**：当 DashScope API 调用失败（网络错误、配额不足、非 200 响应）时，自动回退到启发式重排结果。

### 6.9 桌面端配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `KNOWLEDGE_QA_DESKTOP_PYTHON` | 自动探测 | Python 可执行文件路径 |
| `KNOWLEDGE_QA_DESKTOP_PROJECT_ROOT` | 自动探测 | 项目根目录路径 |

### 6.9 Web 前端配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `VITE_KNOWLEDGE_QA_API_BASE_URL` | 空 | 后端 API 基础 URL（为空时使用 Vite 代理） |

### 6.10 数据存储路径

默认所有运行时数据存储在项目根目录的 `.hello_agents/` 下：

```
.hello_agents/
├── knowledge_bases.json          # 知识库元数据
├── knowledge_qa_traces.jsonl     # 运行追踪记录
├── uploads/                      # 上传文件暂存
│   └── <upload_id>/             # 每次上传的唯一目录
│       ├── 001_filename.md
│       └── 002_another.txt
└── logs/                         # 桌面端后端日志
    └── knowledge_qa_desktop_backend.log
```

---

## 7. 故障排除

### 7.1 服务端问题

#### 问题：API 启动报错 "Knowledge QA requires QDRANT_URL to be configured"

**原因**：未配置 Qdrant 连接信息。

**解决**：

1. 确认 `.env` 文件中已设置 `QDRANT_URL`
2. 确认 Qdrant 服务正在运行：

```bash
curl http://localhost:6333/collections
```

#### 问题：API 启动报错 "Knowledge QA requires embedding configuration"

**原因**：嵌入模型配置不完整。

**解决**：确认 `.env` 文件中同时设置了以下三个变量：

```bash
EMBED_MODEL_NAME=text-embedding-v4
EMBED_API_KEY=your-api-key
EMBED_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

#### 问题：健康检查返回 `qdrant_configured: false` 或 `embedding_configured: false`

**原因**：对应的环境变量未设置或 Qdrant 服务不可达。

**解决**：

1. 检查 `.env` 文件中 `QDRANT_URL` 和 `EMBED_*` 变量
2. 确认 Qdrant 服务正在运行
3. 重启 API 服务

#### 问题：导入知识库时报错 500

**可能原因**：

1. Qdrant 连接超时
2. 嵌入模型 API 调用失败
3. 文档格式不支持

**排查步骤**：

```bash
# 1. 检查 Qdrant 连通性
curl http://localhost:6333/collections

# 2. 检查 API 健康状态
curl http://127.0.0.1:8000/api/health

# 3. 查看后端日志
# 开发模式：直接查看终端输出
# 生产模式：查看应用日志
```

#### 问题：问答返回 "I do not know based on the current knowledge base."

**可能原因**：

1. 知识库中确实没有相关内容
2. `KNOWLEDGE_QA_MIN_RETRIEVED_CHUNKS` 设置过高
3. 检索 top-k 过低，相关 chunk 未被召回

**解决**：

1. 尝试调整检索参数：

```bash
KNOWLEDGE_QA_RETRIEVAL_TOP_K=10
KNOWLEDGE_QA_MIN_RETRIEVED_CHUNKS=1
```

2. 检查知识库状态是否为 `ready`
3. 查看追踪记录了解检索详情

### 7.2 Web 管理台问题

#### 问题：页面加载但显示 "Request failed with status 502"

**原因**：后端 API 未启动或不可达。

**解决**：

1. 确认后端正在运行：`curl http://127.0.0.1:8000/api/health`
2. 检查 Vite 代理配置是否正确指向后端地址

#### 问题：文件上传后知识库状态为 `failed`

**原因**：上传的文件可能格式不支持或内容为空。

**解决**：

1. 确认文件格式在支持列表中
2. 检查后端日志获取具体错误信息
3. 尝试用 CLI 方式导入相同文件以获取更详细的错误输出

#### 问题：CORS 错误

**原因**：前端请求来源不在后端 CORS 允许列表中。

**解决**：

在 `.env` 中添加前端来源：

```bash
KNOWLEDGE_QA_CORS_ORIGINS=http://localhost:5173,http://your-custom-origin:3000
```

### 7.3 Tauri 桌面端问题

#### 问题：启动后停留在引导画面，显示超时错误

**原因**：桌面端无法启动本地 Python API。

**解决**：

1. 检查日志文件：

```bash
cat .hello_agents/logs/knowledge_qa_desktop_backend.log
```

2. 手动指定 Python 路径：

```bash
export KNOWLEDGE_QA_DESKTOP_PYTHON=/path/to/your/.venv/bin/python
```

3. 手动指定项目根目录：

```bash
export KNOWLEDGE_QA_DESKTOP_PROJECT_ROOT=/path/to/hello-agents
```

4. 确认 `.env` 文件存在于项目根目录

#### 问题：桌面端文件选择器无法打开

**原因**：Tauri dialog 权限未正确配置。

**解决**：

确认 [capabilities/default.json](../frontend/knowledge-qa/src-tauri/capabilities/default.json) 包含 `dialog:default` 权限：

```json
{
  "identifier": "default",
  "windows": ["main"],
  "permissions": ["core:default", "dialog:default"]
}
```

#### 问题：桌面端创建知识库时报错 "Please provide at least one source path"

**原因**：桌面端通过路径方式导入，后端需要能访问这些路径。

**解决**：

1. 确认选择的文件路径在本机可访问
2. 确认后端进程的工作目录正确（应指向项目根目录）

### 7.4 Qdrant 问题

#### 问题：Qdrant 连接超时

**解决**：

1. 确认 Qdrant 正在运行：`curl http://localhost:6333`
2. 增加超时时间：`QDRANT_TIMEOUT=30`
3. 检查网络和防火墙设置

#### 问题：集合 Schema 不兼容

**原因**：Qdrant 集合的向量维度或配置与当前嵌入模型不匹配。

**解决**：

方式一：删除旧集合并重新导入

```bash
curl -X DELETE http://localhost:6333/collections/hello_agents_rag
```

方式二：启用自动重建

```bash
RAG_RECREATE_COLLECTION_ON_SCHEMA_MISMATCH=true
```

### 7.5 嵌入模型问题

#### 问题：嵌入 API 调用失败

**可能原因**：API Key 无效、配额耗尽、网络问题。

**解决**：

1. 验证 API Key 有效性
2. 检查 API 配额和余额
3. 确认 `EMBED_BASE_URL` 可达：

```bash
curl -I https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 7.6 通用诊断方法

#### 检查健康状态

```bash
curl -s http://127.0.0.1:8000/api/health | python3 -m json.tool
```

关注以下字段：

- `qdrant_configured`：应为 `true`
- `embedding_configured`：应为 `true`
- `status`：应为 `ok`

#### 查看运行追踪

```bash
hello-agents-knowledge-qa inspect --traces --limit 5
```

追踪中的关键字段：

| 字段 | 诊断价值 |
|------|----------|
| `answered` | 是否成功回答 |
| `reason` | 拒答原因（`no_relevant_context`、`insufficient_evidence`、`empty_answer`） |
| `latency_ms` | 响应延迟 |
| `token_usage` | Token 消耗 |
| `retrieved_chunks` | 检索到的 chunk 数量 |
| `evidence_score` | 证据评分 |

#### 检查数据文件

```bash
# 知识库元数据
cat .hello_agents/knowledge_bases.json | python3 -m json.tool

# 最近追踪
tail -5 .hello_agents/knowledge_qa_traces.jsonl | python3 -m json.tool
```

---

## 附录 A：API 路由速查

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 健康检查 |
| `GET` | `/api/knowledge-bases` | 列出所有知识库 |
| `GET` | `/api/knowledge-bases/{kb_id}` | 获取单个知识库 |
| `POST` | `/api/knowledge-bases` | 通过路径创建知识库 |
| `POST` | `/api/knowledge-bases/upload` | 通过文件上传创建知识库 |
| `POST` | `/api/knowledge-bases/{kb_id}/ask` | 对知识库提问 |
| `GET` | `/api/traces?limit=10` | 获取最近追踪 |

## 附录 B：CLI 命令速查

| 命令 | 说明 |
|------|------|
| `hello-agents-knowledge-qa ingest --name NAME --paths PATHS [--description DESC]` | 导入知识库 |
| `hello-agents-knowledge-qa ask --question Q [--kb-id ID]` | 提问 |
| `hello-agents-knowledge-qa inspect` | 列出知识库 |
| `hello-agents-knowledge-qa inspect --traces [--limit N]` | 查看追踪 |

## 附录 C：项目目录结构

```
hello-agents/
├── .env                              # 环境变量配置（从模板复制）
├── .python-version                   # Python 版本锁定
├── pyproject.toml                    # Python 项目配置
├── config/
│   └── knowledge_qa.example.env      # 环境变量模板
├── docs/                             # 设计文档
├── examples/
│   ├── knowledge_qa_demo_data/       # 演示数据
│   └── knowledge_qa_cli.py           # CLI 兼容包装
├── frontend/knowledge-qa/
│   ├── package.json                  # 前端依赖
│   ├── vite.config.js                # Vite 配置
│   ├── src/                          # React 前端源码
│   │   ├── App.jsx                   # 应用入口（含桌面端引导逻辑）
│   │   ├── api.js                    # API 调用封装
│   │   ├── desktopBackend.js         # 桌面端后端管理
│   │   ├── runtime.js                # 运行时环境检测
│   │   └── routes/                   # 页面组件
│   └── src-tauri/                    # Tauri 桌面端
│       ├── Cargo.toml                # Rust 依赖
│       ├── tauri.conf.json           # Tauri 配置
│       ├── capabilities/default.json # 权限声明
│       └── src/lib.rs                # Rust 后端逻辑
├── src/hello_agents/
│   ├── apps/knowledge_qa/            # 知识库问答应用
│   │   ├── api.py                    # FastAPI 路由
│   │   ├── api_schemas.py            # Pydantic 响应模型
│   │   ├── answer.py                 # 回答生成与引用
│   │   ├── cli.py                    # CLI 入口
│   │   ├── config.py                 # 应用配置
│   │   ├── ingest.py                 # 文档导入
│   │   ├── models.py                 # 领域模型
│   │   ├── retrieve.py               # 检索逻辑
│   │   ├── runtime.py                # 运行时组装
│   │   ├── service.py                # 服务层
│   │   ├── store.py                  # 知识库持久化
│   │   ├── trace.py                  # 运行追踪
│   │   └── uploads.py                # 上传文件管理
│   ├── llm/                          # LLM 客户端
│   ├── memory/                       # 记忆系统
│   ├── rag/                          # RAG 子系统
│   └── tools/                        # 工具系统
└── tests/                            # 测试
```
