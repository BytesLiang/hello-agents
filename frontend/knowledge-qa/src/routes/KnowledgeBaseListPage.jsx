import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import {
  createKnowledgeBase,
  listKnowledgeBases,
  uploadKnowledgeBase,
} from "../api";
import { STATUS_LABELS, formatDate } from "../i18n";
import {
  EmptyState,
  ErrorMessage,
  LoadingIndicator,
  MetricChip,
  StatusPill,
  SuccessMessage,
} from "../components";
import { isTauriRuntime, pickDesktopDocuments } from "../runtime";

const EMPTY_FORM = {
  name: "",
  description: "",
};

const ACCEPT_EXTENSIONS = ".md,.markdown,.txt,.rst,.json,.yaml,.yml,.html,.htm,.pdf";

export function KnowledgeBaseListPage() {
  const [knowledgeBases, setKnowledgeBases] = useState([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [form, setForm] = useState(EMPTY_FORM);
  const [desktopMode] = useState(() => isTauriRuntime());
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [selectedPaths, setSelectedPaths] = useState([]);
  const [fileInputKey, setFileInputKey] = useState(0);

  useEffect(() => {
    let active = true;

    async function loadKnowledgeBases() {
      setLoading(true);
      setError("");
      try {
        const payload = await listKnowledgeBases();
        if (active) {
          setKnowledgeBases(payload);
        }
      } catch (fetchError) {
        if (active) {
          setError(fetchError.message);
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    void loadKnowledgeBases();
    return () => {
      active = false;
    };
  }, []);

  const updateFormField = useCallback((field, value) => {
    setForm((current) => ({ ...current, [field]: value }));
  }, []);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!form.name.trim()) {
      setError("请输入知识库名称。");
      setSuccessMessage("");
      return;
    }
    if (desktopMode && selectedPaths.length === 0) {
      setError("请至少选择一个本地文档路径。");
      setSuccessMessage("");
      return;
    }
    if (!desktopMode && selectedFiles.length === 0) {
      setError("请至少选择一个本地文档。");
      setSuccessMessage("");
      return;
    }

    setSubmitting(true);
    setError("");
    setSuccessMessage("");
    try {
      const created = desktopMode
        ? await createKnowledgeBase({
            name: form.name.trim(),
            description: form.description.trim(),
            paths: selectedPaths,
          })
        : await uploadKnowledgeBase({
            name: form.name.trim(),
            description: form.description.trim(),
            files: selectedFiles,
          });
      const refreshed = await listKnowledgeBases();
      setKnowledgeBases(refreshed);
      setForm(EMPTY_FORM);
      setSelectedFiles([]);
      setSelectedPaths([]);
      setFileInputKey((current) => current + 1);
      setSuccessMessage(`知识库「${created.name}」导入成功。`);
    } catch (submitError) {
      setError(submitError.message);
    } finally {
      setSubmitting(false);
    }
  }

  async function handleChooseDesktopDocuments() {
    setError("");
    try {
      const paths = await pickDesktopDocuments();
      setSelectedPaths(paths);
    } catch (pickError) {
      setError(
        pickError instanceof Error
          ? pickError.message
          : "无法打开本地文件选择器。"
      );
    }
  }

  return (
    <main className="page-shell" id="main-content">
      <section className="hero-panel" aria-labelledby="hero-title">
        <h1 id="hero-title">知识库问答控制台</h1>
        <p className="hero-copy">
          导入本地文档建立知识库，即可进行智能问答。
        </p>
      </section>

      <section className="layout-grid" aria-label="知识库管理区域">
        <article className="panel panel-form" aria-labelledby="create-heading">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">导入</p>
              <h2 id="create-heading">创建知识库</h2>
            </div>
            <span className="status-pill status-pending">同步导入</span>
          </div>

          <form className="stack" onSubmit={handleSubmit} noValidate>
            <label className="field" htmlFor="kb-name">
              <span>名称</span>
              <input
                id="kb-name"
                name="name"
                placeholder="项目文档库"
                required
                value={form.name}
                onChange={(event) => updateFormField("name", event.target.value)}
              />
            </label>

            <label className="field" htmlFor="kb-description">
              <span>描述</span>
              <input
                id="kb-description"
                name="description"
                placeholder="产品与工程知识库"
                value={form.description}
                onChange={(event) => updateFormField("description", event.target.value)}
              />
            </label>

            {desktopMode ? (
              <>
                <div className="field">
                  <span id="kb-docs-label">本地文档</span>
                  <button
                    aria-labelledby="kb-docs-label"
                    className="secondary-button"
                    onClick={handleChooseDesktopDocuments}
                    type="button"
                  >
                    选择文档
                  </button>
                </div>

                {selectedPaths.length ? (
                  <ul className="file-list" aria-label="已选路径">
                    {selectedPaths.map((path) => (
                      <li className="file-chip" key={path}>
                        {path}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="message muted-message compact-message">
                    点击上方按钮选择本地文件。
                  </p>
                )}
              </>
            ) : (
              <>
                <label className="field" htmlFor="kb-files">
                  <span>本地文档</span>
                  <input
                    id="kb-files"
                    key={fileInputKey}
                    accept={ACCEPT_EXTENSIONS}
                    multiple
                    name="files"
                    type="file"
                    onChange={(event) =>
                      setSelectedFiles(Array.from(event.target.files ?? []))
                    }
                  />
                </label>

                {selectedFiles.length ? (
                  <ul className="file-list" aria-label="已选文件">
                    {selectedFiles.map((file) => (
                      <li className="file-chip" key={`${file.name}-${file.size}`}>
                        {file.name}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="message muted-message compact-message">
                    选择一个或多个本地文件进行导入。
                  </p>
                )}
              </>
            )}


            <button
              aria-busy={submitting}
              className="primary-button"
              disabled={submitting}
              type="submit"
            >
              {submitting ? "导入中…" : "创建知识库"}
            </button>
          </form>

          <ErrorMessage>{error}</ErrorMessage>
          <SuccessMessage>{successMessage}</SuccessMessage>
        </article>

        <article className="panel panel-list" aria-labelledby="list-heading">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">清单</p>
              <h2 id="list-heading">知识库列表</h2>
            </div>
            <MetricChip>共 {knowledgeBases.length} 个</MetricChip>
          </div>

          {loading ? (
            <LoadingIndicator text="正在加载知识库..." />
          ) : null}

          {!loading && !error && knowledgeBases.length === 0 ? (
            <EmptyState>
              <p className="message muted-message">
                暂无知识库，请在左侧创建并导入。
              </p>
            </EmptyState>
          ) : null}

          <div className="card-list" role="list">
            {knowledgeBases.map((knowledgeBase) => (
              <div key={knowledgeBase.kb_id} role="listitem">
                <Link
                  aria-label={`${knowledgeBase.name} — ${knowledgeBase.description || "暂无描述"}`}
                  className="kb-card"
                  to={`/knowledge-bases/${knowledgeBase.kb_id}`}
                >
                <div className="kb-card-header">
                  <div>
                    <h3>{knowledgeBase.name}</h3>
                    <p>{knowledgeBase.description || "暂无描述"}</p>
                  </div>
                  <StatusPill status={STATUS_LABELS[knowledgeBase.status] || knowledgeBase.status} />
                </div>
                <dl className="stat-grid">
                  <div>
                    <dt>文档数</dt>
                    <dd>{knowledgeBase.document_count}</dd>
                  </div>
                  <div>
                    <dt>分块数</dt>
                    <dd>{knowledgeBase.chunk_count}</dd>
                  </div>
                </dl>
                <div className="card-footer">
                  <span>{formatDate(knowledgeBase.updated_at)}</span>
                  <span aria-hidden="true">进入工作区 →</span>
                </div>
              </Link>
              </div>
            ))}
          </div>
        </article>
      </section>
    </main>
  );
}
